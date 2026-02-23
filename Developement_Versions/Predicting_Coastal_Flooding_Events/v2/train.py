import pandas as pd
import numpy as np
import scipy.io as sio
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, confusion_matrix, roc_auc_score
from datetime import datetime, timedelta
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# 參數設定
HIST_WINDOW = 7
PRED_WINDOW = 14

# 官方指定的訓練與測試站點
TRAINING_STATIONS = [
    'Annapolis', 'Atlantic_City', 'Charleston', 'Washington', 'Wilmington', 
    'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook'
]
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

class FloodModel:
    def __init__(self):
        self.models = []
        self.threshold_map = {}

    def load_data(self, path_dataset, path_thresholds):
        print(f"Loading data from: {path_dataset}")
        try:
            data = sio.loadmat(path_dataset)
        except FileNotFoundError:
            print(f"❌ 找不到 {path_dataset}，請確認檔案路徑！")
            return None

        lat = data['lattg'].flatten()
        time = data['t'].flatten()
        sea_level = data['sltg']
        
        try:
            station_names = [s.item() for s in data['sname'].flatten()]
        except:
            station_names = [s[0][0] for s in data['sname']]

        def matlab2datetime(matlab_datenum):
            day = datetime.fromordinal(int(matlab_datenum))
            dayfrac = timedelta(days=matlab_datenum % 1) - timedelta(days=366)
            return day + dayfrac
        time_dt = [matlab2datetime(t) for t in time]

        records = []
        for i, name in enumerate(station_names):
            records.append(pd.DataFrame({
                'time': time_dt, 'station_name': name, 'sea_level': sea_level[:, i]
            }))
        df_hourly = pd.concat(records, ignore_index=True)
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])

        thresh_data = sio.loadmat(path_thresholds)
        try:
            th_names = [s.item() for s in thresh_data['sname'].flatten()]
        except:
            th_names = [s[0][0] for s in thresh_data['sname']]
        th_vals = thresh_data['thminor_stnd'].flatten()
        self.threshold_map = dict(zip(th_names, th_vals))

        df_hourly['flood_threshold'] = df_hourly['station_name'].map(self.threshold_map)
        
        print("Aggregating to daily data...")
        df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
            'sea_level': ['mean', 'max'], 
            'flood_threshold': 'first'
        }).reset_index()
        df_daily.columns = ['station_name', 'time', 'sl_mean', 'sl_max', 'threshold']

        print("Generating features...")
        df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['year_norm'] = (df_daily['time'].dt.year - 1950) / 70
        df_daily['dist_to_threshold'] = df_daily['sl_max'] - df_daily['threshold']

        for window in [3, 7, 30]:
            grp = df_daily.groupby('station_name')['sl_max']
            df_daily[f'max_mean_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
            df_daily[f'max_max_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).max())
            df_daily[f'max_std_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).std())
        for window in [3, 7]:
            grp = df_daily.groupby('station_name')['dist_to_threshold']
            df_daily[f'dist_mean_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())

        return df_daily.fillna(0)

    def create_features(self, df, stations, hist_days=7, future_days=14):
        # 定義特徵欄位 (必須與 model.py 一致)
        feature_cols = ['sl_mean', 'sl_max', 'month_sin', 'month_cos', 'year_norm', 'threshold',
                        'dist_to_threshold', 'dist_mean_3d', 'dist_mean_7d',
                        'max_mean_3d', 'max_max_3d', 'max_std_3d',
                        'max_mean_7d', 'max_max_7d', 'max_std_7d',
                        'max_mean_30d', 'max_max_30d', 'max_std_30d']
        
        df_subset = df[df['station_name'].isin(stations)].copy()
        X_list, y_list = [], []
        
        for stn, group in df_subset.groupby('station_name'):
            group = group.sort_values('time').reset_index(drop=True)
            feats = []
            for i in range(hist_days):
                shifted = group[feature_cols].shift(i)
                shifted.columns = [f"{c}_lag{i}" for c in feature_cols]
                feats.append(shifted)
            X_stn = pd.concat(feats, axis=1)
            
            targets = []
            for i in range(future_days):
                # 預測目標：距離門檻還有多少 (正數代表淹水)
                target = group['dist_to_threshold'].shift(-(1 + i))
                targets.append(target)
            y_stn = pd.concat(targets, axis=1)
            
            valid_mask = ~(X_stn.isna().any(axis=1) | y_stn.isna().any(axis=1))
            X_list.append(X_stn[valid_mask])
            y_list.append(y_stn[valid_mask])
            
        return pd.concat(X_list, axis=0).values, pd.concat(y_list, axis=0).values

    def train_and_evaluate(self, df_daily, save_path='model.pkl'):
        print("\n=== 開始訓練與評估 ===")
        
        # 1. 準備訓練數據 (Training Set)
        print(f"準備訓練集 ({len(TRAINING_STATIONS)} stations)...")
        X_train, y_train = self.create_features(df_daily, TRAINING_STATIONS, HIST_WINDOW, PRED_WINDOW)
        
        # 2. 準備驗證數據 (Validation Set - 比賽的 Out-of-Distribution 測試集)
        print(f"準備驗證集 ({len(TESTING_STATIONS)} stations)...")
        X_val, y_val = self.create_features(df_daily, TESTING_STATIONS, HIST_WINDOW, PRED_WINDOW)
        
        # 使用 Regression (預測數值)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1, 
            'n_jobs': -1, 
            'random_state': 42
        }

        self.models = []
        
        # 用來存所有預測結果以計算總分
        val_preds_dist = [] # 預測的距離
        val_true_binary = [] # 真實是否淹水 (0/1)

        print("Training models for 14 days...")
        for d in range(PRED_WINDOW):
            d_train = lgb.Dataset(X_train, label=y_train[:, d])
            bst = lgb.train(params, d_train, num_boost_round=500)
            self.models.append(bst)
            
            # 預測驗證集
            preds = bst.predict(X_val)
            true_dist = y_val[:, d]
            
            # 收集結果
            val_preds_dist.extend(preds)
            val_true_binary.extend((true_dist > 0).astype(int))
        
        val_preds_dist = np.array(val_preds_dist)
        val_true_binary = np.array(val_true_binary)

        # === 自動尋找最佳門檻值 (Best Threshold) ===
        print("\n正在尋找最佳分類門檻 (Best Threshold)...")
        best_f1 = 0
        best_th = 0
        # 測試從 -0.5 到 0.5 的門檻
        for th in np.linspace(-0.5, 0.5, 101):
            preds_bin = (val_preds_dist > th).astype(int)
            f1 = f1_score(val_true_binary, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        
        print(f"\n🏆 最佳門檻值: {best_th:.4f}")
        
        # 使用最佳門檻計算最終指標
        final_preds = (val_preds_dist > best_th).astype(int)
        
        acc = accuracy_score(val_true_binary, final_preds)
        f1 = f1_score(val_true_binary, final_preds, zero_division=0)
        mcc = matthews_corrcoef(val_true_binary, final_preds)
        try:
            auc = roc_auc_score(val_true_binary, val_preds_dist)
        except:
            auc = 0.5
        tn, fp, fn, tp = confusion_matrix(val_true_binary, final_preds).ravel()

        print("\n=== 📊 本地驗證結果 ===")
        print(f"Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
        print(f"AUC:       {auc:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"MCC:       {mcc:.4f}")
        print("==========================")
        
        # 儲存模型
        with open(save_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"\nModel saved to {save_path}")
        print(f"⚠️ 請記得將 BEST_THRESHOLD = {best_th:.4f} 更新到你的 model.py 中！")

if __name__ == "__main__":
    # 修改為你的檔案路徑
    DATA_PATH = 'NEUSTG_19502020_12stations.mat'
    THRESH_PATH = 'Seed_Coastal_Stations_Thresholds.mat'
    
    pipeline = FloodModel()
    df = pipeline.load_data(DATA_PATH, THRESH_PATH)
    if df is not None:
        pipeline.train_and_evaluate(df, save_path='model.pkl')