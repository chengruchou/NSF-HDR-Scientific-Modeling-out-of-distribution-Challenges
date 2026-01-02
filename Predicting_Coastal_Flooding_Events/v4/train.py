# ==========================================
# 檔名：train_final.py (全量訓練_無風險版)
# ==========================================
import pandas as pd
import numpy as np
import scipy.io as sio
import lightgbm as lgb
from sklearn.metrics import f1_score
from datetime import datetime, timedelta
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

HIST_WINDOW = 7
PRED_WINDOW = 14

# 使用全部 12 個站點進行訓練，讓模型看過各種地形
ALL_STATIONS = [
    'Annapolis', 'Atlantic_City', 'Charleston', 'Washington', 'Wilmington', 
    'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook',
    'Lewes', 'Fernandina_Beach', 'The_Battery'
]

class FloodModel:
    def __init__(self):
        self.models = []
        self.threshold_map = {}

    def load_data(self, path_dataset, path_thresholds):
        print(f"Loading data from: {path_dataset}")
        try:
            data = sio.loadmat(path_dataset)
        except: return None

        lat = data['lattg'].flatten()
        time = data['t'].flatten()
        sea_level = data['sltg']
        try: station_names = [s.item() for s in data['sname'].flatten()]
        except: station_names = [s[0][0] for s in data['sname']]
        
        def matlab2datetime(t):
            return datetime.fromordinal(int(t)) + timedelta(days=t%1) - timedelta(days=366)
        time_dt = [matlab2datetime(t) for t in time]

        records = []
        for i, name in enumerate(station_names):
            records.append(pd.DataFrame({
                'time': time_dt, 'station_name': name, 'sea_level': sea_level[:, i]
            }))
        df_hourly = pd.concat(records, ignore_index=True)
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])

        thresh_data = sio.loadmat(path_thresholds)
        try: th_names = [s.item() for s in thresh_data['sname'].flatten()]
        except: th_names = [s[0][0] for s in thresh_data['sname']]
        th_vals = thresh_data['thminor_stnd'].flatten()
        self.threshold_map = dict(zip(th_names, th_vals))
        
        df_hourly['flood_threshold'] = df_hourly['station_name'].map(self.threshold_map)
        
        print("Aggregating...")
        df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
            'sea_level': ['mean', 'max'], 
            'flood_threshold': 'first'
        }).reset_index()
        df_daily.columns = ['station_name', 'time', 'sl_mean', 'sl_max', 'threshold']

        print("Features...")
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

    def create_features(self, df):
        # 移除有風險的空間特徵，只保留穩健的統計特徵
        feature_cols = ['sl_mean', 'sl_max', 'month_sin', 'month_cos', 'year_norm', 'threshold',
                        'dist_to_threshold', 'dist_mean_3d', 'dist_mean_7d',
                        'max_mean_3d', 'max_max_3d', 'max_std_3d',
                        'max_mean_7d', 'max_max_7d', 'max_std_7d',
                        'max_mean_30d', 'max_max_30d', 'max_std_30d']
        
        X_list, y_list, times = [], [], []
        
        for stn, group in df[df['station_name'].isin(ALL_STATIONS)].groupby('station_name'):
            group = group.sort_values('time').reset_index(drop=True)
            feats = []
            for i in range(HIST_WINDOW):
                shifted = group[feature_cols].shift(i)
                shifted.columns = [f"{c}_lag{i}" for c in feature_cols]
                feats.append(shifted)
            X_stn = pd.concat(feats, axis=1)
            
            targets = []
            for i in range(PRED_WINDOW):
                target = group['dist_to_threshold'].shift(-(1 + i))
                targets.append(target)
            y_stn = pd.concat(targets, axis=1)
            
            valid_mask = ~(X_stn.isna().any(axis=1) | y_stn.isna().any(axis=1))
            X_list.append(X_stn[valid_mask])
            y_list.append(y_stn[valid_mask])
            times.append(group.loc[valid_mask, 'time'])
            
        return pd.concat(X_list, axis=0).values, pd.concat(y_list, axis=0).values, pd.concat(times, axis=0)

    def train_and_evaluate(self, df, save_path='model.pkl'):
        X, y, t = self.create_features(df)
        
        # 時間切分 (Time Split)
        split_date = pd.to_datetime('2016-01-01')
        train_mask = t < split_date
        val_mask = t >= split_date
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        params = {
            'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05,
            'num_leaves': 40, 'verbose': -1, 'n_jobs': -1, 'random_state': 42
        }

        self.models = []
        val_preds = []
        val_true = []

        print("Training on Full History (1950-2015)...")
        for d in range(PRED_WINDOW):
            d_train = lgb.Dataset(X_train, label=y_train[:, d])
            # 全量數據比較大，訓練輪數設為 1000
            bst = lgb.train(params, d_train, num_boost_round=1000)
            self.models.append(bst)
            
            p = bst.predict(X_val)
            val_preds.extend(p)
            val_true.extend((y_val[:, d] > 0).astype(int))
            
        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        
        print("Finding Best Threshold on Validation Set (2016-2020)...")
        best_f1 = 0
        best_th = 0
        for th in np.linspace(-0.5, 0.5, 100):
            p_bin = (val_preds > th).astype(int)
            f1 = f1_score(val_true, p_bin)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        
        print(f"\n🏆 Best Threshold: {best_th:.4f}")
        print(f"Validation F1: {best_f1:.4f}")
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    DATA_PATH = 'NEUSTG_19502020_12stations.mat'
    THRESH_PATH = 'Seed_Coastal_Stations_Thresholds.mat'
    if os.path.exists(DATA_PATH):
        pipeline = FloodModel()
        df = pipeline.load_data(DATA_PATH, THRESH_PATH)
        pipeline.train_and_evaluate(df, save_path='model.pkl')