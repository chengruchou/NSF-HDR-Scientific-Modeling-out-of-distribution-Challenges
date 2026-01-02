import pandas as pd
import numpy as np
import scipy.io as sio
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from datetime import datetime, timedelta
import pickle
import warnings
import os
import sys

# 忽略警告
warnings.filterwarnings('ignore')

# ============================================
# 設定與參數
# ============================================
HIST_WINDOW = 7
PRED_WINDOW = 14
TRAINING_STATIONS = [
    'Annapolis', 'Atlantic_City', 'Charleston', 'Washington', 'Wilmington', 
    'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook'
]
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

class FloodModel:
    def __init__(self):
        self.models = []

    def _matlab2datetime(self, matlab_datenum):
        day = datetime.fromordinal(int(matlab_datenum))
        dayfrac = timedelta(days=matlab_datenum % 1) - timedelta(days=366)
        return day + dayfrac

    def _parse_mat_strings(self, mat_array):
        flat_array = mat_array.flatten()
        parsed_strings = []
        for item in flat_array:
            if isinstance(item, np.ndarray) and item.size > 0:
                parsed_strings.append(str(item.item()))
            elif isinstance(item, str):
                parsed_strings.append(item)
            else:
                try:
                    parsed_strings.append(str(item[0]))
                except:
                    parsed_strings.append(str(item))
        return parsed_strings

    def load_and_preprocess(self, path_dataset, path_thresholds):
        print(f"Loading data from: {path_dataset}")
        # 1. Load Data
        data = sio.loadmat(path_dataset)
        lat = data['lattg'].flatten()
        lon = data['lontg'].flatten()
        time = data['t'].flatten()
        sea_level = data['sltg']
        station_names = self._parse_mat_strings(data['sname'])
        
        time_dt = [self._matlab2datetime(t) for t in time]
        
        records = []
        for i, name in enumerate(station_names):
            records.append(pd.DataFrame({
                'time': time_dt,
                'station_name': name,
                'latitude': lat[i],
                'longitude': lon[i],
                'sea_level': sea_level[:, i]
            }))
        df_hourly = pd.concat(records, ignore_index=True)
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])

        # 2. Load Thresholds
        thresh_data = sio.loadmat(path_thresholds)
        th_names = self._parse_mat_strings(thresh_data['sname'])
        th_vals = thresh_data['thminor_stnd'].flatten()
        threshold_map = dict(zip(th_names, th_vals))
        
        df_hourly['flood_threshold'] = df_hourly['station_name'].map(threshold_map)
        df_hourly['is_flood'] = (df_hourly['sea_level'] > df_hourly['flood_threshold']).astype(int)

        # 3. Daily Aggregation
        df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
            'sea_level': 'mean',
            'is_flood': 'max',
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()

        # Feature Engineering
        df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['year_norm'] = (df_daily['time'].dt.year - 1950) / 70

        for window in [3, 7]:
            grp = df_daily.groupby('station_name')['sea_level']
            df_daily[f'sl_mean_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
            df_daily[f'sl_max_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).max())
            df_daily[f'sl_std_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).std())
        
        return df_daily.fillna(0)

    def create_dataset_vectorized(self, df, stations, hist_days=7, future_days=14, feature_cols=None):
        """
        使用向量化操作極速建立訓練數據，取代原本慢速的 for loop
        """
        print("Creating dataset (Vectorized)...")
        df_subset = df[df['station_name'].isin(stations)].copy()
        
        # 為了保持時間連續性，我們分站點處理
        X_list, y_list = [], []
        
        for stn, group in df_subset.groupby('station_name'):
            group = group.sort_values('time').reset_index(drop=True)
            
            # 使用 shift 建立歷史特徵
            # 例如: t-0, t-1, ... t-6
            feats = []
            for i in range(hist_days):
                # 這裡簡單將 features flatten: 每天的特徵平鋪
                # 但為了 LightGBM，我們通常直接用當前特徵即可，
                # 這裡為了維持架構，我們取 'hist_days' 天前的特徵矩陣
                shifted = group[feature_cols].shift(i)
                shifted.columns = [f"{c}_lag{i}" for c in feature_cols]
                feats.append(shifted)
            
            X_stn = pd.concat(feats, axis=1)
            
            # 建立未來標籤: t+7, t+8, ... t+20
            targets = []
            for i in range(future_days):
                # 未來第 i+1 天 (從 hist_days 開始算)
                target = group['is_flood'].shift(-(hist_days + i))
                targets.append(target)
            
            y_stn = pd.concat(targets, axis=1)
            
            # 移除因為 shift 產生的 NaN
            valid_mask = ~(X_stn.isna().any(axis=1) | y_stn.isna().any(axis=1))
            
            X_list.append(X_stn[valid_mask])
            y_list.append(y_stn[valid_mask])
            
        X = pd.concat(X_list, axis=0).values
        y = pd.concat(y_list, axis=0).values
        
        return X, y

    def train(self, df_daily, save_path='model.pkl'):
        print("Training LightGBM models...")
        feat_cols = ['sea_level', 'month_sin', 'month_cos', 'year_norm', 
                     'sl_mean_3d', 'sl_max_3d', 'sl_mean_7d', 'sl_max_7d', 'sl_std_7d']
        
        # 使用加速版的資料生成
        X_train, y_train = self.create_dataset_vectorized(df_daily, TRAINING_STATIONS, HIST_WINDOW, PRED_WINDOW, feat_cols)
        
        print(f"Training Data Shape: {X_train.shape}")
        
        # 計算權重
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1
        
        self.models = []
        for d in range(PRED_WINDOW):
            # 使用 LightGBM
            clf = LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_weight,
                n_jobs=-1, # 使用所有 CPU 核心
                random_state=42,
                verbose=-1
            )
            clf.fit(X_train, y_train[:, d])
            self.models.append(clf)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"Model saved to {save_path}")

    def evaluate(self, df_daily, intervals_file=None, load_model_path='model.pkl'):
        if not self.models and os.path.exists(load_model_path):
            with open(load_model_path, 'rb') as f:
                self.models = pickle.load(f)
        
        if not self.models:
            print("No model found.")
            return

        # Fallback intervals
        intervals_data = {
            'start_date': ['3/6/1962', '7/21/2013', '5/13/2011', '12/21/1995', '9/5/1995',
                           '12/31/2009', '9/16/2020', '10/7/2013', '4/3/1958', '5/13/2011',
                           '4/8/1988', '12/4/1996', '4/14/2003', '1/25/1979', '3/18/2015'],
            'end_date': ['3/12/1962', '7/27/2013', '5/19/2011', '12/27/1995', '9/11/1995',
                         '1/6/2010', '9/22/2020', '10/13/2013', '4/9/1958', '5/19/2011',
                         '4/14/1988', '12/10/1996', '4/20/2003', '1/31/1979', '3/24/2015']
        }
        test_intervals = pd.DataFrame(intervals_data)
        
        # 這裡需要跟 create_dataset_vectorized 邏輯一致，
        # 但為了評估方便，我們還是手動抓取歷史區間的特徵
        feat_cols = ['sea_level', 'month_sin', 'month_cos', 'year_norm', 
                     'sl_mean_3d', 'sl_max_3d', 'sl_mean_7d', 'sl_max_7d', 'sl_std_7d']
        
        all_f1 = []
        for _, row in test_intervals.iterrows():
            h_start = pd.to_datetime(row['start_date'])
            h_end = pd.to_datetime(row['end_date'])
            pred_start = h_end + timedelta(days=1)
            pred_end = pred_start + timedelta(days=13)
            
            y_true, y_pred = [], []
            for stn in TESTING_STATIONS:
                stn_df = df_daily[df_daily['station_name'] == stn]
                
                # 找出歷史區間的最後一天，我們要用它往前推算 Lag 特徵
                # 注意：因為我們改變了特徵生成方式 (Lag Features)，
                # 這裡最簡單的方式是直接抓取 h_start 到 h_end 的數據並 Flatten
                # 這與 create_dataset_vectorized 的邏輯 (Columns: lag0, lag1...) 
                # 在物理意義上是一樣的 (都是 7 天特徵平鋪)
                
                mask_hist = (stn_df['time'] >= h_start) & (stn_df['time'] <= h_end)
                mask_future = (stn_df['time'] >= pred_start) & (stn_df['time'] <= pred_end)
                
                # 這裡要確保抓出的順序是跟 Training 時一致的 (T-0, T-1... 或者是 時間順序)
                # create_dataset_vectorized 中的 shift(0) 是當天，shift(1) 是昨天
                # 所以它的特徵順序是 [Day7, Day6, Day5... Day1]
                # 但原本的代碼是按時間排序 [Day1, Day2... Day7]
                # 為了避免順序錯亂，我們這裡手動構建特徵
                
                # 取得這 7 天的數據
                # hist_df = stn_df.loc[mask_hist, feat_cols].sort_values('time', ascending=False) # 最近的在前面
                # 先篩選列 -> 依時間排序 -> 最後只取 feature columns
                hist_df = stn_df.loc[mask_hist].sort_values('time', ascending=False)[feat_cols]
                hist_data = hist_df.values.flatten() # 展平
                
                true_future = stn_df.loc[mask_future, 'is_flood'].values
                
                if len(hist_data) == HIST_WINDOW * len(feat_cols) and len(true_future) == PRED_WINDOW:
                    preds = [m.predict([hist_data])[0] for m in self.models]
                    y_pred.extend(preds)
                    y_true.extend(true_future)
            
            if y_true:
                all_f1.append(f1_score(y_true, y_pred, zero_division=0))

        print(f"Average F1 Score: {np.mean(all_f1):.4f}")

if __name__ == "__main__":
    DATA_PATH = 'NEUSTG_19502020_12stations.mat'
    THRESH_PATH = 'Seed_Coastal_Stations_Thresholds.mat'
    
    PLATFORM_DIR = '/app/input_data'
    if os.path.exists(PLATFORM_DIR):
        print(f"Detecting platform environment. Using data from {PLATFORM_DIR}")
        DATA_PATH = os.path.join(PLATFORM_DIR, 'NEUSTG_19502020_12stations.mat')
        THRESH_PATH = os.path.join(PLATFORM_DIR, 'Seed_Coastal_Stations_Thresholds.mat')
    
    if os.path.exists(DATA_PATH) and os.path.exists(THRESH_PATH):
        pipeline = FloodModel()
        df = pipeline.load_and_preprocess(DATA_PATH, THRESH_PATH)
        
        # 由於改了模型架構，建議每次都重新訓練 (因為 LightGBM 很快)
        # print("Training new LightGBM model...")
        pipeline.train(df)
        pipeline.evaluate(df)
    else:
        print(f"Error: Data files not found at {DATA_PATH}")