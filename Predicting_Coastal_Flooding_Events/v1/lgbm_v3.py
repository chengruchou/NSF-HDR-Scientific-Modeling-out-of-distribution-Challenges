import pandas as pd
import numpy as np
import scipy.io as sio
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, confusion_matrix
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

class FloodModel:
    def __init__(self):
        self.models = []
        self.threshold_map = {}

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
        self.threshold_map = dict(zip(th_names, th_vals))
        
        df_hourly['flood_threshold'] = df_hourly['station_name'].map(self.threshold_map)
        df_hourly['is_flood'] = (df_hourly['sea_level'] > df_hourly['flood_threshold']).astype(int)

        # 3. Daily Aggregation
        print("Aggregating to daily data...")
        df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
            'sea_level': ['mean', 'max'], 
            'is_flood': 'max',
            'latitude': 'first',
            'longitude': 'first',
            'flood_threshold': 'first'
        }).reset_index()
        
        df_daily.columns = ['station_name', 'time', 'sl_mean', 'sl_max', 'is_flood', 'latitude', 'longitude', 'threshold']

        # 4. Feature Engineering
        print("Engineering features...")
        df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['year_norm'] = (df_daily['time'].dt.year - 1950) / 70

        # Target Proxy: Distance to Threshold
        df_daily['dist_to_threshold'] = df_daily['sl_max'] - df_daily['threshold']

        # Rolling Features
        for window in [3, 7, 30]:
            grp = df_daily.groupby('station_name')['sl_max']
            df_daily[f'max_mean_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
            df_daily[f'max_max_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).max())
            df_daily[f'max_std_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).std())
            
        for window in [3, 7]:
            grp = df_daily.groupby('station_name')['dist_to_threshold']
            df_daily[f'dist_mean_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())

        return df_daily.fillna(0)

    def create_dataset_vectorized(self, df, stations, hist_days=7, future_days=14, feature_cols=None):
        print("Creating dataset (Vectorized)...")
        
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"DataFrame is missing required columns: {missing_cols}")
        if 'dist_to_threshold' not in df.columns:
             raise KeyError("Column 'dist_to_threshold' is missing.")

        # 只使用指定的 stations 進行訓練
        df_subset = df[df['station_name'].isin(stations)].copy()
        
        X_list, y_list = [], []
        
        for stn, group in df_subset.groupby('station_name'):
            group = group.sort_values('time').reset_index(drop=True)
            
            # X: History
            feats = []
            for i in range(hist_days):
                shifted = group[feature_cols].shift(i)
                shifted.columns = [f"{c}_lag{i}" for c in feature_cols]
                feats.append(shifted)
            X_stn = pd.concat(feats, axis=1)
            
            # y: Target (Distance to Threshold)
            targets = []
            for i in range(future_days):
                target = group['dist_to_threshold'].shift(-(1 + i)) 
                targets.append(target)
            y_stn = pd.concat(targets, axis=1)
            
            valid_mask = ~(X_stn.isna().any(axis=1) | y_stn.isna().any(axis=1))
            X_list.append(X_stn[valid_mask])
            y_list.append(y_stn[valid_mask])
            
        if not X_list:
            raise ValueError("No valid training data created!")
            
        X = pd.concat(X_list, axis=0).values
        y = pd.concat(y_list, axis=0).values
        
        return X, y

    def train(self, df_daily, save_path='model.pkl'):
        print("Training LightGBM Regressors (Native API)...")
        
        feat_cols = ['sl_mean', 'sl_max', 'month_sin', 'month_cos', 'year_norm', 'threshold',
                     'dist_to_threshold', 'dist_mean_3d', 'dist_mean_7d',
                     'max_mean_3d', 'max_max_3d', 'max_std_3d',
                     'max_mean_7d', 'max_max_7d', 'max_std_7d',
                     'max_mean_30d', 'max_max_30d', 'max_std_30d']
        
        X_train, y_train = self.create_dataset_vectorized(df_daily, TRAINING_STATIONS, HIST_WINDOW, PRED_WINDOW, feat_cols)
        print(f"Training Data Shape: {X_train.shape}")
        
        params = {
            'objective': 'quantile',
            'alpha': 0.95,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42
        }

        self.models = []
        for d in range(PRED_WINDOW):
            d_train = lgb.Dataset(X_train, label=y_train[:, d])
            bst = lgb.train(params, d_train, num_boost_round=300)
            self.models.append(bst)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"Model saved to {save_path}")

    def generate_submission(self, df_daily, input_dir, output_dir, load_model_path='model.pkl'):
        if not self.models and os.path.exists(load_model_path):
            with open(load_model_path, 'rb') as f:
                self.models = pickle.load(f)
        
        if not self.models:
            print("No model found. Cannot generate submission.")
            return

        # 1. 讀取測試區間
        test_index_path = os.path.join(input_dir, 'test_index.csv')
        if os.path.exists(test_index_path):
            print(f"Reading test intervals from {test_index_path}")
            test_intervals = pd.read_csv(test_index_path)
        else:
            print("[Warning] test_index.csv not found. Using hardcoded intervals.")
            intervals_data = {
                'start_date': ['3/6/1962', '7/21/2013', '5/13/2011', '12/21/1995', '9/5/1995'],
                'end_date': ['3/12/1962', '7/27/2013', '5/19/2011', '12/27/1995', '9/11/1995']
            }
            test_intervals = pd.DataFrame(intervals_data)

        # ========================================================
        # 關鍵修復：欄位映射
        # Log 顯示欄位為: ['id', 'station_name', 'hist_start', 'hist_end', 'future_start', 'future_end']
        # ========================================================
        print(f"Raw CSV Columns: {test_intervals.columns.tolist()}")
        
        # 1. 標準化欄位名稱 (小寫)
        test_intervals.columns = [c.strip().lower() for c in test_intervals.columns]
        
        # 2. 映射 hist_start -> start_date
        if 'hist_start' in test_intervals.columns:
            print("Mapping 'hist_start' to 'start_date'...")
            test_intervals = test_intervals.rename(columns={'hist_start': 'start_date', 'hist_end': 'end_date'})
        
        # 3. 檢查必要欄位
        if 'start_date' not in test_intervals.columns:
             # 如果還是沒有，最後手段才用 iloc，但要避開 id 和 station_name
             print("Warning: 'start_date' still missing. Attempting advanced fallback...")
             # 根據 Log, 順序是 id, station_name, hist_start (index 2), hist_end (index 3)
             if len(test_intervals.columns) >= 4:
                 test_intervals['start_date'] = test_intervals.iloc[:, 2]
                 test_intervals['end_date'] = test_intervals.iloc[:, 3]
             else:
                 raise KeyError("Cannot identify start_date columns from input csv.")

        print("Normalized Test Intervals Head:")
        print(test_intervals.head())
        
        all_stations = df_daily['station_name'].unique()
        feat_cols = ['sl_mean', 'sl_max', 'month_sin', 'month_cos', 'year_norm', 'threshold',
                     'dist_to_threshold', 'dist_mean_3d', 'dist_mean_7d',
                     'max_mean_3d', 'max_max_3d', 'max_std_3d',
                     'max_mean_7d', 'max_max_7d', 'max_std_7d',
                     'max_mean_30d', 'max_max_30d', 'max_std_30d']
        
        results = []
        
        for idx, row in test_intervals.iterrows():
            # 安全日期轉換
            try:
                h_start = pd.to_datetime(row['start_date'])
                h_end = pd.to_datetime(row['end_date'])
            except Exception as e:
                print(f"Date parsing error at row {idx}: {e}")
                continue

            # ====================================================
            # 判斷是預測特定站點 還是 所有站點
            # Log 顯示 test_index.csv 包含 'station_name' 欄位
            # ====================================================
            target_stations_for_row = []
            
            if 'station_name' in row:
                # 如果該行指定了站點，只預測該站點 (避免重複運算)
                target_stations_for_row = [row['station_name']]
            else:
                # 如果沒指定，預設對所有站點跑 (Fallback)
                target_stations_for_row = all_stations

            for stn in target_stations_for_row:
                stn_df = df_daily[df_daily['station_name'] == stn]
                mask_hist = (stn_df['time'] >= h_start) & (stn_df['time'] <= h_end)
                
                hist_df = stn_df.loc[mask_hist].sort_values('time', ascending=False)
                if hist_df.empty: continue
                hist_data = hist_df[feat_cols].values.flatten()
                
                if len(hist_data) == HIST_WINDOW * len(feat_cols):
                    X_input = hist_data.reshape(1, -1)
                    pred_dists = [m.predict(X_input)[0] for m in self.models]
                    pred_flood = [(1 if dist > 0 else 0) for dist in pred_dists]
                    
                    res_entry = {
                        'start_date': row['start_date'], # 保留原始格式字串
                        'end_date': row['end_date'],
                        'station_name': stn
                    }
                    
                    # 如果 input csv 有 id，最好也帶上，方便評測系統對應
                    if 'id' in row:
                        res_entry['id'] = row['id']

                    for day_idx, val in enumerate(pred_flood):
                        res_entry[f'day_{day_idx+1}'] = int(val)
                    results.append(res_entry)

        if results:
            pred_df = pd.DataFrame(results)
            output_path = os.path.join(output_dir, 'predictions.csv')
            print(f"Saving predictions to {output_path} (Rows: {len(pred_df)})...")
            pred_df.to_csv(output_path, index=False)
            print("Done!")
        else:
            print("No predictions generated!")

if __name__ == "__main__":
    INPUT_DIR = '.'
    OUTPUT_DIR = '.'
    DATA_PATH = 'NEUSTG_19502020_12stations.mat'
    THRESH_PATH = 'Seed_Coastal_Stations_Thresholds.mat'
    
    PLATFORM_INPUT_DIR = '/app/input_data'
    PLATFORM_OUTPUT_DIR = '/app/output'
    
    if os.path.exists(PLATFORM_INPUT_DIR):
        print(f"Detecting platform environment.")
        INPUT_DIR = PLATFORM_INPUT_DIR
        OUTPUT_DIR = PLATFORM_OUTPUT_DIR
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        DATA_PATH = os.path.join(INPUT_DIR, 'NEUSTG_19502020_12stations.mat')
        THRESH_PATH = os.path.join(INPUT_DIR, 'Seed_Coastal_Stations_Thresholds.mat')
    
    if os.path.exists(DATA_PATH) and os.path.exists(THRESH_PATH):
        pipeline = FloodModel()
        df = pipeline.load_and_preprocess(DATA_PATH, THRESH_PATH)
        # pipeline.train(df)
        pipeline.generate_submission(df, INPUT_DIR, OUTPUT_DIR)
    else:
        print(f"Error: Data files not found.")