import argparse
import pandas as pd
import numpy as np
import scipy.io as sio
import lightgbm as lgb
from datetime import datetime, timedelta
import pickle
import warnings
import os
import sys

warnings.filterwarnings('ignore')

HIST_WINDOW = 7
PRED_WINDOW = 14

# ⚠️ 請填入 train_local.py 跑出來的最佳門檻
BEST_THRESHOLD = 0

class FloodModel:
    def __init__(self):
        self.models = []
        self.threshold_map = {}

    def preprocess_data_to_dict(self, df_hourly, path_thresholds=None):
        print("Preprocessing data for fast lookup...")
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])
        
        if path_thresholds and os.path.exists(path_thresholds):
            try:
                thresh_data = sio.loadmat(path_thresholds)
                if 'sname' in thresh_data:
                    try:
                        th_names = [s.item() for s in thresh_data['sname'].flatten()]
                    except:
                        th_names = [s[0][0] for s in thresh_data['sname']]
                th_vals = thresh_data['thminor_stnd'].flatten()
                self.threshold_map = dict(zip(th_names, th_vals))
            except: pass
        
        df_hourly['flood_threshold'] = df_hourly['station_name'].map(self.threshold_map).fillna(3.0)
        
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

        df_daily = df_daily.fillna(0)

        data_cache = {}
        feat_cols = ['sl_mean', 'sl_max', 'month_sin', 'month_cos', 'year_norm', 'threshold',
                     'dist_to_threshold', 'dist_mean_3d', 'dist_mean_7d',
                     'max_mean_3d', 'max_max_3d', 'max_std_3d',
                     'max_mean_7d', 'max_max_7d', 'max_std_7d',
                     'max_mean_30d', 'max_max_30d', 'max_std_30d']
        
        for stn, group in df_daily.groupby('station_name'):
            group = group.sort_values('time')
            date_map = {d: i for i, d in enumerate(group['time'])}
            matrix = group[feat_cols].values
            data_cache[stn] = (date_map, matrix)
        return data_cache

    def generate_submission(self, data_cache, test_index_path, output_path, load_model_path='model.pkl'):
        if os.path.exists(load_model_path):
            with open(load_model_path, 'rb') as f:
                self.models = pickle.load(f)
        else:
            return

        try:
            test_intervals = pd.read_csv(test_index_path)
        except: return

        test_intervals.columns = [c.strip().lower() for c in test_intervals.columns]
        if 'hist_start' in test_intervals.columns:
            test_intervals = test_intervals.rename(columns={'hist_start': 'start_date', 'hist_end': 'end_date'})
        if 'start_date' not in test_intervals.columns and len(test_intervals.columns) >= 4:
             test_intervals['start_date'] = test_intervals.iloc[:, 2]
             test_intervals['end_date'] = test_intervals.iloc[:, 3]
        if 'id' not in test_intervals.columns:
            test_intervals['id'] = range(len(test_intervals))

        test_intervals['end_date'] = pd.to_datetime(test_intervals['end_date'])

        results_ids = []
        results_probs = []
        
        print(f"Processing {len(test_intervals)} requests...")
        
        for stn, group in test_intervals.groupby('station_name'):
            if stn not in data_cache:
                results_ids.extend(group['id'].values)
                results_probs.extend([0.0] * len(group))
                continue
            
            date_map, matrix = data_cache[stn]
            batch_X = []
            valid_rows = []
            
            for i, row in group.iterrows():
                end_date = row['end_date']
                if end_date in date_map:
                    idx_end = date_map[end_date]
                    if idx_end >= HIST_WINDOW - 1:
                        window = matrix[idx_end - HIST_WINDOW + 1 : idx_end + 1]
                        batch_X.append(window.flatten())
                        valid_rows.append(row['id'])
                    else:
                        results_ids.append(row['id'])
                        results_probs.append(0.0)
                else:
                    results_ids.append(row['id'])
                    results_probs.append(0.0)
            
            if not batch_X: continue
            
            batch_X = np.array(batch_X)
            
            batch_preds = np.zeros((len(batch_X), PRED_WINDOW))
            for d in range(PRED_WINDOW):
                batch_preds[:, d] = self.models[d].predict(batch_X)
            
            # 取最大距離
            max_dist = np.max(batch_preds, axis=1)
            
            # === 關鍵：Sigmoid 轉換 ===
            # 將 (dist - threshold) 轉換為 (0.5 周圍的機率)
            # dist > threshold -> prob > 0.5
            # dist < threshold -> prob < 0.5
            # 乘以 10 是為了讓 Sigmoid 更陡峭，模擬高信心度
            calibrated_probs = 1 / (1 + np.exp(-(max_dist - BEST_THRESHOLD) * 10))
            
            results_ids.extend(valid_rows)
            results_probs.extend(calibrated_probs)

        output_df = pd.DataFrame({'id': results_ids, 'y_prob': results_probs})
        output_df = output_df.sort_values('id')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_hourly', help='Path to training hourly data') 
    parser.add_argument('--test_hourly', help='Path to testing hourly data', required=True)
    parser.add_argument('--test_index', help='Path to testing index', required=True)
    parser.add_argument('--predictions_out', help='Path to output csv', required=True)
    
    args, unknown = parser.parse_known_args()
    
    pipeline = FloodModel()
    input_dir = os.path.dirname(args.test_hourly)
    thresh_path = os.path.join(input_dir, 'Seed_Coastal_Stations_Thresholds.mat')
    if not os.path.exists(thresh_path):
         thresh_path = os.path.join(os.path.dirname(input_dir), 'input_data', 'Seed_Coastal_Stations_Thresholds.mat')
    
    if os.path.exists(args.test_hourly):
        df_hourly = pd.read_csv(args.test_hourly)
        data_cache = pipeline.preprocess_data_to_dict(df_hourly, thresh_path)
        pipeline.generate_submission(data_cache, args.test_index, args.predictions_out, load_model_path='model.pkl')