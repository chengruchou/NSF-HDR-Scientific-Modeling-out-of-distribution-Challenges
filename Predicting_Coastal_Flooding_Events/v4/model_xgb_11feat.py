import argparse
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import timedelta

HIST_DAYS = 7
PRED_DAYS = 14

# ⚠️ v4 特徵列表 (共 11 個)
FEATURES = [
    'sea_level', 'sea_level_3d_mean', 'sea_level_7d_mean', 
    'dist_to_threshold', 'sl_diff',
    'flood_sum_7d', 'flood_last_day',
    'dist_mean_7d',
    'month_sin', 'month_cos', 'year_norm'
]

# ⚠️ 請填入 train_final_v4.py 跑出來的最佳門檻
BEST_THRESHOLD = 0.46 

class FloodModel:
    def __init__(self, model_path='xgb_models_improved_add_feat.pkl'):
        self.models = None
        self.model_path = model_path

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.models = pickle.load(f)
        else:
            sys.exit(1)

    def preprocess(self, df):
        df['time'] = pd.to_datetime(df['time'])
        
        # 1. 門檻
        thresh_stats = df.groupby('station_name')['sea_level'].agg(['mean', 'std']).reset_index()
        thresh_stats['threshold'] = thresh_stats['mean'] + 1.5 * thresh_stats['std']
        df = df.merge(thresh_stats[['station_name', 'threshold']], on='station_name', how='left')
        
        # 2. 聚合
        df_daily_max = df.groupby(['station_name', pd.Grouper(key='time', freq='D')])['sea_level'].max().reset_index()
        df_daily = df.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
            'sea_level': 'mean',
            'threshold': 'first'
        }).reset_index()
        df_daily['sea_level_max'] = df_daily_max['sea_level']

        # 3. 特徵工程
        # 時間
        df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['year_norm'] = (df_daily['time'].dt.year - 1950) / 70
        
        # 淹水
        df_daily['flood'] = (df_daily['sea_level_max'] > df_daily['threshold']).astype(int)
        df_daily['flood_sum_7d'] = df_daily.groupby('station_name')['flood'].transform(lambda x: x.rolling(7, min_periods=1).sum())
        df_daily['flood_last_day'] = df_daily['flood']
        
        # 物理
        df_daily['dist_to_threshold'] = df_daily['sea_level_max'] - df_daily['threshold']
        df_daily['dist_mean_7d'] = df_daily.groupby('station_name')['dist_to_threshold'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        df_daily['sl_diff'] = df_daily.groupby('station_name')['sea_level'].diff().fillna(0)
        df_daily['sea_level_3d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df_daily['sea_level_7d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(7, min_periods=1).mean())
            
        return df_daily

    def generate_submission(self, df_hourly, test_index_path, output_path):
        df_daily = self.preprocess(df_hourly)
        
        data_cache = {}
        for stn, group in df_daily.groupby('station_name'):
            group = group.sort_values('time')
            time_map = {t: i for i, t in enumerate(group['time'])}
            matrix = group[FEATURES].values
            data_cache[stn] = (time_map, matrix)

        try: test_intervals = pd.read_csv(test_index_path)
        except: return

        test_intervals.columns = [c.strip().lower() for c in test_intervals.columns]
        if 'hist_start' in test_intervals.columns:
            test_intervals = test_intervals.rename(columns={'hist_start': 'start_date', 'hist_end': 'end_date'})
        if 'start_date' not in test_intervals.columns and len(test_intervals.columns) >= 4:
             test_intervals['start_date'] = test_intervals.iloc[:, 2]
             test_intervals['end_date'] = test_intervals.iloc[:, 3]
        
        test_intervals['end_date'] = pd.to_datetime(test_intervals['end_date'])
        if 'id' not in test_intervals.columns: test_intervals['id'] = range(len(test_intervals))

        results_ids, results_probs = [], []
        
        print(f"Processing {len(test_intervals)} requests...")
        for stn, group in test_intervals.groupby('station_name'):
            if stn not in data_cache:
                results_ids.extend(group['id'].values)
                results_probs.extend([0.5] * len(group))
                continue
            
            time_map, matrix = data_cache[stn]
            batch_X, valid_rows = [], []
            
            for i, row in group.iterrows():
                hist_end_date = row['end_date']
                idx_end = -1
                if hist_end_date in time_map: idx_end = time_map[hist_end_date]
                elif (hist_end_date - timedelta(days=1)) in time_map: idx_end = time_map[hist_end_date - timedelta(days=1)]
                
                if idx_end != -1 and idx_end >= HIST_DAYS - 1:
                    hist_window = matrix[idx_end - HIST_DAYS + 1 : idx_end + 1]
                    if len(hist_window) == HIST_DAYS:
                        batch_X.append(hist_window.flatten())
                        valid_rows.append(row['id'])
                    else:
                        results_ids.append(row['id']); results_probs.append(0.5)
                else:
                    results_ids.append(row['id']); results_probs.append(0.5)
            
            if not batch_X: continue
            
            batch_X = np.array(batch_X)
            batch_preds = np.zeros((len(batch_X), PRED_DAYS))
            for d in range(PRED_DAYS):
                batch_preds[:, d] = self.models[d].predict_proba(batch_X)[:, 1]
            
            max_probs = np.max(batch_preds, axis=1)
            
            # Sigmoid Adjustment
            adjusted_probs = 1 / (1 + np.exp(-10 * (max_probs - BEST_THRESHOLD)))
            
            results_ids.extend(valid_rows)
            results_probs.extend(adjusted_probs)

        output_df = pd.DataFrame({'id': results_ids, 'y_prob': results_probs})
        output_df = output_df.sort_values('id')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_hourly', required=True)
    parser.add_argument('--test_index', required=True)
    parser.add_argument('--predictions_out', required=True)
    args, unknown = parser.parse_known_args()
    
    # 記得 requirements.txt: xgboost==1.6.2 (或您本地版本)
    model = FloodModel('xgb_models_improved_add_feat.pkl')
    model.load_model()
    
    if os.path.exists(args.test_hourly):
        df_hourly = pd.read_csv(args.test_hourly)
        model.generate_submission(df_hourly, args.test_index, args.predictions_out)