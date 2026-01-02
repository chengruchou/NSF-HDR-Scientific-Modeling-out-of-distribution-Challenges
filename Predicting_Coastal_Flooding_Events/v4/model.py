import argparse
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import timedelta

# 參數設定 (必須與訓練時一致)
HIST_DAYS = 7
PRED_DAYS = 14
FEATURES = ['sea_level', 'sea_level_3d_mean', 'sea_level_7d_mean']

class FloodModel:
    def __init__(self, model_path='xgb_models.pkl'):
        self.models = None
        self.model_path = model_path

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.models = pickle.load(f)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Error: Model file {self.model_path} not found!")
            sys.exit(1)

    def preprocess(self, df):
        """與訓練時完全一致的特徵工程"""
        print("Preprocessing data...")
        df['time'] = pd.to_datetime(df['time'])
        
        # 轉為每日資料
        # 注意：訓練時有用 latitude/longitude，但這裡我們只聚合必要的特徵
        # 這樣可以避免 'latitude' 缺失的錯誤
        df_daily = df.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
            'sea_level': 'mean'
        }).reset_index()
        
        # 特徵工程: 3日與7日滑動平均
        df_daily['sea_level_3d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df_daily['sea_level_7d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(
            lambda x: x.rolling(7, min_periods=1).mean())
            
        return df_daily

    def generate_submission(self, df_hourly, test_index_path, output_path):
        # 1. 預處理
        df_daily = self.preprocess(df_hourly)
        
        # 2. 建立 Cache 以加速查找
        # Key: (Station, Date) -> Row Values
        data_cache = {}
        for stn, group in df_daily.groupby('station_name'):
            group = group.sort_values('time')
            # 建立 日期 -> Index 的映射
            time_map = {t: i for i, t in enumerate(group['time'])}
            matrix = group[FEATURES].values
            data_cache[stn] = (time_map, matrix)

        # 3. 讀取考卷 (Test Index)
        print(f"Reading test index: {test_index_path}")
        try:
            test_intervals = pd.read_csv(test_index_path)
        except Exception as e:
            print(f"Error reading test index: {e}")
            return

        # 欄位標準化
        test_intervals.columns = [c.strip().lower() for c in test_intervals.columns]
        # 相容不同版本的欄位名稱
        if 'hist_start' in test_intervals.columns:
            test_intervals = test_intervals.rename(columns={'hist_start': 'start_date', 'hist_end': 'end_date'})
        if 'start_date' not in test_intervals.columns and len(test_intervals.columns) >= 4:
             test_intervals['start_date'] = test_intervals.iloc[:, 2]
             test_intervals['end_date'] = test_intervals.iloc[:, 3]
        
        test_intervals['start_date'] = pd.to_datetime(test_intervals['start_date'])
        test_intervals['end_date'] = pd.to_datetime(test_intervals['end_date'])
        
        if 'id' not in test_intervals.columns:
            test_intervals['id'] = range(len(test_intervals))

        results_ids = []
        results_probs = [] # 改為輸出機率 (AUC較高)
        
        print(f"Processing {len(test_intervals)} requests...")
        
        for stn, group in test_intervals.groupby('station_name'):
            if stn not in data_cache:
                # 找不到站點資料，填補 0.5 (不確定)
                results_ids.extend(group['id'].values)
                results_probs.extend([0.5] * len(group))
                continue
            
            time_map, matrix = data_cache[stn]
            batch_X = []
            valid_rows = []
            
            for i, row in group.iterrows():
                # 歷史區間的結束日 (hist_end)
                # 注意：test_index 給的是區間，我們需要這區間的最後一天往回推 HIST_DAYS
                # 這裡假設 end_date 就是我們要預測未來的前一天
                hist_end_date = row['end_date']
                
                # 在 Cache 中尋找這一天
                # 為了容錯，如果找不到該天，嘗試找前一天 (有些時區差異)
                idx_end = -1
                if hist_end_date in time_map:
                    idx_end = time_map[hist_end_date]
                elif (hist_end_date - timedelta(days=1)) in time_map:
                    idx_end = time_map[hist_end_date - timedelta(days=1)]
                
                if idx_end != -1 and idx_end >= HIST_DAYS - 1:
                    # 取出過去 7 天 (包含 hist_end_date)
                    # Array slice: [idx - 6 : idx + 1] -> 長度 7
                    hist_window = matrix[idx_end - HIST_DAYS + 1 : idx_end + 1]
                    
                    if len(hist_window) == HIST_DAYS:
                        batch_X.append(hist_window.flatten())
                        valid_rows.append(row['id'])
                    else:
                        results_ids.append(row['id']); results_probs.append(0.5)
                else:
                    results_ids.append(row['id']); results_probs.append(0.5)
            
            if not batch_X: continue
            
            # 批次預測
            batch_X = np.array(batch_X)
            
            # models 是一個 list，包含 14 個 XGBRegressor
            # 我們對每個未來日進行預測
            batch_preds = np.zeros((len(batch_X), PRED_DAYS))
            for d in range(PRED_DAYS):
                batch_preds[:, d] = self.models[d].predict(batch_X)
            
            # 策略：取 14 天中最大的淹水機率 (或迴歸值)
            # XGBRegressor 輸出的是 0~1 之間的數值 (訓練時是 0/1，MSE loss)
            # 我們可以將其視為機率
            max_probs = np.max(batch_preds, axis=1)
            
            # 限制在 0~1 之間 (回歸有時會超出範圍)
            max_probs = np.clip(max_probs, 0.0, 1.0)
            
            results_ids.extend(valid_rows)
            results_probs.extend(max_probs)

        # 輸出結果
        output_df = pd.DataFrame({'id': results_ids, 'y_prob': results_probs})
        output_df = output_df.sort_values('id')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving predictions to {output_path}...")
        output_df.to_csv(output_path, index=False)
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_hourly', help='Path to training hourly data') # 這裡用不到，但為了相容介面保留
    parser.add_argument('--test_hourly', help='Path to testing hourly data', required=True)
    parser.add_argument('--test_index', help='Path to testing index', required=True)
    parser.add_argument('--predictions_out', help='Path to output csv', required=True)
    
    args, unknown = parser.parse_known_args()
    
    # 初始化模型 (確保 xgb_models.pkl 在同一目錄下)
    model = FloodModel('xgb_models.pkl')
    model.load_model()
    
    if os.path.exists(args.test_hourly):
        df_hourly = pd.read_csv(args.test_hourly)
        model.generate_submission(df_hourly, args.test_index, args.predictions_out)
    else:
        print(f"Error: Test data not found at {args.test_hourly}")