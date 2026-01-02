import argparse
import pandas as pd
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# === 參數設定 (必須與訓練時一致) ===
HIST_WINDOW_DAYS = 7
PRED_WINDOW_DAYS = 14
HOURS_PER_DAY = 24
SEQ_LEN = HIST_WINDOW_DAYS * HOURS_PER_DAY  # 168
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ⚠️ 請填入訓練 Log 中 "Best F1" 對應的 Threshold (Th)
# 例如 Log 顯示: Val F1: 0.1929 (Th: 0.45) -> 這裡就填 0.45
BEST_THRESHOLD = 0.65

# === 模型定義 (必須與訓練時完全一致) ===
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=14):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=0.2, 
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class FloodModel:
    def __init__(self):
        self.model = None
        self.threshold_map = {}

    def preprocess_data_to_dict(self, df_hourly, path_thresholds=None):
        """極速預處理：計算 dist_to_threshold 並轉為 Cache"""
        print("Preprocessing data for fast lookup...")
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])
        
        # 載入門檻
        if path_thresholds and os.path.exists(path_thresholds):
            try:
                thresh_data = sio.loadmat(path_thresholds)
                try:
                    if 'sname' in thresh_data:
                        try:
                            th_names = [s.item() for s in thresh_data['sname'].flatten()]
                        except:
                            th_names = [s[0][0] for s in thresh_data['sname']]
                    th_vals = thresh_data['thminor_stnd'].flatten()
                    self.threshold_map = dict(zip(th_names, th_vals))
                except: pass
            except: pass
        
        # 計算特徵：dist_to_threshold
        df_hourly['flood_threshold'] = df_hourly['station_name'].map(self.threshold_map).fillna(3.0)
        df_hourly['dist_to_threshold'] = df_hourly['sea_level'] - df_hourly['flood_threshold']
        
        # 填補缺失值 (線性插值)
        df_hourly['dist_to_threshold'] = df_hourly.groupby('station_name')['dist_to_threshold'].transform(
            lambda x: x.interpolate(limit_direction='both')
        )
        df_hourly = df_hourly.fillna(0)

        # 建立 Cache: Key=Station, Value=(Date_Index_Map, Data_Array)
        data_cache = {}
        for stn, group in df_hourly.groupby('station_name'):
            group = group.sort_values('time')
            # 建立 時間 -> index 的映射
            time_map = {t: i for i, t in enumerate(group['time'])}
            # 只保留需要的特徵 (N, 1)
            matrix = group[['dist_to_threshold']].values.astype(np.float32)
            data_cache[stn] = (time_map, matrix)
            
        return data_cache

    def generate_submission(self, data_cache, test_index_path, output_path, load_model_path='gru_model.pth'):
        # 1. 載入模型
        if os.path.exists(load_model_path):
            print(f"Loading model from {load_model_path}...")
            self.model = GRUModel().to(DEVICE)
            self.model.load_state_dict(torch.load(load_model_path, map_location=DEVICE))
            self.model.eval()
        else:
            print("Error: Model file not found!")
            return

        # 2. 讀取考卷
        print(f"Reading test index: {test_index_path}")
        try:
            test_intervals = pd.read_csv(test_index_path)
        except: return

        # 欄位標準化
        test_intervals.columns = [c.strip().lower() for c in test_intervals.columns]
        if 'hist_start' in test_intervals.columns:
            test_intervals = test_intervals.rename(columns={'hist_start': 'start_date', 'hist_end': 'end_date'})
        if 'start_date' not in test_intervals.columns and len(test_intervals.columns) >= 4:
             test_intervals['start_date'] = test_intervals.iloc[:, 2]
             test_intervals['end_date'] = test_intervals.iloc[:, 3]

        # 確保有 id
        if 'id' not in test_intervals.columns:
            test_intervals['id'] = range(len(test_intervals))

        test_intervals['end_date'] = pd.to_datetime(test_intervals['end_date'])

        results_ids = []
        results_labels = []
        
        print(f"Processing {len(test_intervals)} entries...")
        
        # 批次處理以加速 (Batch Inference)
        BATCH_SIZE = 1024
        temp_batch_X = []
        temp_batch_ids = []
        
        # 用於記錄無法預測的 ID (資料不足)
        fallback_ids = []

        with torch.no_grad():
            for stn, group in test_intervals.groupby('station_name'):
                if stn not in data_cache:
                    fallback_ids.extend(group['id'].values)
                    continue
                
                time_map, matrix = data_cache[stn]
                
                for i, row in group.iterrows():
                    end_time = row['end_date'] # 這是 7 天歷史視窗的最後一小時
                    
                    # 我們需要往前抓 SEQ_LEN (168小時)
                    # 注意：CSV 中的 end_date 可能是 00:00，我們需要精確匹配
                    # 這裡假設 end_date 是該視窗的最後一個時間點
                    
                    if end_time in time_map:
                        idx_end = time_map[end_time]
                        if idx_end >= SEQ_LEN - 1:
                            # 抓取視窗
                            window = matrix[idx_end - SEQ_LEN + 1 : idx_end + 1]
                            temp_batch_X.append(window)
                            temp_batch_ids.append(row['id'])
                        else:
                            fallback_ids.append(row['id'])
                    else:
                        # 有時候 end_date 可能因為時區或格式對不上，嘗試模糊匹配或 fallback
                        # 這裡簡單處理：找不到就 fallback
                        fallback_ids.append(row['id'])

                    # 批次推論
                    if len(temp_batch_X) >= BATCH_SIZE:
                        x_tensor = torch.tensor(np.array(temp_batch_X)).float().to(DEVICE)
                        outputs = self.model(x_tensor) # (Batch, 14)
                        probs = torch.sigmoid(outputs)
                        
                        # 取 Max > Threshold
                        max_probs, _ = torch.max(probs, dim=1)
                        preds = (max_probs > BEST_THRESHOLD).int().cpu().numpy()
                        
                        results_ids.extend(temp_batch_ids)
                        results_labels.extend(preds)
                        
                        temp_batch_X = []
                        temp_batch_ids = []

            # 處理剩餘的 batch
            if temp_batch_X:
                x_tensor = torch.tensor(np.array(temp_batch_X)).float().to(DEVICE)
                outputs = self.model(x_tensor)
                probs = torch.sigmoid(outputs)
                max_probs, _ = torch.max(probs, dim=1)
                preds = (max_probs > BEST_THRESHOLD).int().cpu().numpy()
                
                results_ids.extend(temp_batch_ids)
                results_labels.extend(preds)

        # 合併結果
        # 對於 fallback 的 ID，我們預設不淹水 (0)
        if fallback_ids:
            results_ids.extend(fallback_ids)
            results_labels.extend([0] * len(fallback_ids))

        output_df = pd.DataFrame({'id': results_ids, 'label': results_labels})
        output_df = output_df.sort_values('id')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving {len(output_df)} predictions to {output_path}...")
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
        pipeline.generate_submission(data_cache, args.test_index, args.predictions_out, load_model_path='model.pth')
    else:
        print(f"Error: Test data not found at {args.test_hourly}")