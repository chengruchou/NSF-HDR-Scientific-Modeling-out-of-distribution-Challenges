import pandas as pd
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
from datetime import datetime, timedelta
import warnings
import os
import random

warnings.filterwarnings('ignore')

# === 設定種子以重現結果 ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# === 參數設定 ===
HIST_WINDOW_DAYS = 7
PRED_WINDOW_DAYS = 14
HOURS_PER_DAY = 24
SEQ_LEN = HIST_WINDOW_DAYS * HOURS_PER_DAY  # 168 小時
BATCH_SIZE = 512
EPOCHS = 15
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 站點劃分
TRAINING_STATIONS = [
    'Annapolis', 'Atlantic_City', 'Charleston', 'Washington', 'Wilmington', 
    'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook'
]
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

# === 資料集類別 ===
class FloodDataset(Dataset):
    def __init__(self, X, y):
        # X shape: (N, 168, 1) -> (Sequence Length, Features)
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === GRU 模型架構 ===
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=14):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional GRU 能同時捕捉前後文資訊
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=0.2, 
            bidirectional=True
        )
        
        # Fully Connected Layer
        # 因為是雙向，所以輸入維度是 hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(DEVICE)
        
        # out: (batch, seq_len, hidden_size*2)
        out, _ = self.gru(x, h0)
        
        # 取最後一個時間點的輸出
        out = out[:, -1, :]
        
        # 預測未來 14 天
        out = self.fc(out)
        return out

# === 資料處理 Pipeline ===
class DataPipeline:
    def __init__(self):
        self.threshold_map = {}
        
    def load_data(self, path_dataset, path_thresholds):
        print(f"Loading data from: {path_dataset}")
        try:
            data = sio.loadmat(path_dataset)
        except FileNotFoundError:
            print("❌ 找不到檔案！")
            return None

        # 解析 Mat 檔
        lat = data['lattg'].flatten()
        time = data['t'].flatten()
        sea_level = data['sltg']
        try:
            station_names = [s.item() for s in data['sname'].flatten()]
        except:
            station_names = [s[0][0] for s in data['sname']]

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

        # 載入門檻
        thresh_data = sio.loadmat(path_thresholds)
        try:
            th_names = [s.item() for s in thresh_data['sname'].flatten()]
        except:
            th_names = [s[0][0] for s in thresh_data['sname']]
        th_vals = thresh_data['thminor_stnd'].flatten()
        self.threshold_map = dict(zip(th_names, th_vals))
        
        # 計算「距離門檻值」 (Distance to Threshold)
        # 這是 Deep Learning 模型最重要的特徵
        df_hourly['flood_threshold'] = df_hourly['station_name'].map(self.threshold_map)
        df_hourly['dist_to_threshold'] = df_hourly['sea_level'] - df_hourly['flood_threshold']
        
        # 填補缺失值 (線性插值)
        df_hourly['dist_to_threshold'] = df_hourly.groupby('station_name')['dist_to_threshold'].transform(
            lambda x: x.interpolate(limit_direction='both')
        )
        
        return df_hourly.fillna(0)

    def create_sequences(self, df, stations):
        X_list, y_list = [], []
        
        df_subset = df[df['station_name'].isin(stations)].copy()
        
        # 預先計算每日最大值，用於標記 Target
        # Target: 未來某一天是否有任何一小時超過門檻 (dist > 0)
        df_subset['date'] = df_subset['time'].dt.date
        daily_max = df_subset.groupby(['station_name', 'date'])['dist_to_threshold'].max().reset_index()
        daily_max = daily_max.rename(columns={'dist_to_threshold': 'daily_max_dist'})
        
        # 用於快速查找
        daily_lookup = daily_max.set_index(['station_name', 'date'])
        
        for stn, group in df_subset.groupby('station_name'):
            group = group.sort_values('time').reset_index(drop=True)
            data = group['dist_to_threshold'].values
            times = group['time'].dt.date.values
            
            # 滑動視窗生成數據
            # 步長設為 24 (每天滑動一次)，這樣跟比賽的 Daily Index 比較接近
            # 如果想增加數據量，可以設為 12 或 6
            step = 24 
            
            for i in range(0, len(data) - SEQ_LEN - (PRED_WINDOW_DAYS * 24), step):
                # Input: 過去 168 小時的 dist
                x_seq = data[i : i + SEQ_LEN]
                
                # Output: 未來 14 天，每天是否淹水
                # 我們需要找到這段序列結束後的日期
                last_time = times[i + SEQ_LEN - 1]
                target_dates = [last_time + timedelta(days=d+1) for d in range(PRED_WINDOW_DAYS)]
                
                y_seq = []
                valid_target = True
                for d in target_dates:
                    if (stn, d) in daily_lookup.index:
                        max_val = daily_lookup.loc[(stn, d), 'daily_max_dist']
                        y_seq.append(1 if max_val > 0 else 0)
                    else:
                        valid_target = False
                        break
                
                if valid_target:
                    X_list.append(x_seq.reshape(-1, 1)) # (168, 1)
                    y_list.append(y_seq)                # (14,)
                    
        return np.array(X_list), np.array(y_list)

# === 訓練流程 ===
def train_local():
    DATA_PATH = 'NEUSTG_19502020_12stations.mat'
    THRESH_PATH = 'Seed_Coastal_Stations_Thresholds.mat'
    
    if not os.path.exists(DATA_PATH):
        print("❌ 請確認 .mat 檔案在當前目錄下")
        return

    pipeline = DataPipeline()
    df = pipeline.load_data(DATA_PATH, THRESH_PATH)
    
    print("生成訓練數據 (Training)...")
    X_train, y_train = pipeline.create_sequences(df, TRAINING_STATIONS)
    print(f"Train shape: {X_train.shape}")
    
    print("生成驗證數據 (Validation)...")
    X_val, y_val = pipeline.create_sequences(df, TESTING_STATIONS)
    print(f"Val shape: {X_val.shape}")

    # Dataset & DataLoader
    train_dataset = FloodDataset(X_train, y_train)
    val_dataset = FloodDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = GRUModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # === 關鍵：使用 Weighted Loss 處理不平衡 ===
    # 正樣本很少，給予較高權重 (例如 10 倍)
    pos_weight = torch.tensor([10.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print("\n=== Start Training (GRU) ===")
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                outputs = model(X_batch)
                # Sigmoid 轉機率
                probs = torch.sigmoid(outputs)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(y_batch.numpy())
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        # 攤平計算 Metrics
        y_true_flat = val_targets.flatten()
        y_prob_flat = val_preds.flatten()
        
        # 尋找最佳門檻 (每回合都找一次來看趨勢)
        best_th_epoch = 0.5
        best_f1_epoch = 0
        for th in np.linspace(0.3, 0.7, 9):
            y_pred_bin = (y_prob_flat > th).astype(int)
            f1 = f1_score(y_true_flat, y_pred_bin, zero_division=0)
            if f1 > best_f1_epoch:
                best_f1_epoch = f1
                best_th_epoch = th
        
        # 使用當前最佳門檻計算其他指標
        final_preds = (y_prob_flat > best_th_epoch).astype(int)
        acc = accuracy_score(y_true_flat, final_preds)
        auc = roc_auc_score(y_true_flat, y_prob_flat)
        tn, fp, fn, tp = confusion_matrix(y_true_flat, final_preds).ravel()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | "
              f"Val AUC: {auc:.4f} | Val F1: {best_f1_epoch:.4f} (Th: {best_th_epoch:.2f})")
        print(f"   -> CM: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
        
        if best_f1_epoch > best_f1:
            best_f1 = best_f1_epoch
            # 儲存模型
            torch.save(model.state_dict(), 'gru_model.pth')
            
    print(f"\nTraining Finished! Best F1: {best_f1:.4f}")
    print("模型已儲存為 'gru_model.pth'")

if __name__ == "__main__":
    train_local()