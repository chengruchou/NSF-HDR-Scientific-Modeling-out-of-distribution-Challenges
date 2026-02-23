# ==========================================
# 檔名：simulate_run.py (本地驗證神器)
# ==========================================
import os
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta

# 1. 建立假的測試環境
print("步驟 1: 建立模擬測試資料...")
os.makedirs('output', exist_ok=True)

# 讀取真實 .mat 來產生測試 csv (這樣才真實)
MAT_FILE = 'NEUSTG_19502020_12stations.mat'
if not os.path.exists(MAT_FILE):
    print(f"❌ 錯誤: 請將 {MAT_FILE} 放在當前目錄下！")
    exit()

data = sio.loadmat(MAT_FILE)
# 簡單轉換一部分數據為 CSV (模擬 ingestion)
# 我們只取前 1000 行來測試流程
print("  -> 正在從 .mat 轉換為 test_hourly.csv...")
lat = data['lattg'].flatten()
time = data['t'].flatten()
sea_level = data['sltg']
try:
    station_names = [s.item() for s in data['sname'].flatten()]
except:
    station_names = [s[0][0] for s in data['sname']]

def matlab2datetime(t):
    return datetime.fromordinal(int(t)) + timedelta(days=t%1) - timedelta(days=366)

time_dt = [matlab2datetime(t) for t in time[:100]] # 取前 100 筆就好，跑得快
records = []
for i, name in enumerate(station_names):
    # 只取一個站點的一個小片段來測試
    records.append(pd.DataFrame({
        'time': time_dt, 'station_name': name, 'sea_level': sea_level[:100, i]
    }))
df_hourly = pd.concat(records)
df_hourly.to_csv('test_hourly.csv', index=False)

# 建立 test_index.csv
print("  -> 建立 test_index.csv...")
# 假設我們要預測剛剛那些數據的時間段
pd.DataFrame({
    'id': range(5),
    'station_name': [station_names[0]] * 5,
    'hist_start': [time_dt[0]] * 5,
    'hist_end': [time_dt[70]] * 5 # 假設有一段長度
}).to_csv('test_index.csv', index=False)

# 2. 執行 model.py
print("\n步驟 2: 執行 model.py (模擬平台呼叫)...")
cmd = "python model.py --test_hourly test_hourly.csv --test_index test_index.csv --predictions_out output/predictions.csv"
ret = os.system(cmd)

if ret != 0:
    print("\n❌ 失敗：model.py 執行報錯！")
    exit()

# 3. 檢查結果
print("\n步驟 3: 檢查輸出結果...")
if os.path.exists('output/predictions.csv'):
    df_out = pd.read_csv('output/predictions.csv')
    print("✅ 成功產生 predictions.csv")
    print("預覽前 5 行：")
    print(df_out.head())
    
    # 檢查欄位
    if 'id' in df_out.columns and 'label' in df_out.columns:
        print("✅ 格式正確：包含 id 和 label 欄位")
    else:
        print(f"❌ 格式錯誤：目前的欄位是 {df_out.columns.tolist()}，應該要是 ['id', 'label']")
else:
    print("❌ 失敗：找不到 predictions.csv")