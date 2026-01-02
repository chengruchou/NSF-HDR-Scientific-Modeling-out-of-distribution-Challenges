import os
import pandas as pd
import numpy as np
import sys
import subprocess

def run_simulation():
    print("🚀 開始本地模擬測試 (Simulating CodaLab)...")

    # 1. 建立假的輸入資料 (Mock Input)
    os.makedirs("input_data", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    print("Step 1: 生成測試用 CSV 檔案...")
    
    # 模擬 test_hourly.csv (隨機數據)
    stations = ['Annapolis', 'Atlantic_City', 'The_Battery']
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    records = []
    for stn in stations:
        for d in dates:
            records.append({
                'time': d, 
                'station_name': stn, 
                'sea_level': np.random.randn() * 2  # 隨機水位
            })
    df_hourly = pd.DataFrame(records)
    df_hourly.to_csv("input_data/test_hourly.csv", index=False)

    # 模擬 test_index.csv (考卷)
    # 格式必須包含 id, station_name, hist_start, hist_end
    test_index = pd.DataFrame({
        'id': [0, 1, 2],
        'station_name': ['Annapolis', 'Atlantic_City', 'The_Battery'],
        'hist_start': ['2020-01-01 00:00:00', '2020-01-02 00:00:00', '2020-01-03 00:00:00'],
        'hist_end':   ['2020-01-07 23:00:00', '2020-01-08 23:00:00', '2020-01-09 23:00:00'],
        # 未來區間通常是 hist_end 之後的 14 天
        'future_start': ['2020-01-08 00:00:00', '2020-01-09 00:00:00', '2020-01-10 00:00:00'],
        'future_end':   ['2020-01-21 23:00:00', '2020-01-22 23:00:00', '2020-01-23 23:00:00']
    })
    test_index.to_csv("input_data/test_index.csv", index=False)

    print("Step 2: 執行您的 model.py...")
    # 這是模擬 ingestion.py 的呼叫方式
    cmd = [
        sys.executable, "-m", "model",
        "--train_hourly", "input_data/test_hourly.csv", # 這裡隨便給一個，因為我們是 Inference Only
        "--test_hourly", "input_data/test_hourly.csv",
        "--test_index", "input_data/test_index.csv",
        "--predictions_out", "output/predictions.csv"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✅ model.py 執行成功 (Exit Code 0)")
    except subprocess.CalledProcessError as e:
        print(f"❌ model.py 執行失敗！錯誤代碼: {e.returncode}")
        return

    print("Step 3: 檢查輸出格式 (Scoring Check)...")
    pred_path = "output/predictions.csv"
    if not os.path.exists(pred_path):
        print("❌ 找不到 predictions.csv，請檢查 model.py 是否有寫入檔案！")
        return

    try:
        preds = pd.read_csv(pred_path)
        print("預覽輸出內容：")
        print(preds.head())

        # 檢查 scoring.py 要求的欄位
        # scoring.py: if "id" not in preds.columns ... if "y_prob" not in ... and "label" not in ...
        required_cols = {'id'}
        value_cols = {'y_prob', 'label'}
        
        current_cols = set(preds.columns)
        
        if not required_cols.issubset(current_cols):
            print(f"❌ 缺少必要欄位: {required_cols - current_cols}")
        elif not (value_cols & current_cols):
            print(f"❌ 缺少預測值欄位: 需要 'y_prob' 或 'label'")
        else:
            print("✅ 格式檢查通過！準備好可以上傳了！")
            
    except Exception as e:
        print(f"❌ 讀取或檢查 CSV 時發生錯誤: {e}")

if __name__ == "__main__":
    run_simulation()