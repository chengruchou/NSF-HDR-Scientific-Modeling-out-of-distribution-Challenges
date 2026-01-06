import numpy as np
import torch
import os
import time

# 嘗試引用你的 model.py
try:
    from model_v16 import Model
    print("✅ 成功 import Model 類別")
except ImportError as e:
    print(f"❌ Import 失敗: {e}")
    print("請檢查 model.py 是否在當前目錄下")
    exit()

def test_local_inference():
    # 1. 設定模擬參數 (Affi 數據集規格)
    BATCH_SIZE = 2
    TIME_STEPS = 20
    # CHANNELS = 239
    CHANNELS = 89
    FEATURES = 9
    
    print("-" * 30)
    print("🚀 開始本地模擬測試...")
    
    # 2. 檢查檔案是否存在
    # required_files = ["model_affi_v14.pth", "train_data_average_std_affi.npz"]
    required_files = ["model_beignet_v14.pth", "train_data_average_std_beignet.npz"]
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ 缺少檔案: {f}")
            print("請確保檔名完全一致，並放在同一目錄下。")
            return
    print("✅ 必要檔案檢查通過")

    # 3. 初始化模型
    try:
        # Codabench 會這樣呼叫你的模型
        # model_wrapper = Model("affi")
        model_wrapper = Model("beignet")
        model_wrapper.load()
        print("✅ 模型初始化與權重載入成功")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    # 4. 建立假資料 (Dummy Data)
    # 注意：比賽輸入是 (N, 20, C, 9)，且只有前 10 步是有意義的數值，後 10 步通常是 0 或 nan (或是無效值)
    # 這裡我們隨機生成一些 float32 數據
    X_dummy = np.random.randn(BATCH_SIZE, TIME_STEPS, CHANNELS, FEATURES).astype(np.float32)
    
    print(f"ℹ️ 輸入資料形狀: {X_dummy.shape}")

    # 5. 執行預測
    try:
        start_time = time.time()
        # 這是最關鍵的一步
        y_pred = model_wrapper.predict(X_dummy) 
        end_time = time.time()
        
        print(f"✅ 推理成功！耗時: {end_time - start_time:.4f} 秒")
    except Exception as e:
        print(f"❌ 推理過程報錯: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 驗證輸出格式
    # 預期輸出: (N, 20, C)
    expected_shape = (BATCH_SIZE, TIME_STEPS, CHANNELS)
    
    if y_pred.shape != expected_shape:
        print(f"❌ 輸出形狀錯誤！")
        print(f"   預期: {expected_shape}")
        print(f"   實際: {y_pred.shape}")
        return
    else:
        print(f"✅ 輸出形狀正確: {y_pred.shape}")

    # 7. 檢查數值是否合理 (有無 NaN 或 Inf)
    if np.isnan(y_pred).any():
        print("❌ 警告：輸出包含 NaN (Not a Number)")
    elif np.isinf(y_pred).any():
        print("❌ 警告：輸出包含 Inf (無限大)")
    else:
        print("✅ 數值檢查通過 (無 NaN/Inf)")

    print("-" * 30)
    print("🎉 恭喜！此版本可以安全上傳。")

if __name__ == "__main__":
    test_local_inference()