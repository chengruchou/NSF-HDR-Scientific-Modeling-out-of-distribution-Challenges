import numpy as np

npz_path = "dataset/train_data_affi.npz"  # ← 改成你的檔案路徑

data = np.load(npz_path, allow_pickle=True)

print("Keys in npz file:")
print(data.files)
print("-" * 50)

for k in data.files:
    v = data[k]
    print(f"Key: {k}")
    print(f"  type   : {type(v)}")
    if isinstance(v, np.ndarray):
        print(f"  shape  : {v.shape}")
        print(f"  dtype  : {v.dtype}")
    print("-" * 50)
