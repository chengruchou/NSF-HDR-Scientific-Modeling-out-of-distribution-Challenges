# ==========================================
# 檔名：train_final_v2.py (針對 High FN 優化)
# ==========================================
import pandas as pd
import numpy as np
from scipy.io import loadmat
from datetime import datetime, timedelta
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import warnings
import os
print(xgboost.__version__)

warnings.filterwarnings('ignore')

# ========= 🎛️ 控制台 =========
VALIDATION_MODE = False  # True=調參看分數, False=生成上傳模型
# ==============================

HIST_DAYS = 7
FUTURE_DAYS = 14
# 新增 'sl_diff' 特徵，捕捉水位上漲速度
FEATURES = ['sea_level', 'sea_level_3d_mean', 'sea_level_7d_mean', 'dist_to_threshold', 'sl_diff']

def matlab2datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)

print("Loading Data...")
try:
    data = loadmat('NEUSTG_19502020_12stations.mat')
except:
    print("❌ 請確認 NEUSTG_19502020_12stations.mat 存在")
    exit()

lat, lon, sea_level = data['lattg'].flatten(), data['lontg'].flatten(), data['sltg']
time_dt = np.array([matlab2datetime(t) for t in data['t'].flatten()])
try: station_names = [s.item() for s in data['sname'].flatten()]
except: station_names = [s[0][0] for s in data['sname']]

records = []
for i, name in enumerate(station_names):
    records.append(pd.DataFrame({'time': time_dt, 'station_name': name, 'sea_level': sea_level[:, i]}))
df_hourly = pd.concat(records, ignore_index=True)
df_hourly['time'] = pd.to_datetime(df_hourly['time'])

print("Preprocessing...")
thresh_df = df_hourly.groupby('station_name')['sea_level'].agg(['mean', 'std']).reset_index()
thresh_df['threshold'] = thresh_df['mean'] + 1.5 * thresh_df['std']
df_hourly = df_hourly.merge(thresh_df[['station_name', 'threshold']], on='station_name', how='left')

print("Aggregating to Daily...")
df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
    'sea_level': ['mean', 'max'],
    'threshold': 'first'
}).reset_index()
df_daily.columns = ['station_name', 'time', 'sea_level', 'sea_level_max', 'threshold']

print("Feature Engineering...")
# 1. 距離門檻
df_daily['dist_to_threshold'] = df_daily['sea_level_max'] - df_daily['threshold']

# 2. 水位變化 (Diff) - 新增特徵
# 捕捉「暴漲」趨勢
df_daily['sl_diff'] = df_daily.groupby('station_name')['sea_level'].diff().fillna(0)

# 3. 滑動平均
df_daily['sea_level_3d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df_daily['sea_level_7d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# Target
df_daily['flood'] = (df_daily['sea_level_max'] > df_daily['threshold']).astype(int)

print("Building Dataset...")
X_all, y_all, t_all = [], [], []

for stn, grp in df_daily.groupby('station_name'):
    grp = grp.sort_values('time').reset_index(drop=True)
    vals = grp[FEATURES].values
    floods = grp['flood'].values
    times = grp['time'].values
    
    for i in range(len(grp) - HIST_DAYS - FUTURE_DAYS):
        hist_block = vals[i : i + HIST_DAYS].flatten()
        future_block = floods[i + HIST_DAYS : i + HIST_DAYS + FUTURE_DAYS]
        
        if not np.isnan(hist_block).any():
            X_all.append(hist_block)
            y_all.append(future_block)
            t_all.append(times[i + HIST_DAYS - 1])

X_all = np.array(X_all)
y_all = np.array(y_all)
t_all = pd.to_datetime(t_all)

if VALIDATION_MODE:
    print("\n🧪 [實驗模式] 2016-2020 驗證...")
    split_date = pd.to_datetime('2016-01-01')
    train_mask = t_all < split_date
    val_mask = t_all >= split_date
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
else:
    print("\n🚀 [上傳模式] 全量訓練...")
    X_train, y_train = X_all, y_all
    X_val, y_val = None, None

print(f"Training XGBClassifier (Features: {FEATURES})...")
models = []
val_probs = []
val_true = []

for d in range(FUTURE_DAYS):
    # 自動計算 class weight 以解決 High FN 問題
    # scale_pos_weight = 負樣本數 / 正樣本數
    n_pos = np.sum(y_train[:, d])
    n_neg = len(y_train) - n_pos
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=300, 
        max_depth=6, 
        learning_rate=0.03,
        objective='binary:logistic', 
        eval_metric='auc',
        scale_pos_weight=spw, # ⚠️ 關鍵修正：讓模型更重視淹水樣本
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train, y_train[:, d])
    models.append(model)
    
    if VALIDATION_MODE:
        p = model.predict_proba(X_val)[:, 1]
        val_probs.extend(p)
        val_true.extend(y_val[:, d])

if VALIDATION_MODE:
    val_probs = np.array(val_probs)
    val_true = np.array(val_true)
    
    # ⚠️ 關鍵修正：尋找並使用最佳門檻，而不是傻傻用 0.5
    best_f1 = 0
    best_th = 0.5
    for th in np.linspace(0.1, 0.9, 81):
        p_bin = (val_probs > th).astype(int)
        f1 = f1_score(val_true, p_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    val_preds = (val_probs > best_th).astype(int) # 使用最佳門檻算最後指標
    
    acc = accuracy_score(val_true, val_preds)
    try: auc = roc_auc_score(val_true, val_probs)
    except: auc = 0.5
    tn, fp, fn, tp = confusion_matrix(val_true, val_preds).ravel()
    
    print(f"\n🏆 最佳門檻 (Best Threshold): {best_th:.4f}")
    print("\n📊 驗證成績 (優化後):")
    print(f"Confusion Matrix: [TP={tp}, FP={fp}, TN={tn}, FN={fn}]")
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {best_f1:.4f}")
    print("------------------------------------------------")
    print("👉 滿意請改 VALIDATION_MODE=False 生成模型，並記得更新 model.py 的特徵列表！")
else:
    with open("xgb_models_improved.pkl", "wb") as f:
        pickle.dump(models, f)
    print("\n✅ 模型已儲存為 xgb_models_improved.pkl")

# AUC:      0.7538
# Accuracy: 0.6619
# F1 Score: 0.7180
# CM: [TP=130115, FP=25631, TN=69955, FN=76573]

# 最佳門檻 (Best Threshold): 0.3500

# 驗證成績 (優化後):
# Confusion Matrix: [TP=197072, FP=70096, TN=25490, FN=9616]
# AUC:       0.7580
# Accuracy:  0.7363
# F1 Score:  0.8318
    
# {
#   "auc": 0.65013490372748,
#   "acc": 0.8734611970825454,
#   "f1": 0.9319714248172558,
#   "mcc": 0.07193670359641022,
#   "n": 77739
# }