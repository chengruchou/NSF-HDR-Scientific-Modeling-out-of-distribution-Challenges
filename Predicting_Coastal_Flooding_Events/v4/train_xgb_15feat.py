# ==========================================
# 檔名：train_final_v5.py (波動率 + Lags)
# ==========================================
import pandas as pd
import numpy as np
from scipy.io import loadmat
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# ========= 🎛️ 控制台 =========
VALIDATION_MODE = False  # True=驗證, False=上傳
# ==============================

HIST_DAYS = 7
FUTURE_DAYS = 14

# 🚀 v5 終極特徵列表 (共 15 個)
FEATURES = [
    'sea_level', 'sea_level_3d_mean', 'sea_level_7d_mean', 'sea_level_30d_mean', # 均值
    'sea_level_std_7d',            # 波動率 (新)
    'sea_level_lag1', 'sea_level_lag2', # 精確路徑 (新)
    'dist_to_threshold', 'sl_diff',
    'flood_sum_7d', 'flood_last_day',
    'dist_mean_7d',
    'month_sin', 'month_cos', 'year_norm'
]

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
df_daily_max = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')])['sea_level'].max().reset_index()

df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
    'sea_level': 'mean',
    'threshold': 'first'
}).reset_index()

df_daily['sea_level_max'] = df_daily_max['sea_level']

print("Feature Engineering...")
# 1. 時間特徵
df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['time'].dt.month / 12)
df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['time'].dt.month / 12)
df_daily['year_norm'] = (df_daily['time'].dt.year - 1950) / 70

# 2. 標記淹水
df_daily['flood'] = (df_daily['sea_level_max'] > df_daily['threshold']).astype(int)
df_daily['flood_sum_7d'] = df_daily.groupby('station_name')['flood'].transform(lambda x: x.rolling(7, min_periods=1).sum())
df_daily['flood_last_day'] = df_daily['flood']

# 3. 物理特徵
df_daily['dist_to_threshold'] = df_daily['sea_level_max'] - df_daily['threshold']
df_daily['dist_mean_7d'] = df_daily.groupby('station_name')['dist_to_threshold'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df_daily['sl_diff'] = df_daily.groupby('station_name')['sea_level'].diff().fillna(0)

# 4. 統計特徵 (含 Lags & Volatility)
# 滑動平均
df_daily['sea_level_3d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df_daily['sea_level_7d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df_daily['sea_level_30d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(30, min_periods=1).mean())

# 波動率 (標準差)
df_daily['sea_level_std_7d'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))

# Lags (昨天, 前天)
df_daily['sea_level_lag1'] = df_daily.groupby('station_name')['sea_level'].shift(1).fillna(0)
df_daily['sea_level_lag2'] = df_daily.groupby('station_name')['sea_level'].shift(2).fillna(0)

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

print(f"Training XGBClassifier (Features: {len(FEATURES)} per day)...")
models = []
val_probs = []
val_true = []

for d in range(FUTURE_DAYS):
    n_pos = np.sum(y_train[:, d])
    n_neg = len(y_train) - n_pos
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    
    # 稍微增加模型複雜度 (n_estimators 300 -> 400) 以適應更多特徵
    model = XGBClassifier(
        n_estimators=400, 
        max_depth=6, 
        learning_rate=0.03,
        objective='binary:logistic', 
        eval_metric='auc',
        scale_pos_weight=spw,
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
    
    best_f1 = 0
    best_th = 0.5
    for th in np.linspace(0.1, 0.9, 81):
        p_bin = (val_probs > th).astype(int)
        f1 = f1_score(val_true, p_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            
    val_preds = (val_probs > best_th).astype(int)
    
    acc = accuracy_score(val_true, val_preds)
    try: auc = roc_auc_score(val_true, val_probs)
    except: auc = 0.5
    tn, fp, fn, tp = confusion_matrix(val_true, val_preds).ravel()
    
    print(f"\n🏆 最佳門檻 (Best Threshold): {best_th:.4f}")
    print("\n📊 驗證成績 (V5 終極特徵):")
    print(f"Confusion Matrix: [TP={tp}, FP={fp}, TN={tn}, FN={fn}]")
    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {best_f1:.4f}")
    print("------------------------------------------------")
    print("👉 滿意請改 VALIDATION_MODE=False 生成模型")
else:
    with open("xgb_models_all_feat.pkl", "wb") as f:
        pickle.dump(models, f)
    print("\n✅ 模型已儲存為 xgb_models_all_feat.pkl")

# 最佳門檻 (Best Threshold): 0.4600

# 驗證成績 (V5 終極特徵):
# Confusion Matrix: [TP=196781, FP=65947, TN=29639, FN=9907]
# AUC:       0.8052
# Accuracy:  0.7491
# F1 Score:  0.8384

# th = 0.46
# {
#   "auc": 0.7045146862546281,
#   "acc": 0.8142759747359755,
#   "f1": 0.892855022559962,
#   "mcc": 0.2020611453287194,
#   "n": 77739
# }

# th=0.15
# {
#   "auc": 0.7045146862546281,
#   "acc": 0.8867492506978479,
#   "f1": 0.9399626300786951,
#   "mcc": 0.026041684020665624,
#   "n": 77739
# }

# th=0.05
# {
#   "auc": 0.7045146862546281,
#   "acc": 0.8867235235853304,
#   "f1": 0.939961274135486,
#   "mcc": 0.0,
#   "n": 77739
# }

# th = 0.01
# {
#     "auc": 0.7045146862546281,
#     "acc": 0.8867235235853304,
#     "f1": 0.939961274135486,
#     "mcc": 0.0,
#     "n": 77739
# }