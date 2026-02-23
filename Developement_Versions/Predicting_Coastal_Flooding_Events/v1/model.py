import pandas as pd
import numpy as np
import scipy.io as sio
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from datetime import datetime, timedelta
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================
# 設定與參數
# ============================================
HIST_WINDOW = 7
PRED_WINDOW = 14
TRAINING_STATIONS = [
    'Annapolis', 'Atlantic_City', 'Charleston', 'Washington', 'Wilmington', 
    'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook'
]
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

class FloodModel:
    def __init__(self):
        self.models = [] # 存放 14 個模型

    def _matlab2datetime(self, matlab_datenum):
        day = datetime.fromordinal(int(matlab_datenum))
        dayfrac = timedelta(days=matlab_datenum % 1) - timedelta(days=366)
        return day + dayfrac

    def _parse_mat_strings(self, mat_array):
        flat_array = mat_array.flatten()
        parsed_strings = []
        for item in flat_array:
            if isinstance(item, np.ndarray) and item.size > 0:
                parsed_strings.append(str(item.item()))
            elif isinstance(item, str):
                parsed_strings.append(item)
            else:
                try:
                    parsed_strings.append(str(item[0]))
                except:
                    parsed_strings.append(str(item))
        return parsed_strings

    def load_and_preprocess(self, path_dataset, path_thresholds):
        print("Loading and preprocessing data...")
        # 1. Load Data
        data = sio.loadmat(path_dataset)
        lat = data['lattg'].flatten()
        lon = data['lontg'].flatten()
        time = data['t'].flatten()
        sea_level = data['sltg']
        station_names = self._parse_mat_strings(data['sname'])
        
        time_dt = [self._matlab2datetime(t) for t in time]
        
        records = []
        for i, name in enumerate(station_names):
            records.append(pd.DataFrame({
                'time': time_dt,
                'station_name': name,
                'latitude': lat[i],
                'longitude': lon[i],
                'sea_level': sea_level[:, i]
            }))
        df_hourly = pd.concat(records, ignore_index=True)
        df_hourly['time'] = pd.to_datetime(df_hourly['time'])

        # 2. Load Thresholds
        thresh_data = sio.loadmat(path_thresholds)
        th_names = self._parse_mat_strings(thresh_data['sname'])
        th_vals = thresh_data['thminor_stnd'].flatten()
        threshold_map = dict(zip(th_names, th_vals))
        
        df_hourly['flood_threshold'] = df_hourly['station_name'].map(threshold_map)
        df_hourly['is_flood'] = (df_hourly['sea_level'] > df_hourly['flood_threshold']).astype(int)

        # 3. Daily Aggregation & Feature Engineering
        df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
            'sea_level': 'mean',
            'is_flood': 'max',
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()

        df_daily['month_sin'] = np.sin(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['month_cos'] = np.cos(2 * np.pi * df_daily['time'].dt.month / 12)
        df_daily['year_norm'] = (df_daily['time'].dt.year - 1950) / 70

        for window in [3, 7]:
            grp = df_daily.groupby('station_name')['sea_level']
            df_daily[f'sl_mean_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
            df_daily[f'sl_max_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).max())
            df_daily[f'sl_std_{window}d'] = grp.transform(lambda x: x.rolling(window, min_periods=1).std())
        
        return df_daily.fillna(0)

    def create_sequences(self, df, stations, hist_days=7, future_days=14, feature_cols=None):
        X, y = [], []
        df_subset = df[df['station_name'].isin(stations)].copy()
        
        for stn, group in df_subset.groupby('station_name'):
            group = group.sort_values('time').reset_index(drop=True)
            vals = group[feature_cols].values
            targets = group['is_flood'].values
            
            for i in range(0, len(group) - hist_days - future_days, 1):
                X.append(vals[i : i + hist_days].flatten())
                y.append(targets[i + hist_days : i + hist_days + future_days])
                
        return np.array(X), np.array(y)

    def train(self, df_daily, save_path='model.pkl'):
        print("Training models...")
        feat_cols = ['sea_level', 'month_sin', 'month_cos', 'year_norm', 
                     'sl_mean_3d', 'sl_max_3d', 'sl_mean_7d', 'sl_max_7d', 'sl_std_7d']
        
        X_train, y_train = self.create_sequences(df_daily, TRAINING_STATIONS, HIST_WINDOW, PRED_WINDOW, feat_cols)
        
        # Calculate weights
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1
        
        self.models = []
        for d in range(PRED_WINDOW):
            clf = XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.05,
                scale_pos_weight=scale_weight, eval_metric='logloss',
                use_label_encoder=False, n_jobs=-1, random_state=42
            )
            clf.fit(X_train, y_train[:, d])
            self.models.append(clf)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"Model saved to {save_path}")

    def evaluate(self, df_daily, intervals_file=None, load_model_path='model.pkl'):
        # Load model if not present
        if not self.models and os.path.exists(load_model_path):
            with open(load_model_path, 'rb') as f:
                self.models = pickle.load(f)
        
        if not self.models:
            print("No model found. Please train first.")
            return

        # Define Intervals (Hardcoded from guidelines as fallback)
        intervals_data = {
            'start_date': ['3/6/1962', '7/21/2013', '5/13/2011', '12/21/1995', '9/5/1995',
                           '12/31/2009', '9/16/2020', '10/7/2013', '4/3/1958', '5/13/2011',
                           '4/8/1988', '12/4/1996', '4/14/2003', '1/25/1979', '3/18/2015'],
            'end_date': ['3/12/1962', '7/27/2013', '5/19/2011', '12/27/1995', '9/11/1995',
                         '1/6/2010', '9/22/2020', '10/13/2013', '4/9/1958', '5/19/2011',
                         '4/14/1988', '12/10/1996', '4/20/2003', '1/31/1979', '3/24/2015']
        }
        test_intervals = pd.DataFrame(intervals_data)
        
        feat_cols = ['sea_level', 'month_sin', 'month_cos', 'year_norm', 
                     'sl_mean_3d', 'sl_max_3d', 'sl_mean_7d', 'sl_max_7d', 'sl_std_7d']
        
        all_f1 = []
        for _, row in test_intervals.iterrows():
            h_start = pd.to_datetime(row['start_date'])
            h_end = pd.to_datetime(row['end_date'])
            pred_start = h_end + timedelta(days=1)
            pred_end = pred_start + timedelta(days=13)
            
            y_true, y_pred = [], []
            for stn in TESTING_STATIONS:
                stn_df = df_daily[df_daily['station_name'] == stn]
                mask_hist = (stn_df['time'] >= h_start) & (stn_df['time'] <= h_end)
                mask_future = (stn_df['time'] >= pred_start) & (stn_df['time'] <= pred_end)
                
                hist_data = stn_df.loc[mask_hist, feat_cols].values.flatten()
                true_future = stn_df.loc[mask_future, 'is_flood'].values
                
                if len(hist_data) == HIST_WINDOW * len(feat_cols) and len(true_future) == PRED_WINDOW:
                    preds = [m.predict([hist_data])[0] for m in self.models]
                    y_pred.extend(preds)
                    y_true.extend(true_future)
            
            if y_true:
                all_f1.append(f1_score(y_true, y_pred, zero_division=0))

        print(f"Average F1 Score: {np.mean(all_f1):.4f}")

if __name__ == "__main__":
    # Example usage for reproduction
    # Update these paths to where the files are located in the submission folder
    DATA_PATH = 'NEUSTG_19502020_12stations.mat'
    THRESH_PATH = 'Seed_Coastal_Stations_Thresholds.mat'
    
    # 2. 檢查是否在比賽平台環境 (/app/input_data)
    PLATFORM_DIR = '/app/input_data'
    if os.path.exists(PLATFORM_DIR):
        print(f"Detecting platform environment. Using data from {PLATFORM_DIR}")
        DATA_PATH = os.path.join(PLATFORM_DIR, 'NEUSTG_19502020_12stations.mat')
        THRESH_PATH = os.path.join(PLATFORM_DIR, 'Seed_Coastal_Stations_Thresholds.mat')
    
    # 3. 執行
    if os.path.exists(DATA_PATH) and os.path.exists(THRESH_PATH):
        pipeline = FloodModel()
        df = pipeline.load_and_preprocess(DATA_PATH, THRESH_PATH)
        
        # 如果有 model.pkl 就直接載入評估，沒有就重新訓練
        if os.path.exists('model.pkl'):
            print("Loading pre-trained model...")
        else:
            print("Training new model...")
            pipeline.train(df)
            
        pipeline.evaluate(df)
    else:
        # 印出當前目錄內容幫助 Debug
        print(f"Error: Data files not found.")
        print(f"Looking for: {DATA_PATH}")
        print(f"Current dir files: {os.listdir('.')}")
        if os.path.exists(PLATFORM_DIR):
             print(f"Platform dir files: {os.listdir(PLATFORM_DIR)}")