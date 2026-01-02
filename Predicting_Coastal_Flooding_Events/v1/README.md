# Coastal Flooding Prediction Model (iHARP Challenge)

## Approach
This solution uses an ensemble of 14 XGBoost Classifiers (one for each future day) to predict flooding events.

### Features
- **Seasonality**: Month sine/cosine encoding.
- **Trend**: Normalized year index to capture sea level rise.
- **Rolling Stats**: 3-day and 7-day rolling Mean, Max, and Std of sea levels.
- **Station**: Trained on 9 stations, tested on 3 out-of-domain stations.

### Files
- `model.py`: Contains the `FloodModel` class for preprocessing, training, and evaluation.
- `model.pkl`: Serialized XGBoost models (list of 14 classifiers).
- `requirements.txt`: Python dependencies.

## How to Run
1. Ensure the dataset files (`NEUSTG_19502020_12stations.mat` and `Seed_Coastal_Stations_Thresholds.mat`) are in the same directory.
2. Run the script:
   ```bash
   python model.py