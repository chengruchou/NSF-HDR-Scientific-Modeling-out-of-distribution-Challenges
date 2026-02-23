# Flood Prediction Challenge - XGBoost & Feature Engineering Approach

## Overview
This submission presents a comprehensive Machine Learning solution for the Flood Prediction Challenge. Our approach evolved from a basic baseline to a highly optimized **XGBoost Classifier**, focusing heavily on **Time-Series Feature Engineering** and **Physics-Based Constraints**.

The final model leverages 15 distinct features—including flood inertia, seasonality, and volatility—to achieve top-tier performance on the leaderboard.

## 🏆 Performance
- **F1-Score:** ~0.94
- **Strategy:** High-recall thresholding combined with a precision-optimized XGBoost model.

## 📂 Project Structure & File Descriptions

This repository contains the main submission script, training pipelines for different model iterations, and simulation tools.

### 1. Main Submission
- **`model_xgb_15_feat.py`**: The final inference script used for the competition submission. It loads the best pre-trained model (typically based on the 15-feature architecture) and handles data preprocessing, feature extraction, and prediction generation.

### 2. Model Iterations (Development History)
We developed the solution in three progressive stages, each improving upon the last:

#### Stage 1: Basic Classification
- **`train_xgb.py`** / **`model_xgb.py`**
    - **Goal:** Switch from Regression (Baseline) to Classification.
    - **Features:** Basic sea level statistics (`mean`, `max`) and physical distance to threshold (`dist_to_threshold`).
    - **Model:** `XGBClassifier` with class weight adjustments.

#### Stage 2: Seasonality & Trend 
- **`train_xgb_11feat.py`** / **`model_xgb_11feat.py`**
    - **Goal:** Capture cyclical patterns and long-term trends.
    - **New Features (11 Total):** Added `month_sin`, `month_cos` (Cyclical Encoding) and `year_norm` (Trend) to handle seasonal tides and sea-level rise.

#### Stage 3: Inertia & Volatility (Best Performance)
- **`train_xgb_15feat.py`** / **`model_xgb_15_feat.py`**
    - **Goal:** Maximize MCC by modeling the "persistence" of flooding events.
    - **New Features (15 Total):**
        - **Inertia:** `flood_sum_7d` (how many days flooded recently), `flood_last_day`.
        - **Volatility:** `sea_level_std_7d` (storm surge detection).
        - **Fine-grained Lags:** `sea_level_lag1`, `sea_level_lag2`.

### 3. Other Baselines
- **`train_baseline.py`**: The initial Regressor-based baseline provided as a starting point.
- **`train_lgbm.py`**: An experimental script using LightGBM to compare training speed and accuracy against XGBoost. (Training script only, no inference counterpart).

### 4. Tools
- **`simulate_run.py`**: A local testing script that mocks the CodaLab environment. It generates dummy `test_hourly.csv` and `test_index.csv` to verify that `model.py` runs without errors and produces valid output format.

## ⚙️ Key Technical Approaches

### 1. Feature Engineering
We engineered features to capture the physical dynamics of coastal flooding:
* **Physics:** Distance to dynamic thresholds (`mean + 1.5*std`).
* **Temporal:** Cyclical encoding for seasonality.
* **Inertia:** Historical flood counts to model the persistence of high-water events (storm surges often last multiple days).

### 2. Handling Class Imbalance
* **Threshold Tuning:** We optimized the decision boundary (Threshold < 0.5) to maximize F1-score on the highly imbalanced dataset (90%+ positive class in test distribution).
* **Class Weights:** Used `scale_pos_weight` during training to penalize false negatives.

### 3. Robust Training
* **Time-Series Split:** Training data (1950-2015) and Validation data (2016-2020) were split chronologically to prevent look-ahead bias.

## 📦 Dependencies
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `xgboost` (Version consistency recommended between local training and remote inference)

## 🚀 Usage

To run the main model (simulating the competition environment):

```bash
python model_xgb_15_feat.py \
    --train_hourly input_data/train_hourly.csv \
    --test_hourly input_data/test_hourly.csv \
    --test_index input_data/test_index.csv \
    --predictions_out output/predictions.csv