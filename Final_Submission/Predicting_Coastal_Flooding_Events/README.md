# Flood Prediction Challenge - XGBoost & Feature Engineering Approach

## Overview
This submission presents a comprehensive Machine Learning solution for the Flood Prediction Challenge. Our approach evolved from a basic baseline to a highly optimized **XGBoost Classifier**, focusing heavily on **Time-Series Feature Engineering** and **Physics-Based Constraints**.

The final model leverages 15 distinct features—including flood inertia, seasonality, and volatility—to achieve top-tier performance on the leaderboard.

## Dev phase Performance
- **F1-Score:** ~0.94
- **Strategy:** High-recall thresholding combined with a precision-optimized XGBoost model.

## Project Structure & File Descriptions

This repository contains the main submission script, training pipelines for different model iterations, and simulation tools.

### Main Submission
- **`model.py`**: The final inference script used for the competition submission. It loads the best pre-trained model (typically based on the 15-feature architecture) and handles data preprocessing, feature extraction, and prediction generation.
- **`train.py`**: The training script used for generating the weight file.

### Tools
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
python model.py \
    --train_hourly input_data/train_hourly.csv \
    --test_hourly input_data/test_hourly.csv \
    --test_index input_data/test_index.csv \
    --predictions_out output/predictions.csv