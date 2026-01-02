# Flood Prediction Challenge - GRU Submission

## Overview
This submission contains a Deep Learning approach for the Flood Prediction Challenge. It utilizes a **Bidirectional GRU (Gated Recurrent Unit)** network to predict coastal flooding events based on hourly sea level data.

The model processes 7 days of historical hourly data (168 time steps) to predict the maximum probability of flooding over the next 14 days.

## Model Architecture
- **Type:** Bidirectional GRU
- **Input Features:** Distance to Flood Threshold (interpolated)
- **Sequence Length:** 168 hours (7 days)
- **Hidden Layers:** 2
- **Hidden Size:** 64
- **Output:** Binary classification (Flood / No Flood) based on optimized probability threshold.

## Key Features
1.  **Fast Inference:** Implements a dictionary-based caching mechanism to handle 180k+ test queries efficiently (O(1) lookup), preventing time-limit exceeded errors.
2.  **Data Handling:** Uses linear interpolation to handle missing sea level data.
3.  **Threshold Optimization:** The decision boundary is tuned based on F1-score maximization during local validation.

## Files Included
- `model.py`: Main inference script. Handles data preprocessing, caching, and batch prediction.
- `model.pth`: Pre-trained PyTorch model weights.
- `Requirements.txt`: Python dependencies.
- `README.md`: Documentation.

## Dependencies
- torch
- pandas
- numpy
- scipy
- scikit-learn

## Usage
The model script accepts command-line arguments consistent with the competition ingestion program:

```bash
python model.py \
    --train_hourly path/to/train_hourly.csv \
    --test_hourly path/to/test_hourly.csv \
    --test_index path/to/test_index.csv \
    --predictions_out path/to/predictions.csv