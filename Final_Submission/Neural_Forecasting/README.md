# Neural Forecasting - Transformer-Based Time-Series Approach

## Overview
This submission provides a Neural Forecasting solution for the NSF-HDR Scientific Modeling OOD challenge.  
The core method is a Transformer-based model with change-aware features (`x`, `dx`, `ddx`) and time-aware encoding, designed for multi-step neural signal forecasting.

The inference pipeline supports monkey-specific configurations (for example `affi` and `beignet`) and follows competition-compatible input/output behavior.

## Dev Phase Notes
- **Forecasting setup:** multi-step prediction with observed-step masking (first 10 steps observed, future steps predicted).
- **Stability strategy:** gradient clipping, validation-based checkpointing, and optional learning-rate scheduling.

## Project Structure & File Descriptions

This folder contains model definition, training utilities, and data inspection tools.

### Main Submission
- **`model.py`**: Main model/inference entry. Defines `NFSTNDTModel` and competition-style `Model` class with `load()` and `predict()` APIs.
- **`trainer.py`**: Training loop implementation with one-step/multi-step data preparation, validation, and best-checkpoint saving.

### Tools
- **`utils.py`**: Dataset loading/splitting, normalization, seed control, and dataset wrapper (`NeuroForcastDataset`).
- **`inspect_train_data_affi.py`**: Utility script to inspect `train_data_affi.npz` statistics and save histogram plots for feature 0.
- **`requirement.txt`**: Minimal Python dependencies for this module.

## Key Technical Approaches

### 1. Change-Aware Representation
- Injects first-order and second-order temporal differences (`dx`, `ddx`) before Transformer encoding.
- Improves sensitivity to temporal transitions and spike-like behavior.

### 2. Time-Aware Encoding
- Supports `Time2Vec` and delta-time features.
- Uses synthetic time tensors during inference when explicit timestamps are not provided.

### 3. Competition-Compatible Inference Logic
- Enforces shape checks for `(N, 20, C, 9)` input format.
- Applies normalization/denormalization using saved dataset statistics.
- Keeps first observed steps fixed and predicts future horizons in a competition-aligned way.

## Dependencies
- `torch`
- `numpy`
- `matplotlib`
- `tqdm`

## Usage

Install dependencies:

```bash
pip install -r requirement.txt
```

Inspect training data example:

```bash
python inspect_train_data_affi.py
```

Run training by integrating `Trainer`/`NeuroForcastDataset` in your training script:

```bash
python trainer.py
```

## Notes
- Update dataset paths under `dataset/` according to your local environment before running experiments.
