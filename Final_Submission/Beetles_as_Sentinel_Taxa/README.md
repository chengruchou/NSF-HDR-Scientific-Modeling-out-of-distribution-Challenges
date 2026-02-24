# Beetles as Sentinel Taxa - Event-Level MIL with ConvNeXtV2

## Overview
This submission presents a deep learning solution for the Beetles as Sentinel Taxa task using an **event-level Multiple Instance Learning (MIL)** framework.

The model combines **ConvNeXtV2 visual features**, **species/domain metadata embeddings**, and **calibration image cues** (color picker + scale bar) to predict probabilistic targets for `SPEI_30d`, `SPEI_1y`, and `SPEI_2y` (`mu`, `sigma`).

## Dev phase Performance
- **Objective:** probabilistic regression with CRPS-oriented optimization.
- **Validation setup:** event-based evaluation with CRPS and R2 metrics across 3 SPEI horizons.

## Project Structure & File Descriptions

This repository contains the main inference script, model/training pipeline, and evaluation utilities.

### Main Submission
- **`model.py`**: Final inference module used for submission. Defines the Event MIL architecture and the competition-style `Model` class with `load()` and `predict(batch)` APIs.
- **`train.py`**: End-to-end training script. Loads the `imageomics/sentinel-beetles` dataset, builds dataloaders, trains the model, and saves checkpoints.

### Tools
- **`utils.py`**: Data transforms, dataset/collate utilities, training argument parser, CRPS computation, and R2 evaluation helpers.
- **`evaluation.py`**: Local evaluation/smoke-test script for model loading, output format verification, and validation metrics on sampled events.
- **`requirements.txt`**: Python package dependencies.

## Key Technical Approaches

### 1. Event-Level Multiple Instance Learning
Each event is modeled as a bag of beetle images; attention pooling aggregates instance-level features into an event representation for SPEI prediction.

### 2. Metadata + Calibration Fusion
The pipeline fuses:
* **Species embedding** for taxonomic context.
* **Domain embedding** for domain-aware calibration.
* **Color/scale calibration encoders** to incorporate acquisition context.

### 3. Probabilistic Multi-Horizon Prediction
The model predicts Gaussian parameters (`mu`, `sigma`) for three horizons, enabling uncertainty-aware inference and CRPS-based evaluation.

## Dependencies
- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `tqdm`
- `datasets`
- `scikit-learn`

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python train.py --model_version v6 --checkpoint_name model_v6.pth
```

Run local evaluation / smoke test:

```bash
python evaluation.py
```

## Notes
- Update dataset paths to your local environment before running experiments.
