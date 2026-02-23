from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from model_v6 import Model
from utils import evaluate_spei_r2_scores, gaussian_crps


def _get_first_present(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _gather_event_rows(ds_split, max_events=None):
    event_to_indices = {}
    for idx, event_id in enumerate(ds_split["eventID"]):
        event_to_indices.setdefault(event_id, []).append(idx)
    events = list(event_to_indices.values())
    if max_events is not None:
        events = events[:max_events]
    return events


def _pred_to_arrays(pred):
    mu = np.array([pred["SPEI_30d"]["mu"], pred["SPEI_1y"]["mu"], pred["SPEI_2y"]["mu"]], dtype=np.float32)
    sigma = np.array(
        [pred["SPEI_30d"]["sigma"], pred["SPEI_1y"]["sigma"], pred["SPEI_2y"]["sigma"]],
        dtype=np.float32,
    )
    return mu, sigma


def smoke_test(model: Model):
    dummy = []
    for _ in range(3):
        dummy.append(
            {
                "relative_img": Image.new("RGB", (64, 64), color=(128, 128, 128)),
                "colorpicker_img": Image.new("RGB", (64, 64), color=(64, 64, 64)),
                "scalebar_img": Image.new("RGB", (64, 64), color=(32, 32, 32)),
                "scientificName": "Unknown",
                "domainID": 0,
            }
        )
    pred = model.predict(dummy)
    if not isinstance(pred, dict) or "SPEI_30d" not in pred:
        raise ValueError("Smoke test failed: predict output format invalid.")


def main():
    save_dir = Path(__file__).resolve().parent
    model_path = save_dir / "model.pth"
    model = Model(checkpoint_path=str(model_path))
    model.load()

    print("Running smoke test...")
    smoke_test(model)
    print("Smoke test OK")

    ds = load_dataset("imageomics/sentinel-beetles", split="validation")
    events = _gather_event_rows(ds, max_events=5)

    all_mu = []
    all_sigma = []
    all_gts = []

    print("Running validation on first 5 events...")
    for indices in tqdm(events):
        rows = [ds[i] for i in indices]
        if not rows:
            continue
        pred = model.predict(rows)
        mu, sigma = _pred_to_arrays(pred)
        all_mu.append(mu)
        all_sigma.append(sigma)
        all_gts.append(
            [
                float(rows[0]["SPEI_30d"]),
                float(rows[0]["SPEI_1y"]),
                float(rows[0]["SPEI_2y"]),
            ]
        )

    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    all_gts = np.array(all_gts)

    crps_vals = gaussian_crps(all_gts, all_mu, all_sigma)
    crps_per_target = np.mean(crps_vals, axis=0)
    crps_mean = float(np.mean(crps_per_target))

    r2_vals = evaluate_spei_r2_scores(all_gts, all_mu)
    valid_r2 = [r for r in r2_vals if r is not None and np.isfinite(r)]
    r2_mean = float(np.mean(valid_r2)) if valid_r2 else float("nan")

    print("\nValidation (first 5 events):")
    print(f"CRPS SPEI_30d: {crps_per_target[0]:.4f}")
    print(f"CRPS SPEI_1y:  {crps_per_target[1]:.4f}")
    print(f"CRPS SPEI_2y:  {crps_per_target[2]:.4f}")
    print(f"CRPS Mean:    {crps_mean:.4f}")
    print(f"R2 SPEI_30d:  {r2_vals[0] if r2_vals[0] is not None else float('nan'):.4f}")
    print(f"R2 SPEI_1y:   {r2_vals[1] if r2_vals[1] is not None else float('nan'):.4f}")
    print(f"R2 SPEI_2y:   {r2_vals[2] if r2_vals[2] is not None else float('nan'):.4f}")
    print(f"R2 Mean:     {r2_mean:.4f}")


if __name__ == "__main__":
    main()
