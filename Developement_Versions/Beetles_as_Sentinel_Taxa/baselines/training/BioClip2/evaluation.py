from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset


from utils import (
    get_training_args,
    get_bioclip,
    evalute_spei_r2_scores,
    save_results,
    compile_event_predictions,
    get_collate_fn,
)
from model import BioClip2_DeepRegressor


def evaluate(model, dataloader):
    with torch.inference_mode():
        abs_error = 0
        tbar = tqdm(dataloader, desc="Evaluating model")
        all_preds = []
        all_gts = []
        all_events = []
        for imgs, targets, eventIDs in tbar:
            imgs = imgs.cuda()
            targets = targets.cuda()

            outputs = model(imgs)
            abs_error += torch.mean(torch.abs(outputs - targets), dim=0)

            all_preds.extend(outputs.detach().cpu().numpy())
            all_gts.extend(targets.detach().cpu().numpy())
            all_events.extend(np.array(eventIDs))

        gts_event, preds_event = compile_event_predictions(
            all_gts, all_preds, all_events
        )

        gts = gts_event
        preds = preds_event
        spei_30_r2, spei_1y_r2, spei_2y_r2 = evalute_spei_r2_scores(gts, preds)

        MAE = abs_error / len(dataloader)
        print(f"test loss MAE SPEI_30d {MAE[0].item()}")
        print(f"test loss MAE SPEI_1y {MAE[1].item()}")
        print(f"test loss MAE SPEI_2y {MAE[2].item()}")

        print(f"test r2 SPEI_30d {spei_30_r2}")
        print(f"test r2 SPEI_1y {spei_1y_r2}")
        print(f"test r2 SPEI_2y {spei_2y_r2}")

    return [x.item() for x in MAE], [spei_30_r2, spei_1y_r2, spei_2y_r2]


def test_and_save(
    test_dataset,
    save_path,
    batch_size,
    num_workers,
    model,
):
    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=get_collate_fn(["eventID"]),
    )

    mae_scores, r2_scores = evaluate(dataloader=dataloader, model=model)

    save_results(save_path, mae_scores, r2_scores)


def main():
    args = get_training_args()
    save_dir = Path(__file__).resolve().parent

    # load model
    bioclip, transforms = get_bioclip()
    model = BioClip2_DeepRegressor(bioclip).cuda()
    model.regressor.load_state_dict(torch.load(save_dir / "model.pth"))

    # Get datasets
    ds = load_dataset(
        "imageomics/sentinel-beetles",
        token=args.hf_token,
        split="validation",
    )

    # Transform images for model input
    def dset_transforms(examples):
        examples["pixel_values"] = [
            transforms(img.convert("RGB")) for img in examples["file_path"]
        ]
        return examples

    test_dset = ds.with_transform(dset_transforms)

    # evaluate model
    test_and_save(
        test_dset,
        save_dir / "results.json",
        args.batch_size,
        args.num_workers,
        model,
    )


if __name__ == "__main__":
    main()
