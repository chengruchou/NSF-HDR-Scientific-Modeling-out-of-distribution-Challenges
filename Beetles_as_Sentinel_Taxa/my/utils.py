from datetime import date
import json
from argparse import ArgumentParser

import numpy as np
import torch
from transformers import AutoModel
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import r2_score
from open_clip import create_model_and_transforms

def get_collate_fn(other_columns=None):
    def collate_fn(batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        spei_values = torch.stack([torch.tensor([example["SPEI_30d"], example["SPEI_1y"], example["SPEI_2y"]]) for example in batch])
        rv = [pixel_values, spei_values]

        if other_columns:
            for col in other_columns:
                vals = [example[col] for example in batch]

                if isinstance(vals[0], (int, float, np.integer)):
                     rv.append(torch.tensor(vals))
                else:
                    rv.append(vals)

        return rv

    return collate_fn

def get_str_date():
    today = date.today()
    str_date = today.strftime("%m-%d-%Y")
    return str_date


def compile_event_predictions(all_gts, all_preds, all_events):
    # Convert to NumPy arrays
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    all_events = np.array(all_events)

    # Aggregate by unique event ID
    unique_events = np.unique(all_events)
    preds_event = []
    gts_event = []

    for uevent in unique_events:
        indices = np.where(all_events == uevent)[0]
        if indices.size == 0:
            continue  # skip if no matching indices

        preds_event.append(all_preds[indices].mean(axis=0))
        gts_event.append(all_gts[indices].mean(axis=0))

    preds_event = np.stack(preds_event)
    gts_event = np.stack(gts_event)

    return gts_event, preds_event


def extract_deep_features(dataloader, model):
    X = []
    Y = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting Features"):
            X.append(model.forward_frozen(x.cuda()).detach().cpu())
            Y.append(y)

    X = torch.cat(X)
    Y = torch.cat(Y)

    return X, Y


def extract_deep_features_with_domain_id(dataloader, model):
    X = []
    Y = []
    DID = []
    with torch.no_grad():
        for x, y, did in tqdm(dataloader, desc="Extracting Features"):
            X.append(model.forward_frozen(x.cuda()).detach().cpu())
            Y.append(y)
            DID.append(did)

    X = torch.cat(X)
    Y = torch.cat(Y)
    DID = torch.cat(DID)

    return X, Y, DID


def extract_bioclip_features(dataloader, bioclip, eventID=False):
    X = []
    Y = []
    EIDs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            x, y = batch[:2]
            X.append(bioclip(x.cuda())["image_features"].detach().cpu())
            Y.append(y)

            if eventID:
                EIDs.append(batch[2])

    X = torch.cat(X)
    Y = torch.cat(Y)

    if eventID:
        EIDs = torch.cat(EIDs)
        return X, Y, EIDs

    return X, Y


def extract_dino_features(dataloader, dino, eventID=False):
    X = []
    Y = []
    EIDs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            x, y = batch[:2]
            features = dino(x.cuda())[0][:, 1:]
            # adjust features so Bx256x768 -> B x 768 x 16x16 so that positional data can be conserved(hopefully)?
            transposed_patches = features.transpose(1, 2)  # -> Bx768x256
            unflat = transposed_patches.unflatten(dim=2, sizes=(16, 16))
            X.append(unflat.detach().cpu())
            Y.append(y)

            if eventID:
                EIDs.append(batch[2])

    X = torch.cat(X)
    Y = torch.cat(Y)

    if eventID:
        EIDs = torch.cat(EIDs)
        return X, Y, EIDs

    return X, Y


def evalute_spei_r2_scores(gts, preds):
    spei_30_r2 = r2_score(gts[:, 0], preds[:, 0])
    spei_1y_r2 = r2_score(gts[:, 1], preds[:, 1])
    spei_2y_r2 = r2_score(gts[:, 2], preds[:, 2])
    return spei_30_r2, spei_1y_r2, spei_2y_r2


def get_bioclip():
    """function that returns frozen bioclip model

    model: bioclip
    """
    # bioclip = create_model("hf-hub:imageomics/bioclip-2", output_dict=True, require_pretrained=True).cuda()
    bioclip, _, preprocess = create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2", output_dict=True, require_pretrained=True
    )
    bioclip = bioclip.cuda()
    return bioclip, preprocess

def get_training_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n_last_trainable_blocks", type=int, default=2)
    parser.add_argument("--domain_id_aug_prob", type=float, default=0.2)
    parser.add_argument("--hf_token", type=str, default=None)

    return parser.parse_args()


def save_results(save_path, mae_scores, r2_scores):
    with open(save_path, "w") as f:
        save_data = {}
        for i, tgt in enumerate(["SPEI_30d", "SPEI_1y", "SPEI_2y"]):
            save_data[tgt] = {
                "MAE": mae_scores[i],
                "r2": r2_scores[i],
            }
        json.dump(save_data, f)

def extract_spatial_features_with_metadata(dataloader, backbone_model, backbone_type="dino"):
    X_spatial = []
    Y = []
    SP_ID = []
    DOM_ID = []

    backbone_model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Spatial Features"):
            # Unpack (Images, Targets, Species, Domain)
            imgs, targets, species_ids, domain_ids = batch[0], batch[1], batch[2], batch[3]
            imgs = imgs.cuda()

            if backbone_type == "dino":
                # DINOv2 forward_features returns a dict
                outputs = backbone_model.forward_features(imgs)

                # 'x_norm_patchtokens' is the patch grid (Batch, Num_Patches, Dim)
                # For 224x224 image & patch size 14, Num_Patches = 256 (16x16)
                features = outputs["x_norm_patchtokens"]

                # Reshape: (B, 256, 768) -> (B, 768, 16, 16) for our CNN head
                B, N, C = features.shape
                H_grid = W_grid = int(N**0.5) # Should be 16

                # Transpose to (B, C, N) then reshape
                features = features.transpose(1, 2).reshape(B, C, H_grid, W_grid)

            elif backbone_type == "bioclip":
                 # Fallback for BioCLIP if needed
                 # (Requires specific BioCLIP logic, omitting for brevity as you asked for DINO)
                 pass

            X_spatial.append(features.cpu())
            Y.append(targets)
            SP_ID.append(species_ids)
            DOM_ID.append(domain_ids)

    # Concatenate all
    return torch.cat(X_spatial), torch.cat(Y), torch.cat(SP_ID), torch.cat(DOM_ID)

def get_dino_and_transforms():
    """
    Loads DINOv2 (ViT-B/14) and standard ImageNet transforms.
    """
    print("🦕 Loading DINOv2 model...")
    # Load from Torch Hub (requires internet)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = model.cuda()
    model.eval()

    # DINO Standard Transforms (ImageNet stats)
    # We resize to 224x224 to get a clean 16x16 feature grid (14*16 = 224)
    dino_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, dino_transform