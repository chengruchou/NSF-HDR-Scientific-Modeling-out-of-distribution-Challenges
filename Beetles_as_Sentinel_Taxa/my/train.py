import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datasets import load_dataset

from utils import (
    get_training_args,
    get_dino_and_transforms, # Changed to DINO loader
    evalute_spei_r2_scores,
    extract_spatial_features_with_metadata,
    get_collate_fn,
)
from model import GrandBeetleModel

def train(model, train_loader, val_loader, lr, epochs, save_dir, domain_aug_prob):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    save_path = Path(save_dir) / "dino_grand_model.pth"
    best_r2 = -float('inf')

    print(f"🚀 Training with {len(train_loader.dataset)} samples")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        preds, gts = [], []

        for feats, targets, sp_ids, dom_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            feats, targets = feats.cuda(), targets.cuda()
            sp_ids, dom_ids = sp_ids.cuda(), dom_ids.cuda()

            optimizer.zero_grad()
            outputs = model(feats, sp_ids, dom_ids, domain_dropout_prob=domain_aug_prob)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds.extend(outputs.detach().cpu().numpy())
            gts.extend(targets.detach().cpu().numpy())

        # Validation
        model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for feats, targets, sp_ids, dom_ids in val_loader:
                feats, targets = feats.cuda(), targets.cuda()
                sp_ids, dom_ids = sp_ids.cuda(), dom_ids.cuda()
                outputs = model(feats, sp_ids, dom_ids, domain_dropout_prob=0.0)
                val_preds.extend(outputs.detach().cpu().numpy())
                val_gts.extend(targets.detach().cpu().numpy())

        # Metrics
        val_r2 = evalute_spei_r2_scores(np.array(val_gts), np.array(val_preds))
        avg_val_r2 = sum(val_r2) / 3.0

        if avg_val_r2 > best_r2:
            best_r2 = avg_val_r2
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | Val R2 {avg_val_r2:.4f} (Best: {best_r2:.4f})")

def main():
    args = get_training_args()
    save_dir = Path(__file__).resolve().parent

    # 1. Load Dataset
    print("📂 Loading Dataset...")
    ds = load_dataset("imageomics/sentinel-beetles", token=args.hf_token)

    # Mappings
    all_species = set(ds['train']['scientificName']) | set(ds['validation']['scientificName'])
    species_map = {name: i+1 for i, name in enumerate(sorted(all_species))}
    species_map['Unknown'] = 0

    all_domains = set(ds['train']['domainID']) | set(ds['validation']['domainID'])
    domain_map = {name: i+1 for i, name in enumerate(sorted(all_domains))}
    domain_map['Unknown'] = 0

    # 2. Load DINO & Preprocess
    backbone, transforms = get_dino_and_transforms()

    def preprocess(examples):
        # DINO expects 224x224 RGB
        examples["pixel_values"] = [transforms(img.convert("RGB")) for img in examples["file_path"]]
        examples["species_idx"] = [species_map.get(s, 0) for s in examples["scientificName"]]
        examples["domain_idx"] = [domain_map.get(d, 0) for d in examples["domainID"]]
        return examples

    train_ds = ds["train"].with_transform(preprocess)
    val_ds = ds["validation"].with_transform(preprocess)

    # 3. Extract Features (FIX: num_workers=0)
    dataloaders = {}
    extracted_data = {}

    for split, dset in zip(["train", "val"], [train_ds, val_ds]):
        print(f"📸 Extracting DINO features for {split}...")

        # ERROR FIX IS HERE: num_workers=0
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=get_collate_fn(other_columns=["species_idx", "domain_idx"])
        )

        feats, targets, sp_ids, dom_ids = extract_spatial_features_with_metadata(
            loader, backbone, backbone_type="dino"
        )

        extracted_data[split] = DataLoader(
            TensorDataset(feats, targets, sp_ids, dom_ids),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers # Can use workers here since data is in RAM
        )

    # 4. Clean up DINO to save VRAM
    del backbone
    torch.cuda.empty_cache()

    # 5. Initialize Model
    # DINO ViT-B/14 output dim is 768
    model = GrandBeetleModel(
        backbone=None,
        num_species=len(species_map),
        num_domains=len(domain_map),
        backbone_dim=768
    ).cuda()

    train(model, extracted_data["train"], extracted_data["val"], args.lr, args.epochs, save_dir, args.domain_id_aug_prob)

if __name__ == "__main__":
    main()