import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from datasets import load_dataset
from tqdm import tqdm

# Import your custom modules
from utils import get_training_args, get_dino_and_transforms, get_collate_fn
from model import GrandBeetleModel

def load_mappings(save_dir):
    """Reloads the exact integer mappings used during training."""
    with open(save_dir / "species_map.json", "r") as f:
        species_map = json.load(f)
    with open(save_dir / "domain_map.json", "r") as f:
        domain_map = json.load(f)
    return species_map, domain_map

def predict_batch(dino, model, batch):
    """Runs the full pipeline: Image -> DINO -> GrandModel -> Prediction"""
    imgs, targets, sp_ids, dom_ids = batch[0], batch[1], batch[2], batch[3]
    imgs = imgs.cuda()
    sp_ids = sp_ids.cuda()
    dom_ids = dom_ids.cuda()

    # 1. Extract Spatial Features (DINO)
    with torch.no_grad():
        outputs = dino.forward_features(imgs)
        features = outputs["x_norm_patchtokens"] # (B, 256, 768)

        # Reshape for CNN head: (B, 768, 16, 16)
        B, N, C = features.shape
        H = W = int(N**0.5)
        features = features.transpose(1, 2).reshape(B, C, H, W)

        # 2. Predict (GrandModel)
        preds = model(features, sp_ids, dom_ids, domain_dropout_prob=0.0)

    return preds.cpu().numpy(), targets.numpy(), imgs.cpu()

def plot_scatter(gts, preds, save_dir):
    """Generates correlation plots for 30d, 1y, and 2y SPEI."""
    targets = ["SPEI_30d", "SPEI_1y", "SPEI_2y"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, ax in enumerate(axes):
        sns.regplot(x=gts[:, i], y=preds[:, i], ax=ax, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        r2 = r2_score(gts[:, i], preds[:, i])
        mae = mean_absolute_error(gts[:, i], preds[:, i])

        ax.set_title(f"{targets[i]}\nR2: {r2:.3f} | MAE: {mae:.3f}")
        ax.set_xlabel("Actual SPEI")
        ax.set_ylabel("Predicted SPEI")
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_dir / "evaluation_matrix.png")
    print(f"📊 Scatter plots saved to {save_dir / 'evaluation_matrix.png'}")

def visualize_inference(imgs, preds, gts, sp_ids, dom_ids, species_map, save_dir):
    """Shows actual images with predictions."""
    # Reverse mapping to get names back
    idx_to_species = {v: k for k, v in species_map.items()}

    # Denormalize images for plotting (approximate ImageNet mean/std)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i in range(min(8, len(imgs))):
        img = imgs[i] * std + mean # Denormalize
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()

        ax = axes[i]
        ax.imshow(img)

        sp_name = idx_to_species.get(sp_ids[i].item(), "Unknown")
        # Shorten name
        sp_name = " ".join(sp_name.split()[:2])

        # Compare 1-Year SPEI (Index 1)
        actual = gts[i][1]
        pred = preds[i][1]
        diff = abs(pred - actual)
        color = "green" if diff < 0.5 else "red"

        ax.set_title(f"{sp_name}\nAct: {actual:.2f} | Pred: {pred:.2f}", color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / "inference_examples.png")
    print(f"🖼️  Inference examples saved to {save_dir / 'inference_examples.png'}")

def main():
    args = get_training_args()
    save_dir = Path(__file__).resolve().parent

    # 1. Load Mappings
    species_map, domain_map = load_mappings(save_dir)
    print(f"✅ Loaded {len(species_map)} species and {len(domain_map)} domains.")

    # 2. Load Models
    # A. DINO (Feature Extractor)
    dino, transforms = get_dino_and_transforms()

    # B. GrandModel (Regressor)
    model = GrandBeetleModel(
        backbone=None,
        num_species=len(species_map),
        num_domains=len(domain_map),
        backbone_dim=768
    ).cuda()

    # Load Weights
    model_path = save_dir / "best.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("✅ Model weights loaded.")

    # 3. Prepare Validation Data
    ds = load_dataset("imageomics/sentinel-beetles", split="validation", token=args.hf_token)

    def preprocess(examples):
        examples["pixel_values"] = [transforms(img.convert("RGB")) for img in examples["file_path"]]
        examples["species_idx"] = [species_map.get(s, 0) for s in examples["scientificName"]]
        examples["domain_idx"] = [domain_map.get(d, 0) for d in examples["domainID"]]
        return examples

    val_ds = ds.with_transform(preprocess)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=get_collate_fn(other_columns=["species_idx", "domain_idx"])
    )

    # 4. Evaluation Loop
    all_preds = []
    all_gts = []
    all_imgs = [] # Keep a few for visualization
    all_sp = []

    print("🚀 Running Inference...")
    for batch in tqdm(val_loader):
        preds, gts, imgs = predict_batch(dino, model, batch)

        all_preds.append(preds)
        all_gts.append(gts)

        # Save first batch for visualization
        if len(all_imgs) == 0:
            all_imgs = imgs
            all_sp = batch[2] # Species IDs

    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)

    # 5. Generate Outputs
    # A. Metrics
    r2_30d, r2_1y, r2_2y = r2_score(all_gts, all_preds, multioutput='raw_values')
    print("\n" + "="*30)
    print(f"🏆 Final R2 Scores:")
    print(f"  SPEI 30d: {r2_30d:.4f}")
    print(f"  SPEI 1y:  {r2_1y:.4f}")
    print(f"  SPEI 2y:  {r2_2y:.4f}")
    print("="*30 + "\n")

    # B. Matrix Plot
    plot_scatter(all_gts, all_preds, save_dir)

    # C. Visual Examples
    visualize_inference(all_imgs, all_preds, all_gts, all_sp, None, species_map, save_dir)

if __name__ == "__main__":
    main()