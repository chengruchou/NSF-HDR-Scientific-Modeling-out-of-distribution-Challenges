# v4 vs v3:
# - Horizon-weighted robust loss + per-horizon heads support.
# - Backbone unfreeze schedule tuned for long-horizon stability.
# - Train-time augmentation toggle via build_transforms(train=True).
# Why OOD improves:
# - Extra weight on SPEI_2y reduces long-horizon drift.
# - Gradual unfreeze lowers overfitting risk.
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

from utils import (
    build_transforms,
    collate_events,
    evaluate_spei_r2_scores,
    EventDataset,
    get_training_args,
    gaussian_crps,
)
from model_v1 import EventMILModel
from model_v2 import EventMILModelV2, Model as InferenceModelV2
from model_v3 import EventMILModelV3
from model_v4 import EventMILModelV4, Model as InferenceModelV4
from model_v5 import EventMILModelV5, Model as InferenceModelV5
from model_v6 import EventMILModelV6, Model as InferenceModelV6


def _gaussian_nll(mu, sigma, target):
    var = sigma ** 2
    return 0.5 * ((target - mu) ** 2 / var + torch.log(var)).mean()


def _gaussian_crps(mu, sigma, target):
    z = (target - mu) / sigma
    sqrt_2 = math.sqrt(2.0)
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / sqrt_2))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return crps.mean()


def _extract_mu_sigma(outputs, model_version):
    if model_version in ("v3", "v4", "v5"):
        mu = outputs["mu"]
        sigma = outputs["sigma"]
        return mu, sigma
    mu = outputs[:, :3]
    log_sigma = outputs[:, 3:].clamp(-6.0, 3.0)
    sigma = F.softplus(log_sigma) + 1e-3
    return mu, sigma


def _set_backbone_requires_grad(model, layer_prefixes, requires_grad):
    for name, param in model.backbone.named_parameters():
        for prefix in layer_prefixes:
            if name.startswith(prefix + ".") or name == prefix:
                param.requires_grad = requires_grad
                break


def _backbone_layer_trainable(model, layer_prefix):
    for name, param in model.backbone.named_parameters():
        if name.startswith(layer_prefix + ".") or name == layer_prefix:
            if param.requires_grad:
                return True
    return False


def train(
    model,
    train_loader,
    val_loader,
    lr,
    epochs,
    save_path,
    domain_aug_prob,
    target_mean,
    target_std,
    amp_enabled,
    grad_accum,
    image_size,
    calib_size,
    max_instances,
    model_version,
):
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]

    backbone_lr = lr * 0.1 if model_version == "v1" else 0.0
    optimizer = optim.AdamW(
        [
            {"params": head_params, "lr": lr},
            {"params": backbone_params, "lr": backbone_lr},
        ]
    )
    scaler = GradScaler("cuda", enabled=amp_enabled)
    best_val_metric = float("inf")
    patience = 5
    patience_left = patience
    device = next(model.parameters()).device

    print(f"Training with {len(train_loader.dataset)} events")
    print(f"Initial Backbone LR: {optimizer.param_groups[1]['lr']:.6f}")

    for epoch in range(epochs):
        epoch_num = epoch + 1
        if model_version == "v4":
            if epoch_num == 3 and not _backbone_layer_trainable(model, "layer4"):
                _set_backbone_requires_grad(model, ["layer4"], True)
                optimizer.param_groups[1]["lr"] = 1e-5
                print("Unfreeze layer4 at epoch 3")
                print(f"Backbone LR: {optimizer.param_groups[1]['lr']:.6f}")
            if epoch_num == 6 and not _backbone_layer_trainable(model, "layer3"):
                _set_backbone_requires_grad(model, ["layer3"], True)
                optimizer.param_groups[1]["lr"] = 5e-6
                print("Unfreeze layer3 at epoch 6")
                print(f"Backbone LR: {optimizer.param_groups[1]['lr']:.6f}")
            if epoch_num == 10 and not _backbone_layer_trainable(model, "layer2"):
                _set_backbone_requires_grad(
                    model, ["conv1", "bn1", "layer1", "layer2"], True
                )
                optimizer.param_groups[1]["lr"] = 3e-6
                print("Unfreeze all backbone layers at epoch 10")
                print(f"Backbone LR: {optimizer.param_groups[1]['lr']:.6f}")
        elif model_version in ("v2", "v3", "v5"):
            if epoch_num == 4 and not _backbone_layer_trainable(model, "layer4"):
                _set_backbone_requires_grad(
                    model, ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"], True
                )
                optimizer.param_groups[1]["lr"] = 1e-5
                print("Unfreeze all backbone layers at epoch 4")
                print(f"Backbone LR: {optimizer.param_groups[1]['lr']:.6f}")

        elif model_version == "v6":
            if epoch_num == 4 and not _backbone_layer_trainable(model, "stages.3"):
                convnext_layers = ["stem", "stages.0", "stages.1", "stages.2", "stages.3"]

                _set_backbone_requires_grad(model, convnext_layers, True)

                optimizer.param_groups[1]["lr"] = 1e-5
                print("Unfreeze all ConvNeXt backbone layers at epoch 4")
                print(f"Backbone LR: {optimizer.param_groups[1]['lr']:.6f}")
        else:
            if epoch_num == 10 and not _backbone_layer_trainable(model, "layer3"):
                _set_backbone_requires_grad(model, ["layer3"], True)
                print("Unfreeze layer3 at epoch 10")
                print(f"Backbone LR: {optimizer.param_groups[1]['lr']:.6f}")
            if epoch_num == 15 and not _backbone_layer_trainable(model, "layer2"):
                _set_backbone_requires_grad(model, ["layer2"], True)
                optimizer.param_groups[1]["lr"] = lr * 0.01
                print("Unfreeze layer2 at epoch 15")
                print(f"Backbone LR: {optimizer.param_groups[1]['lr']:.6f}")

        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        step = 0

        try:
            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch_num}"), start=1):
                images = batch["images"].to(device, non_blocking=True)
                species_ids = batch["species_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                color = batch["color"].to(device, non_blocking=True)
                scale = batch["scale"].to(device, non_blocking=True)
                domain_id = batch["domain_id"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)

                targets_norm = (targets - target_mean) / target_std
                with autocast(device_type="cuda", enabled=amp_enabled):
                    outputs = model(
                        images,
                        species_ids,
                        attention_mask,
                        color,
                        scale,
                        domain_id,
                        domain_aug_prob,
                    )
                    mu, sigma = _extract_mu_sigma(outputs, model_version)
                    if model_version == "v4":
                        weights = torch.tensor([1.0, 1.2, 1.5], device=mu.device, dtype=mu.dtype)
                        huber = F.smooth_l1_loss(mu, targets_norm, beta=0.5, reduction="none")
                        var = sigma ** 2
                        nll = 0.5 * ((targets_norm - mu) ** 2 / var + torch.log(var))
                        per_horizon = []
                        for i in range(3):
                            huber_i = huber[:, i].mean()
                            nll_i = nll[:, i].mean()
                            per_horizon.append(weights[i] * (huber_i + 0.05 * nll_i))
                        loss = torch.stack(per_horizon).sum() + 0.005 * sigma.mean()
                    elif model_version == "v3":
                        # RMSE-optimized v3: prioritize mean accuracy, keep sigma as light regularizer.
                        mse = F.mse_loss(mu, targets_norm)
                        nll = _gaussian_nll(mu, sigma, targets_norm)
                        sigma_reg = (sigma - sigma.mean()).pow(2).mean()
                        loss = mse + 0.1 * nll + 0.01 * sigma_reg
                    elif model_version == "v5":
                        loss = gaussian_crps(targets_norm, mu, sigma).mean() + 1e-4 * sigma.mean()
                        if "attn_weights" in outputs:
                            attn = outputs["attn_weights"]
                            if attention_mask is not None:
                                valid = attention_mask.float()
                                attn = attn * valid
                                denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-12)
                                attn = attn / denom
                            attn_entropy = -(attn * torch.log(attn.clamp_min(1e-12))).sum(dim=1).mean()
                            loss = loss + 1e-4 * (-attn_entropy)
                    else:
                        loss = _gaussian_crps(mu, sigma, targets_norm)
                        if not torch.isfinite(loss):
                            loss = _gaussian_nll(mu, sigma, targets_norm)
                        loss = loss + 1e-3 * sigma.mean()
                    loss = loss / max(grad_accum, 1)

                scaler.scale(loss).backward()
                if step % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item() * max(grad_accum, 1)

            if step > 0 and step % grad_accum != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    "CUDA OOM: try reducing batch_size, max_instances, image_size, "
                    "or enable amp / increase grad_accum."
                )
            raise

        model.eval()
        val_preds, val_gts = [], []
        val_crps = 0.0
        val_batches = 0
        tm = target_mean.detach().cpu().numpy()
        ts = target_std.detach().cpu().numpy()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device, non_blocking=True)
                species_ids = batch["species_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                color = batch["color"].to(device, non_blocking=True)
                scale = batch["scale"].to(device, non_blocking=True)
                domain_id = batch["domain_id"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=amp_enabled):
                    outputs = model(images, species_ids, attention_mask, color, scale, domain_id, 0.0)
                    mu, sigma = _extract_mu_sigma(outputs, model_version)
                mu_np = mu.detach().cpu().numpy()
                sigma_np = sigma.detach().cpu().numpy()
                mu_denorm = mu_np * ts + tm
                sigma_denorm = sigma_np * ts
                targets_np = targets.detach().cpu().numpy()
                val_preds.extend(mu_denorm)
                val_gts.extend(targets_np)

                batch_crps = gaussian_crps(targets_np, mu_denorm, sigma_denorm)
                val_crps += float(np.mean(batch_crps))
                val_batches += 1

        val_r2 = evaluate_spei_r2_scores(np.array(val_gts), np.array(val_preds))
        valid_r2 = [score for score in val_r2 if score is not None and np.isfinite(score)]
        avg_val_r2 = float(np.mean(valid_r2)) if valid_r2 else float("nan")
        avg_val_crps = val_crps / max(val_batches, 1)
        val_gts_np = np.array(val_gts)
        val_preds_np = np.array(val_preds)
        val_rmse = float(np.sqrt(np.mean((val_preds_np - val_gts_np) ** 2)))
        if model_version == "v3":
            monitor_value = val_rmse
            monitor_label = "RMSE"
        else:
            monitor_value = avg_val_crps
            monitor_label = "CRPS"

        if monitor_value < best_val_metric:
            best_val_metric = monitor_value
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "species_to_idx": model.species_to_idx,
                    "domain_to_idx": model.domain_to_idx,
                    "target_mean": target_mean.detach().cpu().tolist(),
                    "target_std": target_std.detach().cpu().tolist(),
                    "backbone_name": "resnet50",
                    "model_version": model_version,
                    "image_size": image_size,
                    "calib_size": calib_size,
                    "max_instances": max_instances,
                },
                save_path,
            )
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch_num} with best Val {monitor_label} {best_val_metric:.4f}")
                break

        print(
            f"Epoch {epoch_num}: Train Loss {train_loss:.4f} | "
            f"Val CRPS {avg_val_crps:.4f} | "
            f"Val RMSE {val_rmse:.4f} (Best {monitor_label}: {best_val_metric:.4f}) | "
            f"Val R2 {avg_val_r2:.4f}"
        )


def main():
    args = get_training_args()
    if args.sanity_check:
        quick_sanity_check()
        return
    save_dir = Path(__file__).resolve().parent
    if args.model_version == "v6" and args.checkpoint_name == "model_v6.pth":
        args.checkpoint_name = "model_v6.pth"
    save_path = save_dir / args.checkpoint_name
    print(f"Model will be saved to: {save_path}")
    print("Loading Dataset...")
    ds = load_dataset("imageomics/sentinel-beetles")

    all_species = set(ds["train"]["scientificName"]) | set(ds["validation"]["scientificName"])
    species_map = {name: i + 1 for i, name in enumerate(sorted(all_species))}
    species_map["Unknown"] = 0

    all_domains = set(ds["train"]["domainID"]) | set(ds["validation"]["domainID"])
    domain_map = {}
    next_idx = 1
    for name in sorted(all_domains):
        try:
            key = int(name)
        except (TypeError, ValueError):
            key = 0
        if key not in domain_map and key != 0:
            domain_map[key] = next_idx
            next_idx += 1
    domain_map[0] = 0

    img_transform, calib_transform = build_transforms(
        args.image_size,
        args.calib_size,
        train=True,
        model_version=args.model_version,
    )
    train_events = EventDataset(
        ds["train"],
        species_map,
        domain_map,
        img_transform,
        calib_transform,
        max_instances=args.max_instances,
        sample_strategy=args.sample_strategy,
    )
    val_img_transform, val_calib_transform = build_transforms(
        args.image_size,
        args.calib_size,
        train=False,
        model_version=args.model_version,
    )
    val_events = EventDataset(
        ds["validation"],
        species_map,
        domain_map,
        val_img_transform,
        val_calib_transform,
        max_instances=args.max_instances,
        sample_strategy=args.sample_strategy,
    )

    train_targets = torch.tensor(
        [[ex["SPEI_30d"], ex["SPEI_1y"], ex["SPEI_2y"]] for ex in ds["train"]],
        dtype=torch.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_mean = train_targets.mean(dim=0).to(device)
    target_std = train_targets.std(dim=0).clamp_min(1e-6).to(device)

    train_loader = DataLoader(
        train_events,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_events,
        pin_memory=True,
    )
    print(f"train_loader with {len(train_loader.dataset)} events")
    val_loader = DataLoader(
        val_events,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_events,
        pin_memory=True,
    )
    print(f"val_loader with {len(val_loader.dataset)} events")

    if args.model_version == "v2":
        model = EventMILModelV2(num_species=len(species_map), num_domains=len(domain_map))
    elif args.model_version == "v3":
        model = EventMILModelV3(num_species=len(species_map), num_domains=len(domain_map))
    elif args.model_version == "v4":
        model = EventMILModelV4(num_species=len(species_map), num_domains=len(domain_map))
    elif args.model_version == "v5":
        model = EventMILModelV5(num_species=len(species_map), num_domains=len(domain_map))
    elif args.model_version == "v6":
        model = EventMILModelV6(num_species=len(species_map), num_domains=len(domain_map))
    else:
        model = EventMILModel(num_species=len(species_map), num_domains=len(domain_map))
    model.species_to_idx = species_map
    model.domain_to_idx = domain_map
    if args.model_version == "v1" and args.freeze_backbone:
        model.freeze_backbone_stages(args.freeze_backbone)
        print(f"Backbone frozen until layer{args.freeze_backbone} (only upper layers + head trainable)")
    if args.model_version in ("v2", "v3", "v4", "v5", "v6"):
        _set_backbone_requires_grad(
            model, ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"], False
        )
        if args.model_version == "v4":
            print("Backbone frozen for first 2 epochs")
        else:
            print("Backbone frozen for first 3 epochs")
    model = model.to(device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    train(
        model,
        train_loader,
        val_loader,
        args.lr,
        args.epochs,
        save_path,
        args.domain_id_aug_prob,
        target_mean,
        target_std,
        amp_enabled,
        args.grad_accum,
        args.image_size,
        args.calib_size,
        args.max_instances,
        args.model_version,
    )


def quick_sanity_check():
    print("Running sanity check...")
    import evaluation  # noqa: F401
    import utils  # noqa: F401
    import model_v2  # noqa: F401
    import model_v4  # noqa: F401
    import model_v5  # noqa: F401
    import train  # noqa: F401

    checkpoint = Path(__file__).resolve().parent / "model_v6.pth"
    model = InferenceModelV6(checkpoint_path=str(checkpoint))
    if checkpoint.exists():
        model.load()
        print("Loaded checkpoint for sanity check.")
    else:
        print("Checkpoint not found; skipping predict.")
        return

    ds = load_dataset("imageomics/sentinel-beetles")
    if len(ds["validation"]) == 0:
        print("No validation samples available; skipping predict.")
        return

    sample_rows = [ds["validation"][0]]
    if len(ds["validation"]) > 1:
        sample_rows.append(ds["validation"][1])
    batch = []
    for row in sample_rows:
        batch.append(
            {
                "scientificName": row.get("scientificName", "Unknown"),
                "domainID": row.get("domainID", 0),
                "relative_img": row.get("relative_img"),
                "file_path": row.get("file_path"),
                "image": row.get("image"),
                "colorpicker_img": row.get("colorpicker_img"),
                "colorpicker_path": row.get("colorpicker_path"),
                "scalebar_img": row.get("scalebar_img"),
                "scalebar_path": row.get("scalebar_path"),
            }
        )
    pred = model.predict(batch)
    print("Sanity check prediction keys:", list(pred.keys()))
    print(
        "Sanity check prediction shapes:",
        {k: {sub: np.shape(val) for sub, val in v.items()} for k, v in pred.items()},
    )


if __name__ == "__main__":
    main()
