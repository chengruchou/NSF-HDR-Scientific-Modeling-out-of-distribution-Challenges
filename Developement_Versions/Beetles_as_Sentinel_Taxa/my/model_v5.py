import os
from typing import Dict, List, Optional

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.2, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        if x.dim() == 4:
            mu = x.mean(dim=(2, 3), keepdim=True)
            sigma = x.var(dim=(2, 3), keepdim=True, unbiased=False).add(self.eps).sqrt()
            shape = (x.size(0), 1, 1, 1)
        elif x.dim() == 2:
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt()
            shape = (x.size(0), 1)
        else:
            return x

        perm = torch.randperm(x.size(0), device=x.device)
        mu_perm = mu[perm]
        sigma_perm = sigma[perm]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample(shape).to(x.device)

        mu_mix = lam * mu + (1.0 - lam) * mu_perm
        sigma_mix = lam * sigma + (1.0 - lam) * sigma_perm
        x_norm = (x - mu) / sigma
        return x_norm * sigma_mix + mu_mix


class SmallCalibCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class EventMILModelV5(nn.Module):
    def __init__(
        self,
        num_species: int,
        num_domains: int,
        embedding_dim: int = 64,
        domain_emb_dim: int = 32,
        attn_dim: int = 256,
        attn_dropout: float = 0.1,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.2,
    ):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.species_embed = nn.Embedding(num_species, embedding_dim)
        self.domain_embed = nn.Embedding(num_domains, domain_emb_dim)

        attn_in_dim = 2048 + embedding_dim
        self.attn_fc_v = nn.Linear(attn_in_dim, attn_dim)
        self.attn_fc_u = nn.Linear(attn_in_dim, attn_dim)
        self.attn_score = nn.Linear(attn_dim, 1, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)

        self.color_encoder = SmallCalibCNN(out_dim=128)
        self.scale_encoder = SmallCalibCNN(out_dim=128)
        self.calib_mlp = nn.Sequential(
            nn.Linear(128 + 128 + domain_emb_dim, 256),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(attn_in_dim + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

        self.sigma_bias = nn.Parameter(torch.zeros(3))
        self.sigma_scale_u = nn.Parameter(torch.zeros(3))

        self.species_to_idx: Dict[str, int] = {}
        self.domain_to_idx: Dict[int, int] = {}

    def forward(
        self,
        images,
        species_ids,
        attention_mask,
        color,
        scale,
        domain_id,
        domain_dropout_prob=0.0,
    ):
        batch_size, max_n = images.shape[:2]
        flat_images = images.view(batch_size * max_n, *images.shape[2:])
        flat_species = species_ids.view(batch_size * max_n)

        feats = self.backbone(flat_images)
        feats = self.mixstyle(feats)
        feats = feats.view(batch_size, max_n, -1)
        sp_emb = self.species_embed(flat_species).view(batch_size, max_n, -1)
        attn_input = torch.cat([feats, sp_emb], dim=2)

        v = torch.tanh(self.attn_fc_v(attn_input))
        u = torch.sigmoid(self.attn_fc_u(attn_input))
        gated = v * u
        attn_logits = self.attn_score(gated).squeeze(-1)
        if attention_mask is not None:
            attn_min = torch.finfo(attn_logits.dtype).min
            attn_logits = attn_logits.masked_fill(~attention_mask, attn_min)
        attn_logits = self.attn_dropout(attn_logits)
        attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)
        attn_pooled = (attn_weights * attn_input).sum(dim=1)

        if self.training and domain_dropout_prob > 0:
            drop_mask = torch.rand_like(domain_id.float()) < domain_dropout_prob
            domain_id = torch.where(drop_mask, torch.zeros_like(domain_id), domain_id)

        color_feat = self.color_encoder(color)
        scale_feat = self.scale_encoder(scale)
        dom_emb = self.domain_embed(domain_id)
        calib = self.calib_mlp(torch.cat([color_feat, scale_feat, dom_emb], dim=1))

        out = self.head(torch.cat([attn_pooled, calib], dim=1))
        mu = out[:, :3]
        log_sigma_raw = out[:, 3:].clamp(-6.0, 3.0)
        sigma_scale = F.softplus(self.sigma_scale_u) + 1e-3
        log_sigma = (sigma_scale * log_sigma_raw + self.sigma_bias).clamp(-6.0, 3.0)
        sigma = F.softplus(log_sigma) + 1e-3

        outputs = {
            "mu": mu,
            "sigma": sigma,
        }
        if self.training:
            outputs["attn_weights"] = attn_weights.squeeze(-1)
        return outputs


def _build_transforms(image_size: int = 224, calib_size: int = 128):
    img_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    calib_transform = transforms.Compose(
        [
            transforms.Resize(calib_size),
            transforms.CenterCrop(calib_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return img_transform, calib_transform


def _load_image(value: Optional[object]) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        return Image.open(value)
    return None


def _to_int_domain(value) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class Model:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.model = None
        self.image_transform = None
        self.calib_transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.species_to_idx: Dict[str, int] = {}
        self.domain_to_idx: Dict[int, int] = {}
        self.target_mean = torch.zeros(3, dtype=torch.float32)
        self.target_std = torch.ones(3, dtype=torch.float32)
        self.image_size = 224
        self.calib_size = 128
        self.checkpoint_path = checkpoint_path or os.path.join(os.path.dirname(__file__), "model_v5.pth")
        print(f"checkpoint_path: {self.checkpoint_path}")

    def load(self):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint

        self.species_to_idx = checkpoint.get("species_to_idx", {}) if isinstance(checkpoint, dict) else {}
        raw_domain = checkpoint.get("domain_to_idx", {}) if isinstance(checkpoint, dict) else {}
        self.domain_to_idx = {int(k): v for k, v in raw_domain.items()} if raw_domain else {}
        if isinstance(checkpoint, dict) and "target_mean" in checkpoint:
            self.target_mean = torch.tensor(checkpoint["target_mean"], dtype=torch.float32)
        if isinstance(checkpoint, dict) and "target_std" in checkpoint:
            self.target_std = torch.tensor(checkpoint["target_std"], dtype=torch.float32)
        if isinstance(checkpoint, dict) and "image_size" in checkpoint:
            self.image_size = int(checkpoint["image_size"])
        if isinstance(checkpoint, dict) and "calib_size" in checkpoint:
            self.calib_size = int(checkpoint["calib_size"])

        if self.species_to_idx:
            num_species = max(self.species_to_idx.values(), default=0) + 1
        else:
            num_species = state_dict["species_embed.weight"].shape[0]
        if self.domain_to_idx:
            num_domains = max(self.domain_to_idx.values(), default=0) + 1
        else:
            num_domains = state_dict["domain_embed.weight"].shape[0]

        self.model = EventMILModelV5(num_species=num_species, num_domains=num_domains)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.image_transform, self.calib_transform = _build_transforms(
            image_size=self.image_size, calib_size=self.calib_size
        )

    def _encode_metadata(self, batch: List[Dict]):
        species_idx = []
        domain_idx = []
        for entry in batch:
            sp_name = entry.get("scientificName", "Unknown")
            species_idx.append(self.species_to_idx.get(sp_name, 0))
            did = _to_int_domain(entry.get("domainID", 0))
            domain_idx.append(self.domain_to_idx.get(did, 0))
        return torch.tensor(species_idx, dtype=torch.long), torch.tensor(domain_idx, dtype=torch.long)

    def predict(self, batch: List[Dict]) -> Dict:
        if self.model is None or self.image_transform is None:
            self.load()

        images = []
        for entry in batch:
            img_val = entry.get("relative_img") or entry.get("file_path") or entry.get("image")
            img = _load_image(img_val)
            if img is None:
                raise ValueError("Missing specimen image in predict input.")
            images.append(self.image_transform(img.convert("RGB")))
        tensor_images = torch.stack(images).unsqueeze(0).to(self.device)

        species_idx, domain_idx = self._encode_metadata(batch)
        species_idx = species_idx.unsqueeze(0).to(self.device)
        domain_idx = domain_idx[:1].to(self.device)
        attention_mask = torch.ones(1, species_idx.shape[1], dtype=torch.bool, device=self.device)

        first = batch[0]
        color_val = first.get("colorpicker_img") or first.get("colorpicker_path")
        scale_val = first.get("scalebar_img") or first.get("scalebar_path")
        color_img = _load_image(color_val)
        scale_img = _load_image(scale_val)
        if color_img is None:
            color_tensor = torch.zeros(1, 3, self.calib_size, self.calib_size, device=self.device)
        else:
            color_tensor = self.calib_transform(color_img.convert("RGB")).unsqueeze(0).to(self.device)
        if scale_img is None:
            scale_tensor = torch.zeros(1, 3, self.calib_size, self.calib_size, device=self.device)
        else:
            scale_tensor = self.calib_transform(scale_img.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(
                tensor_images,
                species_idx,
                attention_mask,
                color_tensor,
                scale_tensor,
                domain_idx,
                domain_dropout_prob=0.0,
            )
            mu = out["mu"]
            sigma = out["sigma"]

        mu = mu * self.target_std.to(self.device) + self.target_mean.to(self.device)
        sigma = sigma * self.target_std.to(self.device)
        mu = mu.squeeze(0)
        sigma = sigma.squeeze(0)

        return {
            "SPEI_30d": {"mu": float(mu[0].item()), "sigma": float(sigma[0].item())},
            "SPEI_1y": {"mu": float(mu[1].item()), "sigma": float(sigma[1].item())},
            "SPEI_2y": {"mu": float(mu[2].item()), "sigma": float(sigma[2].item())},
        }
