import math
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import r2_score


def get_training_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--domain_id_aug_prob", type=float, default=0.1)
    parser.add_argument("--checkpoint_name", type=str, default="model_v2.pth")
    parser.add_argument("--freeze_backbone", type=int, default=3)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--max_instances", type=int, default=12)
    parser.add_argument("--sample_strategy", type=str, choices=["random"], default="random")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--calib_size", type=int, default=128)
    parser.add_argument("--model_version", type=str, choices=["v1", "v2"], default="v2")
    parser.add_argument("--sanity_check", action="store_true")
    return parser.parse_args()


def build_transforms(image_size: int = 224, calib_size: int = 128):
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


def gaussian_crps(y_true, mu, sigma):
    sigma = np.maximum(sigma, 1e-3)
    z = (y_true - mu) / sigma
    sqrt_2 = np.sqrt(2.0)
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    erf = np.vectorize(math.erf)
    Phi = 0.5 * (1.0 + erf(z / sqrt_2))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
    return crps


def evaluate_spei_r2_scores(gts, preds, eps: float = 1e-6):
    scores = []
    for i in range(3):
        if np.var(gts[:, i]) < eps:
            scores.append(None)
        else:
            scores.append(r2_score(gts[:, i], preds[:, i]))
    return tuple(scores)


def _load_image(value: Any, base_dir: Optional[str] = None) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        path = value
        if not os.path.exists(path) and base_dir:
            path = os.path.join(base_dir, value)
        if not os.path.exists(path):
            return None
        return Image.open(path)
    return None


def _get_first_present(row: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
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


class EventDataset(Dataset):
    def __init__(
        self,
        dataset,
        species_to_idx: Dict[str, int],
        domain_to_idx: Dict[int, int],
        image_transform,
        calib_transform,
        max_instances: int = 12,
        sample_strategy: str = "random",
    ):
        self.dataset = dataset
        self.species_to_idx = species_to_idx
        self.domain_to_idx = domain_to_idx
        self.image_transform = image_transform
        self.calib_transform = calib_transform
        self.max_instances = max_instances
        self.sample_strategy = sample_strategy

        event_to_indices: Dict[Any, List[int]] = defaultdict(list)
        for idx, event_id in enumerate(dataset["eventID"]):
            event_to_indices[event_id].append(idx)
        self.event_ids = list(event_to_indices.keys())
        self.event_to_indices = event_to_indices

        self._calib_size = 128
        if hasattr(calib_transform, "transforms"):
            for t in calib_transform.transforms:
                if isinstance(t, transforms.CenterCrop):
                    self._calib_size = t.size
                    break
        if isinstance(self._calib_size, (list, tuple)):
            self._calib_size = self._calib_size[0]

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        event_id = self.event_ids[idx]
        indices = self.event_to_indices[event_id]
        if self.max_instances and len(indices) > self.max_instances:
            if self.sample_strategy == "random":
                indices = np.random.choice(indices, size=self.max_instances, replace=False).tolist()
            else:
                indices = indices[: self.max_instances]
        rows = [self.dataset[i] for i in indices]

        images = []
        species_ids = []
        base_dir = None
        for row in rows:
            img_val = _get_first_present(row, ["file_path", "relative_img", "image"])
            if base_dir is None and isinstance(img_val, str):
                base_dir = os.path.dirname(img_val)
            img = _load_image(img_val, base_dir=base_dir)
            if img is None:
                raise ValueError("Missing specimen image for event.")
            images.append(self.image_transform(img.convert("RGB")))
            species_ids.append(self.species_to_idx.get(row.get("scientificName", "Unknown"), 0))

        first = rows[0]
        domain_id = self.domain_to_idx.get(_to_int_domain(first.get("domainID", 0)), 0)

        target = torch.tensor(
            [
                float(first["SPEI_30d"]),
                float(first["SPEI_1y"]),
                float(first["SPEI_2y"]),
            ],
            dtype=torch.float32,
        )

        color_val = _get_first_present(first, ["colorpicker_img", "colorpicker_path"])
        scale_val = _get_first_present(first, ["scalebar_img", "scalebar_path"])
        color_img = _load_image(color_val, base_dir=base_dir)
        scale_img = _load_image(scale_val, base_dir=base_dir)
        if color_img is None:
            color_tensor = torch.zeros(3, self._calib_size, self._calib_size)
        else:
            color_tensor = self.calib_transform(color_img.convert("RGB"))
        if scale_img is None:
            scale_tensor = torch.zeros(3, self._calib_size, self._calib_size)
        else:
            scale_tensor = self.calib_transform(scale_img.convert("RGB"))

        return {
            "images": torch.stack(images),
            "color": color_tensor,
            "scale": scale_tensor,
            "species_ids": torch.tensor(species_ids, dtype=torch.long),
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "target": target,
        }


def collate_events(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_n = max(item["images"].shape[0] for item in batch)
    batch_size = len(batch)
    img_shape = batch[0]["images"].shape[1:]

    images = torch.zeros(batch_size, max_n, *img_shape, dtype=torch.float32)
    species_ids = torch.zeros(batch_size, max_n, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    colors = torch.stack([item["color"] for item in batch])
    scales = torch.stack([item["scale"] for item in batch])
    domain_ids = torch.stack([item["domain_id"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])

    for i, item in enumerate(batch):
        n = item["images"].shape[0]
        images[i, :n] = item["images"]
        species_ids[i, :n] = item["species_ids"]
        attention_mask[i, :n] = True

    return {
        "images": images,
        "species_ids": species_ids,
        "attention_mask": attention_mask,
        "color": colors,
        "scale": scales,
        "domain_id": domain_ids,
        "target": targets,
    }
