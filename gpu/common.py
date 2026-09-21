"""Shared preprocessing, data, models and metrics for GPU training/inference."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet18,
)
from torchvision.transforms import functional as TF

from dataset import locate
from ml.preprocessing import normalize_solar_azimuth


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ARCHITECTURES = ("efficientnet_b0", "resnet18", "convnext_tiny")
SPATIAL_POLICIES = ("reflect", "crop180", "circle180")


@dataclass(frozen=True)
class ImagePolicy:
    """Angle normalization followed by angle-independent boundary geometry."""

    name: str = "crop180"
    output_size: int = 224
    crop_size: int = 180
    circle_radius: int = 86

    def __post_init__(self):
        if self.name not in SPATIAL_POLICIES:
            raise ValueError(f"Unknown spatial policy {self.name!r}")
        if self.crop_size > 180:
            raise ValueError("crop_size must be <=180 to exclude reflected corners at every angle")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def preprocess_pil(source: str | Path, angle: float, policy: ImagePolicy) -> Image.Image:
    """Apply the official -azimuth rotation and a fixed geometry policy.

    The 180px crop fits inside the original 256px square at the worst 45-degree
    rotation, so it contains no reflect-padding pixels for any azimuth.  The
    optional circular mask is fixed after resizing and therefore cannot encode
    the supplied angle through its boundary.
    """
    with Image.open(source) as opened:
        image = opened.convert("L")
    image = normalize_solar_azimuth(image, angle, border_mode="reflect")
    if policy.name in {"crop180", "circle180"}:
        left = (image.width - policy.crop_size) // 2
        top = (image.height - policy.crop_size) // 2
        image = image.crop((left, top, left + policy.crop_size, top + policy.crop_size))
    image = image.resize((policy.output_size, policy.output_size), Image.Resampling.BICUBIC)
    if policy.name == "circle180":
        values = np.asarray(image).copy()
        yy, xx = np.ogrid[: policy.output_size, : policy.output_size]
        centre = (policy.output_size - 1) / 2
        mask = (xx - centre) ** 2 + (yy - centre) ** 2 <= policy.circle_radius**2
        values[~mask] = 0
        image = Image.fromarray(values, mode="L")
    return image


def tensor_from_pil(image: Image.Image, train: bool, rng: random.Random) -> torch.Tensor:
    """Photometric-only training augmentation; validation/inference are deterministic."""
    if train:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.92, 1.08))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.92, 1.08))
    tensor = TF.pil_to_tensor(image).float().div_(255.0).repeat(3, 1, 1)
    if train and rng.random() < 0.25:
        tensor = (tensor + torch.randn_like(tensor) * (1.0 / 255.0)).clamp_(0, 1)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class LunarDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, paths: list[Path], indices, policy: ImagePolicy, train: bool, seed: int, cache_dir=None):
        self.frame = frame.reset_index(drop=True)
        self.paths = paths
        self.indices = np.asarray(indices, dtype=int)
        self.policy = policy
        self.train = train
        self.seed = seed
        self.cache_dir = Path(cache_dir) / policy.name if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        row = self.frame.iloc[index]
        cached = self.cache_dir / str(row.image_id) if self.cache_dir else None
        if cached and cached.exists():
            with Image.open(cached) as opened:
                image = opened.convert("L").copy()
        else:
            image = preprocess_pil(self.paths[index], float(row.sun_azimuth_angle), self.policy)
            if cached:
                temporary = cached.with_suffix(cached.suffix + f".{os.getpid()}.tmp")
                image.save(temporary, format="PNG")
                os.replace(temporary, cached)
        # Python's worker RNG is seeded by worker_seed; this changes photometric
        # augmentation across epochs while remaining reproducible for a run.
        rng = random.Random(self.seed + index * 1_000_003 + random.randrange(2**31))
        tensor = tensor_from_pil(image, self.train, rng)
        label = int(row.label) if "label" in self.frame else -1
        return tensor, label, index


def load_groups(frame: pd.DataFrame, groups_csv: str | Path) -> np.ndarray:
    mapping = pd.read_csv(groups_csv)
    required = {"image_id", "group"}
    if not required.issubset(mapping.columns) or mapping.image_id.duplicated().any():
        raise ValueError("groups CSV must contain unique image_id and group columns")
    groups = mapping.set_index("image_id").reindex(frame.image_id).group
    if groups.isna().any():
        raise ValueError("groups CSV does not cover every training image")
    return groups.to_numpy()


def grouped_folds(frame: pd.DataFrame, groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    y = frame.label.to_numpy()
    assignment = np.full(len(frame), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        assignment[validation] = fold
    if (assignment < 0).any():
        raise AssertionError("Incomplete fold assignment")
    if pd.DataFrame({"group": groups, "fold": assignment}).groupby("group").fold.nunique().max() != 1:
        raise AssertionError("Detected group leakage across folds")
    return assignment


def build_model(name: str, pretrained: bool = True) -> torch.nn.Module:
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    elif name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    else:
        raise ValueError(f"Unknown architecture {name!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def classification_metrics(y, probability, angles) -> dict:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    bins = []
    for lower in range(0, 360, 45):
        mask = (angles >= lower) & (angles < lower + 45)
        if not mask.any() or len(np.unique(y[mask])) < 2:
            continue
        bins.append({
            "lower": lower,
            "n": int(mask.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(y[mask], prediction[mask])),
        })
    depth_recall = float(cm[0, 0] / cm[0].sum())
    rise_recall = float(cm[1, 1] / cm[1].sum())
    return {
        "n": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float((prediction == y).mean()),
        "depth_recall": depth_recall,
        "rise_recall": rise_recall,
        "recall_gap": abs(depth_recall - rise_recall),
        "confusion_matrix": cm.tolist(),
        "azimuth_bins": bins,
        "macro_azimuth_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in bins])),
        "worst_azimuth_balanced_accuracy": float(min(row["balanced_accuracy"] for row in bins)),
    }


def checkpoint_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def policy_dict(policy: ImagePolicy) -> dict:
    return asdict(policy)


def locate_data(data_root: str | Path):
    return locate(data_root)
