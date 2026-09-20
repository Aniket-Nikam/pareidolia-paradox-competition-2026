from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps


ImageSource = str | Path | BinaryIO | Image.Image


@dataclass(frozen=True)
class PreprocessingConfig:
    output_size: tuple[int, int] = (64, 64)
    fill_value: int = 0
    resample: Image.Resampling = Image.Resampling.BICUBIC
    border_mode: str = "constant"


def load_grayscale(source: ImageSource) -> Image.Image:
    """Load an observation as a detached single-channel image."""
    if isinstance(source, Image.Image):
        return ImageOps.grayscale(source).copy()
    with Image.open(source) as image:
        return ImageOps.grayscale(image).copy()


def normalize_solar_azimuth(
    image: Image.Image,
    sun_azimuth_angle: float,
    *,
    fill_value: int = 0,
    resample: Image.Resampling = Image.Resampling.BICUBIC,
    border_mode: str = "constant",
) -> Image.Image:
    """Rotate by the challenge-defined negative solar azimuth.

    Pillow uses positive angles for counter-clockwise rotation, so passing the
    negative metadata angle implements the specified `-sun_azimuth_angle`
    transform directly. Canvas dimensions stay fixed at 256 x 256.
    """
    angle = float(sun_azimuth_angle)
    if not np.isfinite(angle):
        raise ValueError("sun_azimuth_angle must be finite")
    if border_mode not in {"constant", "reflect"}:
        raise ValueError("border_mode must be constant or reflect")
    if border_mode == "reflect":
        width, height = image.size
        # Symmetric integer padding keeps the pixel-centre rotation unchanged.
        pad = int(np.ceil(np.hypot(width, height) / 2)) + 4
        values = np.asarray(image)
        pads = ((pad, pad), (pad, pad)) + (((0, 0),) if values.ndim == 3 else ())
        padded = Image.fromarray(np.pad(values, pads, mode="reflect"))
        rotated = padded.rotate(-angle, resample=resample, expand=False)
        return rotated.crop((pad, pad, pad + width, pad + height))
    return image.rotate(
        -angle,
        resample=resample,
        expand=False,
        fillcolor=int(fill_value),
    )


def preprocess_observation(
    source: ImageSource,
    sun_azimuth_angle: float,
    *,
    apply_solar_normalization: bool = True,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Central preprocessing used by training, validation, and inference."""
    resolved = config or PreprocessingConfig()
    image = load_grayscale(source)
    if apply_solar_normalization:
        image = normalize_solar_azimuth(
            image,
            sun_azimuth_angle,
            fill_value=resolved.fill_value,
            resample=resolved.resample,
            border_mode=resolved.border_mode,
        )
    image = image.resize(resolved.output_size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def to_model_tensor(array: np.ndarray) -> np.ndarray:
    """Standardize one grayscale observation without dataset leakage."""
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale array, received {array.shape}")
    mean = float(array.mean())
    std = max(float(array.std()), 1e-6)
    return ((array - mean) / std)[None, ...].astype(np.float32)
