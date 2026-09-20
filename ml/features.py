from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.dataset import Observation
from ml.preprocessing import PreprocessingConfig, preprocess_observation


FEATURE_VERSION = "solar-hog-v1"


@dataclass(frozen=True)
class FeatureConfig:
    image_size: int = 48
    cell_size: int = 8
    orientation_bins: int = 8
    intensity_size: int = 12
    include_metadata: bool = False
    border_mode: str = "constant"


def _resize_area(array: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L")
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _hog(array: np.ndarray, cell_size: int, bins: int) -> np.ndarray:
    gradient_y, gradient_x = np.gradient(array)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)
    edges = np.linspace(0.0, np.pi, bins + 1)
    cells_y = array.shape[0] // cell_size
    cells_x = array.shape[1] // cell_size
    descriptor: list[float] = []
    for cell_y in range(cells_y):
        for cell_x in range(cells_x):
            y0 = cell_y * cell_size
            x0 = cell_x * cell_size
            cell_angles = orientation[y0 : y0 + cell_size, x0 : x0 + cell_size]
            cell_weights = magnitude[y0 : y0 + cell_size, x0 : x0 + cell_size]
            hist, _ = np.histogram(cell_angles, bins=edges, weights=cell_weights)
            hist = hist.astype(np.float32)
            hist /= max(float(np.linalg.norm(hist)), 1e-6)
            descriptor.extend(hist.tolist())
    return np.asarray(descriptor, dtype=np.float32)


def extract_feature_vector(
    source,
    sun_azimuth_angle: float,
    *,
    apply_solar_normalization: bool,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    resolved = config or FeatureConfig()
    array = preprocess_observation(
        source,
        sun_azimuth_angle,
        apply_solar_normalization=apply_solar_normalization,
        config=PreprocessingConfig(output_size=(resolved.image_size, resolved.image_size), border_mode=resolved.border_mode),
    )
    intensity = _resize_area(array, resolved.intensity_size)
    mean = float(array.mean())
    std = max(float(array.std()), 1e-6)
    standardized_intensity = ((intensity - mean) / std).ravel()
    hog = _hog(array, resolved.cell_size, resolved.orientation_bins)
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95]).astype(np.float32)
    gradient_y, gradient_x = np.gradient(array)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    globals_ = np.asarray(
        [
            mean,
            std,
            float(array.min()),
            float(array.max()),
            float((array < 0.03).mean()),
            float(gradient_magnitude.mean()),
            float(gradient_magnitude.std()),
        ],
        dtype=np.float32,
    )
    features = [standardized_intensity.astype(np.float32), hog, quantiles, globals_]
    if resolved.include_metadata:
        radians = np.deg2rad(float(sun_azimuth_angle))
        features.append(
            np.asarray(
                [np.sin(radians), np.cos(radians), np.sin(2 * radians), np.cos(2 * radians)],
                dtype=np.float32,
            )
        )
    return np.concatenate(features)


def extract_feature_matrix(
    observations: list[Observation],
    *,
    apply_solar_normalization: bool,
    config: FeatureConfig | None = None,
    progress_every: int = 500,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for index, observation in enumerate(observations, start=1):
        rows.append(
            extract_feature_vector(
                observation.image_path,
                observation.sun_azimuth_angle,
                apply_solar_normalization=apply_solar_normalization,
                config=config,
            )
        )
        if progress_every and index % progress_every == 0:
            print(f"features {index}/{len(observations)}", flush=True)
    return np.vstack(rows).astype(np.float32)
