from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from ml.preprocessing import PreprocessingConfig, preprocess_observation


REQUIRED_COMMON_COLUMNS = {"image_id", "sun_azimuth_angle"}


@dataclass(frozen=True)
class Observation:
    image_id: str
    sun_azimuth_angle: float
    label: int | None
    image_path: Path


def read_metadata(csv_path: str | Path, *, require_labels: bool) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = set(REQUIRED_COMMON_COLUMNS)
    if require_labels:
        required.add("label")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if frame["image_id"].isna().any() or frame["image_id"].duplicated().any():
        raise ValueError("image_id values must be present and unique")
    angles = pd.to_numeric(frame["sun_azimuth_angle"], errors="raise")
    if not np.isfinite(angles).all() or not angles.between(0, 360).all():
        raise ValueError("sun_azimuth_angle must be finite and between 0 and 360")
    if any(Path(str(value)).name != str(value) or "\\" in str(value) for value in frame.image_id):
        raise ValueError("image_id must be a filename without a path")
    if require_labels:
        if not frame["label"].isin([0, 1]).all():
            raise ValueError("Labels must be binary integer values")
    return frame


def build_observations(
    csv_path: str | Path,
    image_dir: str | Path,
    *,
    require_labels: bool,
    validate_files: bool = True,
) -> list[Observation]:
    frame = read_metadata(csv_path, require_labels=require_labels)
    root = Path(image_dir)
    observations: list[Observation] = []
    missing_files: list[str] = []
    for row in frame.itertuples(index=False):
        path = root / str(row.image_id)
        if validate_files and not path.is_file():
            missing_files.append(str(row.image_id))
        observations.append(
            Observation(
                image_id=str(row.image_id),
                sun_azimuth_angle=float(row.sun_azimuth_angle),
                label=int(row.label) if require_labels else None,
                image_path=path,
            )
        )
    if missing_files:
        sample = ", ".join(missing_files[:5])
        raise FileNotFoundError(f"Missing {len(missing_files)} images. First: {sample}")
    return observations


class LunarDataset:
    """Framework-neutral deterministic dataset for baselines and API use."""

    def __init__(
        self,
        observations: list[Observation],
        *,
        apply_solar_normalization: bool = True,
        config: PreprocessingConfig | None = None,
    ) -> None:
        self.observations = observations
        self.apply_solar_normalization = apply_solar_normalization
        self.config = config or PreprocessingConfig()

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int):
        observation = self.observations[index]
        image = preprocess_observation(
            observation.image_path,
            observation.sun_azimuth_angle,
            apply_solar_normalization=self.apply_solar_normalization,
            config=self.config,
        )
        return image, observation.label, observation
