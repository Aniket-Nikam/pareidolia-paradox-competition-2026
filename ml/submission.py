from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = ["image_id", "label"]


def validate_submission(frame: pd.DataFrame, *, expected_rows: int = 2000, expected_ids: list[str] | None = None) -> None:
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Columns must be exactly {EXPECTED_COLUMNS}")
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, received {len(frame)}")
    if frame["image_id"].isna().any():
        raise ValueError("Submission contains missing image IDs")
    if frame["image_id"].duplicated().any():
        raise ValueError("Submission contains duplicate image IDs")
    if not pd.api.types.is_integer_dtype(frame["label"]) or not frame["label"].isin([0, 1]).all():
        raise ValueError("Submission labels must contain only integer 0 or 1")
    if expected_ids is not None and frame.image_id.tolist() != list(expected_ids):
        raise ValueError("Submission IDs and row order must match evaluation metadata")


def write_submission(
    image_ids: list[str],
    labels,
    output_path: str | Path,
    *,
    expected_rows: int = 2000,
) -> pd.DataFrame:
    frame = pd.DataFrame({"image_id": image_ids, "label": labels})
    validate_submission(frame, expected_rows=expected_rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    validate_submission(reloaded, expected_rows=expected_rows)
    return frame
