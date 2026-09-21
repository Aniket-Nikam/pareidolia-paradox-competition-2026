"""Combine complete grouped folds into genuine OOF morphology metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpu.common import classification_metrics, load_groups, locate_data, save_json
from reliability_common import group_bootstrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--groups-csv", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-metrics", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    config = json.loads((run / "run_config.json").read_text())
    expected_folds = list(range(int(config["folds"])))
    files = [run / f"fold_{fold}_validation.csv" for fold in expected_folds]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise ValueError(f"Full OOF is incomplete; missing {missing}")
    oof = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame = locate_data(args.data_root)["train"][0]
    if len(oof) != len(frame) or oof.image_id.duplicated().any() or set(oof.image_id) != set(frame.image_id):
        raise ValueError("OOF predictions do not cover each training image exactly once")
    oof = frame[["image_id", "label", "sun_azimuth_angle"]].merge(
        oof[["image_id", "fold", "probability_rise"]], on="image_id", how="left", validate="one_to_one"
    )
    groups = load_groups(frame, args.groups_csv)
    group_by_id = dict(zip(frame.image_id, groups))
    oof["group"] = oof.image_id.map(group_by_id)
    if oof.groupby("group").fold.nunique().max() != 1:
        raise AssertionError("Group leakage in saved OOF predictions")
    prediction = (oof.probability_rise >= 0.5).astype(int)
    oof["prediction"] = prediction
    metrics = classification_metrics(oof.label, oof.probability_rise, oof.sun_azimuth_angle.to_numpy())
    bootstrap = group_bootstrap(oof.label.to_numpy(), oof.group.to_numpy(), [prediction.to_numpy()], resamples=10000, seed=2026)[:, 0]
    metrics["confidence_interval_95"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    fold_metrics = []
    for fold in expected_folds:
        mask = oof.fold == fold
        fold_metrics.append({"fold": fold, **classification_metrics(oof.label[mask], oof.probability_rise[mask], oof.sun_azimuth_angle.to_numpy()[mask])})
    metrics["fold_metrics"] = fold_metrics
    metrics["worst_fold_balanced_accuracy"] = min(row["balanced_accuracy"] for row in fold_metrics)
    reference = json.loads(Path(args.reference_metrics).read_text())
    candidates = reference.get("candidates", reference)
    comparisons = {}
    for name in ["existing-metadata-logistic", "reflect-rbf-balanced", "angle-only-diagnostic"]:
        if name in candidates:
            comparisons[name] = {
                key: candidates[name].get(key)
                for key in ["balanced_accuracy", "recall_depth", "recall_rise", "macro_azimuth_bin_ba", "worst_azimuth_bin"]
            }
    payload = {
        "architecture": config["architecture"],
        "spatial_policy": config["spatial_policy"],
        "folds": config["folds"],
        "full_network_finetuned": True,
        "threshold": 0.5,
        "oof": metrics,
        "comparisons": comparisons,
        "bootstrap": {
            "resamples": 10000,
            "seed": 2026,
            "method": "Composition-stratified duplicate-group cluster bootstrap on fixed OOF predictions.",
            "limitations": "Conditional interval; excludes retraining, model selection and test distribution shift.",
        },
    }
    oof.to_csv(run / "oof_predictions.csv", index=False)
    np.savez_compressed(run / "bootstrap.npz", balanced_accuracy=bootstrap)
    save_json(run / "full_oof_metrics.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
