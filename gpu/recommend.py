"""Apply the morphology gate and name the exact CSV recommended for submission."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from gpu.common import save_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fallback", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.metrics).read_text())
    result = payload["oof"]
    reference = payload["comparisons"]["reflect-rbf-balanced"]
    gate_checks = {
        "pooled_ba_exceeds_existing_image_model": result["balanced_accuracy"] > reference["balanced_accuracy"],
        "macro_azimuth_ba_improves_by_at_least_3pp": result["macro_azimuth_balanced_accuracy"] >= reference["macro_azimuth_bin_ba"] + 0.03,
        "worst_azimuth_ba_improves_by_at_least_2pp": result["worst_azimuth_balanced_accuracy"] >= reference["worst_azimuth_bin"] + 0.02,
        "both_class_recalls_at_least_60pct": min(result["depth_recall"], result["rise_recall"]) >= 0.60,
        "all_expected_folds_complete": len(result["fold_metrics"]) == int(payload["folds"]),
    }
    promoted = all(gate_checks.values())
    selected = Path(args.candidate if promoted else args.fallback)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recommended = output / "submission.csv"
    shutil.copy2(selected, recommended)
    candidate = pd.read_csv(args.candidate)
    fallback = pd.read_csv(args.fallback)
    if candidate.image_id.tolist() != fallback.image_id.tolist():
        raise ValueError("Candidate/fallback image order differs")
    decision = {
        "status": "PROMOTED" if promoted else "FALLBACK RETAINED",
        "recommended_source": str(selected),
        "recommended_output": str(recommended),
        "recommended_sha256": sha256(recommended),
        "candidate_sha256": sha256(args.candidate),
        "fallback_sha256": sha256(args.fallback),
        "candidate_vs_fallback_label_changes": int((candidate.label != fallback.label).sum()),
        "gate_checks": gate_checks,
        "reason": "Candidate promoted only when pooled OOF, both class recalls, macro within-bin and worst-bin robustness all pass prespecified checks." if promoted else "At least one prespecified morphology gate failed; preserve and submit the existing validated fallback CSV.",
    }
    save_json(output / "final_recommendation.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
