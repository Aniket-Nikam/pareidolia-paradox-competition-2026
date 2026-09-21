"""Time-bounded screening, full grouped training, inference and recommendation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gpu.common import save_json


def execute(arguments):
    print("RUN", " ".join(map(str, arguments)), flush=True)
    subprocess.run([sys.executable, *map(str, arguments)], check=True)


def result_key(path):
    metrics = json.loads(Path(path).read_text())["validation"]
    return metrics["macro_azimuth_balanced_accuracy"], metrics["balanced_accuracy"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--groups-csv", required=True)
    parser.add_argument("--reference-metrics", required=True)
    parser.add_argument("--fallback-submission", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--screen-epochs", type=int, default=6)
    parser.add_argument("--full-epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--include-convnext", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    cache = output / "preprocessed_cache"
    screen = output / "screen"
    full_root = output / "full"
    screen.mkdir(parents=True, exist_ok=True)
    common = [
        "-m", "gpu.train", "--data-root", args.data_root, "--groups-csv", args.groups_csv,
        "--folds", "3", "--run-folds", "0", "--epochs", str(args.screen_epochs),
        "--patience", str(min(args.patience, 3)), "--batch-size", str(args.batch_size),
        "--workers", str(args.workers), "--cache-dir", str(cache), "--resume",
    ]
    efficient_runs = []
    for policy in ["reflect", "crop180", "circle180"]:
        target = screen / f"efficientnet_b0_{policy}"
        execute([*common, "--architecture", "efficientnet_b0", "--spatial-policy", policy, "--output-dir", target])
        efficient_runs.append(target)
    best_policy_run = max(efficient_runs, key=lambda path: result_key(path / "fold_0_metrics.json"))
    best_policy = json.loads((best_policy_run / "run_config.json").read_text())["spatial_policy"]
    architecture_runs = [best_policy_run]
    for architecture in ["resnet18"] + (["convnext_tiny"] if args.include_convnext else []):
        target = screen / f"{architecture}_{best_policy}"
        batch = min(args.batch_size, 32) if architecture == "convnext_tiny" else args.batch_size
        architecture_common = list(common)
        architecture_common[architecture_common.index("--batch-size") + 1] = str(batch)
        execute([*architecture_common, "--architecture", architecture, "--spatial-policy", best_policy, "--output-dir", target])
        architecture_runs.append(target)
    selected_screen = max(architecture_runs, key=lambda path: result_key(path / "fold_0_metrics.json"))
    selected_config = json.loads((selected_screen / "run_config.json").read_text())
    architecture = selected_config["architecture"]
    policy = selected_config["spatial_policy"]
    screening = []
    for path in efficient_runs + [p for p in architecture_runs if p not in efficient_runs]:
        fold = json.loads((path / "fold_0_metrics.json").read_text())
        screening.append({
            "architecture": fold["architecture"],
            "spatial_policy": fold["spatial_policy"]["name"],
            "fold": 0,
            "balanced_accuracy": fold["validation"]["balanced_accuracy"],
            "macro_azimuth_balanced_accuracy": fold["validation"]["macro_azimuth_balanced_accuracy"],
            "worst_azimuth_balanced_accuracy": fold["validation"]["worst_azimuth_balanced_accuracy"],
            "checkpoint": str(path / "fold_0.pt"),
        })
    save_json(output / "screening_results.json", {
        "selection_rule": "Macro 45-degree-bin balanced accuracy first; pooled balanced accuracy breaks ties.",
        "results": screening,
        "selected_architecture": architecture,
        "selected_spatial_policy": policy,
    })
    full = full_root / f"{architecture}_{policy}"
    full_batch = min(args.batch_size, 32) if architecture == "convnext_tiny" else args.batch_size
    execute([
        "-m", "gpu.train", "--data-root", args.data_root, "--groups-csv", args.groups_csv,
        "--output-dir", full, "--architecture", architecture, "--spatial-policy", policy,
        "--folds", "3", "--run-folds", "all", "--epochs", str(args.full_epochs),
        "--patience", str(args.patience), "--batch-size", str(full_batch), "--workers", str(args.workers),
        "--cache-dir", str(cache), "--resume",
    ])
    execute([
        "-m", "gpu.aggregate", "--data-root", args.data_root, "--groups-csv", args.groups_csv,
        "--run-dir", full, "--reference-metrics", args.reference_metrics,
    ])
    candidate = output / "candidate_submission.csv"
    execute(["-m", "gpu.inference", "--data-root", args.data_root, "--run-dir", full, "--output", candidate, "--workers", str(args.workers), "--cache-dir", str(cache)])
    recommendation = output / "best_verified"
    execute([
        "-m", "gpu.recommend", "--metrics", full / "full_oof_metrics.json", "--candidate", candidate,
        "--fallback", args.fallback_submission, "--output-dir", recommendation,
    ])
    execute([
        "-m", "gpu.build_report", "--workflow-dir", output, "--full-run", full,
        "--reference-metrics", args.reference_metrics, "--output", output / "updated_reliability_report.md",
    ])
    print(json.dumps({"status": "complete", "selected_architecture": architecture, "selected_spatial_policy": policy, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
