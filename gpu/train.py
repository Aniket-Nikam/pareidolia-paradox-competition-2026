"""Fine-tune one complete ImageNet model on grouped competition folds."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gpu.common import (
    ARCHITECTURES,
    SPATIAL_POLICIES,
    ImagePolicy,
    LunarDataset,
    build_model,
    checkpoint_digest,
    classification_metrics,
    grouped_folds,
    load_groups,
    locate_data,
    policy_dict,
    save_json,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--groups-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="efficientnet_b0")
    parser.add_argument("--spatial-policy", choices=SPATIAL_POLICIES, default="crop180")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--run-folds", default="all", help="Comma-separated zero-based folds, or all")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-cpu", action="store_true", help="Only for smoke tests; official runs require CUDA")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def worker_seed(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    import random
    random.seed(seed)


@torch.inference_mode()
def predict(model, loader, device, use_amp):
    model.eval()
    probabilities, labels, indices = [], [], []
    for images, target, index in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        probabilities.append(logits.softmax(1)[:, 1].float().cpu().numpy())
        labels.append(target.numpy())
        indices.append(index.numpy())
    return np.concatenate(probabilities), np.concatenate(labels), np.concatenate(indices)


def fit_fold(args, frame, paths, groups, fold_assignment, fold, output, device):
    seed_everything(args.seed + fold)
    train_indices = np.flatnonzero(fold_assignment != fold)
    valid_indices = np.flatnonzero(fold_assignment == fold)
    if set(groups[train_indices]) & set(groups[valid_indices]):
        raise AssertionError("Group leakage")
    policy = ImagePolicy(args.spatial_policy)
    train_set = LunarDataset(frame, paths, train_indices, policy, True, args.seed + fold, args.cache_dir)
    valid_set = LunarDataset(frame, paths, valid_indices, policy, False, args.seed + fold, args.cache_dir)
    generator = torch.Generator().manual_seed(args.seed + fold)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        generator=generator,
    )
    if args.workers:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, drop_last=False, **loader_kwargs)
    model = build_model(args.architecture, pretrained=True).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable != total:
        raise AssertionError("The complete network must be trainable")
    counts = np.bincount(frame.label.to_numpy()[train_indices], minlength=2)
    weights = counts.sum() / (2 * counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate / 50)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint = output / f"fold_{fold}.pt"
    history, best_key, patience_left = [], (-np.inf, -np.inf), args.patience
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, target, _ in train_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        probability, labels, indices = predict(model, valid_loader, device, use_amp)
        order = np.argsort(indices)
        probability, labels, indices = probability[order], labels[order], indices[order]
        metrics = classification_metrics(labels, probability, frame.sun_azimuth_angle.to_numpy()[indices])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **metrics,
        }
        history.append(row)
        key = (metrics["macro_azimuth_balanced_accuracy"], metrics["balanced_accuracy"])
        print(json.dumps({"fold": fold, **row}), flush=True)
        if key > best_key:
            best_key = key
            patience_left = args.patience
            torch.save(
                {
                    "architecture": args.architecture,
                    "state_dict": model.state_dict(),
                    "policy": policy_dict(policy),
                    "fold": fold,
                    "folds": args.folds,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation": metrics,
                    "trainable_parameters": trainable,
                    "class_weights": weights.tolist(),
                },
                checkpoint,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    elapsed = time.perf_counter() - start
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    probability, labels, indices = predict(model, valid_loader, device, use_amp)
    order = np.argsort(indices)
    probability, labels, indices = probability[order], labels[order], indices[order]
    final_metrics = classification_metrics(labels, probability, frame.sun_azimuth_angle.to_numpy()[indices])
    pd.DataFrame(
        {
            "image_id": frame.image_id.to_numpy()[indices],
            "label": labels,
            "fold": fold,
            "probability_rise": probability,
            "prediction": (probability >= 0.5).astype(int),
        }
    ).to_csv(output / f"fold_{fold}_validation.csv", index=False)
    result = {
        "fold": fold,
        "architecture": args.architecture,
        "spatial_policy": policy_dict(policy),
        "validation": final_metrics,
        "best_epoch": int(saved["epoch"]),
        "epochs_run": len(history),
        "seconds": elapsed,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "trainable_parameters": trainable,
        "full_network_finetuned": True,
        "mixed_precision": use_amp,
        "class_weights": weights.tolist(),
        "history": history,
    }
    save_json(output / f"fold_{fold}_metrics.json", result)
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for a smoke test.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    found = locate_data(args.data_root)
    frame, paths, _ = found["train"]
    groups = load_groups(frame, args.groups_csv)
    assignment = grouped_folds(frame, groups, args.folds, args.seed)
    requested = list(range(args.folds)) if args.run_folds == "all" else [int(value) for value in args.run_folds.split(",")]
    if not requested or min(requested) < 0 or max(requested) >= args.folds:
        raise ValueError("run-folds contains an invalid fold")
    run = {
        "architecture": args.architecture,
        "spatial_policy": args.spatial_policy,
        "folds": args.folds,
        "requested_folds": requested,
        "seed": args.seed,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "full_network_finetuned": True,
        "selection_rule": "Checkpoint maximizes validation macro 45-degree-bin balanced accuracy; pooled BA breaks ties. Threshold is fixed at 0.5.",
        "validation_augmentation": False,
        "training_augmentation": "Photometric brightness/contrast and sigma=1/255 noise only; no spatial transform after solar normalization.",
    }
    config_path = output / "run_config.json"
    if args.resume and config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ["architecture", "spatial_policy", "folds", "seed", "epochs", "patience", "batch_size", "learning_rate", "weight_decay"]:
            if previous.get(key) != run.get(key):
                raise ValueError(f"Cannot resume: {key} changed")
    save_json(config_path, run)
    results = []
    for fold in requested:
        metric_path = output / f"fold_{fold}_metrics.json"
        checkpoint_path = output / f"fold_{fold}.pt"
        prediction_path = output / f"fold_{fold}_validation.csv"
        if args.resume and metric_path.exists() and checkpoint_path.exists() and prediction_path.exists():
            result = json.loads(metric_path.read_text())
            if checkpoint_digest(checkpoint_path) != result["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash changed for fold {fold}")
            print(json.dumps({"fold": fold, "status": "reused"}), flush=True)
        else:
            result = fit_fold(args, frame, paths, groups, assignment, fold, output, device)
        results.append(result)
    save_json(output / "completed_metrics.json", {**run, "results": results})
    print(json.dumps({"status": "complete", "output": str(output), "folds": requested}), flush=True)


if __name__ == "__main__":
    main()
