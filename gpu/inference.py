"""Average fine-tuned fold checkpoints and create a candidate submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gpu.common import ImagePolicy, LunarDataset, build_model, checkpoint_digest, locate_data, save_json
from gpu.train import predict
from validate_submission import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official candidate inference")
    device = torch.device("cuda")
    run = Path(args.run_dir)
    config = json.loads((run / "run_config.json").read_text())
    found = locate_data(args.data_root)
    frame, paths, metadata = found["test"]
    policy = ImagePolicy(config["spatial_policy"])
    dataset = LunarDataset(frame, paths, np.arange(len(frame)), policy, False, int(config["seed"]), args.cache_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    probabilities, checkpoint_info = [], []
    for fold in range(int(config["folds"])):
        path = run / f"fold_{fold}.pt"
        if not path.exists():
            raise ValueError(f"Missing required checkpoint {path}")
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved["architecture"] != config["architecture"] or saved["policy"]["name"] != config["spatial_policy"]:
            raise ValueError("Checkpoint configuration mismatch")
        model = build_model(config["architecture"], pretrained=False).to(device)
        model.load_state_dict(saved["state_dict"])
        probability, _, indices = predict(model, loader, device, True)
        order = np.argsort(indices)
        if not np.array_equal(indices[order], np.arange(len(frame))):
            raise AssertionError("Inference row order mismatch")
        probabilities.append(probability[order])
        checkpoint_info.append({"file": path.name, "sha256": checkpoint_digest(path), "bytes": path.stat().st_size})
        del model
        torch.cuda.empty_cache()
    stack = np.vstack(probabilities)
    mean_probability = stack.mean(axis=0)
    labels = (mean_probability >= 0.5).astype(int)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": frame.image_id, "label": labels}).to_csv(output, index=False)
    validation = check(output, metadata)
    pd.DataFrame({
        "image_id": frame.image_id,
        "probability_rise": mean_probability,
        "probability_std": stack.std(axis=0),
        "label": labels,
        **{f"fold_{fold}_probability": stack[fold] for fold in range(len(stack))},
    }).to_csv(run / "evaluation_probabilities.csv", index=False)
    save_json(run / "inference_manifest.json", {
        "architecture": config["architecture"],
        "spatial_policy": config["spatial_policy"],
        "checkpoints": checkpoint_info,
        "submission": validation,
        "mean_confidence": float(np.where(labels == 1, mean_probability, 1 - mean_probability).mean()),
        "mean_probability_std": float(stack.std(axis=0).mean()),
    })
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
