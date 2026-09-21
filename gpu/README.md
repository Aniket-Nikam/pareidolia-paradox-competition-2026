# GPU morphology workflow — evaluated but not selected

**Evaluated but not selected.** These checkpoints and predictions do not reproduce the final competition submission. The selected fallback is the root-level classical `reflect-rbf-balanced` pipeline documented in `README.md` and `fallback_manifest.json`.

This package fine-tunes complete ImageNet-initialized networks on competition labels. It is intentionally separate from the historical classical `train.py` and `inference.py` so the validated fallback remains reproducible.

The official solar transform is always applied through `ml.preprocessing.normalize_solar_azimuth`: Pillow rotates counter-clockwise by `-sun_azimuth_angle`. Training, validation and inference share the same deterministic spatial policy. Validation and evaluation are never augmented.

Spatial policies:

- `reflect`: reflect-pad, rotate, center-crop back to 256, resize to 224.
- `crop180`: the same normalization followed by a fixed central 180×180 crop and resize. A 180-pixel square lies within the unpadded support at every rotation angle, including 45 degrees.
- `circle180`: `crop180` plus an identical fixed circular mask after resize.

Training uses only mild brightness/contrast changes and sigma=1/255 noise after normalization. It performs no random spatial transform that could undo the canonical sun direction. The full network is trainable. CUDA mixed precision, ImageNet initialization, class-balanced cross entropy, cosine learning-rate decay, early stopping and checkpointing are enabled.

From the repository root on a CUDA machine:

```bash
python -m gpu.workflow \
  --data-root /path/to/data \
  --groups-csv /path/to/duplicate_groups.csv \
  --reference-metrics reports/reliability_metrics.json \
  --fallback-submission /path/to/existing_submission.csv \
  --output-dir /path/to/gpu_results
```

The time-bounded workflow screens EfficientNet-B0 with all three boundary policies on one grouped fold, screens ResNet-18 using the best policy, and optionally screens ConvNeXt-Tiny with `--include-convnext`. It then trains only the best screened architecture/policy on three complete grouped folds. Selection prioritizes macro within-azimuth-bin balanced accuracy, not pooled accuracy.

Outputs include fold checkpoints, fold histories, genuine OOF predictions, 10,000 group-bootstrap samples, evaluation probabilities, `candidate_submission.csv`, `updated_reliability_report.md`, and `best_verified/submission.csv`. The latter is copied from the candidate only if every prespecified morphology gate passes; otherwise it is the unchanged fallback.

The Colab notebook is `notebooks/pareidolia_gpu_finetuning.ipynb`. Colab runtimes are not guaranteed. Confirm a GPU is assigned before training. The notebook downloads the public source repository and organizer data, or allows manual uploads when Drive download is unavailable. It does not publish datasets or weights.

## Verified local result

The completed RTX 3050 run selected EfficientNet-B0 with `circle180`, then trained three grouped folds. Pooled OOF balanced accuracy was 68.00%, macro within-azimuth-bin balanced accuracy was 50.85%, and the worst bin was 48.49%. The model failed the promotion gate, so the validated fallback CSV remains the exact recommendation. See `reports/gpu_reliability_report.md` for screening results, comparisons, limitations, and the decision.
