"""Build a concise, measured GPU reliability and final recommendation report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pc(value):
    return f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-dir", required=True)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--reference-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    workflow = Path(args.workflow_dir)
    full = Path(args.full_run)
    screening = json.loads((workflow / "screening_results.json").read_text())
    measured = json.loads((full / "full_oof_metrics.json").read_text())
    inference = json.loads((full / "inference_manifest.json").read_text())
    recommendation = json.loads((workflow / "best_verified" / "final_recommendation.json").read_text())
    primary = measured["oof"]
    ci = primary["confidence_interval_95"]
    rows = "\n".join(
        f"| {row['architecture']} | {row['spatial_policy']} | {pc(row['balanced_accuracy'])} | {pc(row['macro_azimuth_balanced_accuracy'])} | {pc(row['worst_azimuth_balanced_accuracy'])} |"
        for row in screening["results"]
    )
    comparisons = "\n".join(
        f"| {name} | {pc(values['balanced_accuracy'])} | {pc(values['macro_azimuth_bin_ba'])} | {pc(values['worst_azimuth_bin'])} |"
        for name, values in measured["comparisons"].items()
    )
    folds = "\n".join(
        f"| {row['fold']} | {pc(row['balanced_accuracy'])} | {pc(row['macro_azimuth_balanced_accuracy'])} | {pc(row['worst_azimuth_balanced_accuracy'])} |"
        for row in primary["fold_metrics"]
    )
    report = f'''# End-to-end GPU morphology training report

## Recommendation

**{recommendation['status']}**. {recommendation['reason']}

Exact recommended CSV: `{recommendation['recommended_output']}`

SHA-256: `{recommendation['recommended_sha256']}`

Candidate versus fallback label changes: {recommendation['candidate_vs_fallback_label_changes']}.

## Full grouped OOF result

Architecture: **{measured['architecture']}**. Spatial policy: **{measured['spatial_policy']}**. All network parameters were fine-tuned from ImageNet initialization using mixed precision on CUDA, class-balanced loss, early stopping and saved best checkpoints. Threshold remained fixed at 0.5.

- Pooled balanced accuracy: {pc(primary['balanced_accuracy'])}
- 95% group-bootstrap interval: {pc(ci[0])}–{pc(ci[1])}
- Depth recall: {pc(primary['depth_recall'])}
- Rise recall: {pc(primary['rise_recall'])}
- Macro within-azimuth-bin balanced accuracy: {pc(primary['macro_azimuth_balanced_accuracy'])}
- Worst azimuth-bin balanced accuracy: {pc(primary['worst_azimuth_balanced_accuracy'])}
- Worst fold balanced accuracy: {pc(primary['worst_fold_balanced_accuracy'])}

| Fold | Pooled BA | Macro azimuth-bin BA | Worst azimuth-bin BA |
| --- | --- | --- | --- |
{folds}

Every OOF row is predicted by a model that did not train on its detected duplicate group. Validation and evaluation use deterministic identical geometry and no augmentation. Training adds only mild photometric jitter/noise after solar normalization.

## Fast screening

One grouped fold was used per architecture/policy screen. The fixed selection rule prioritizes macro 45-degree-bin balanced accuracy, with pooled balanced accuracy only as a tie-breaker.

| Architecture | Boundary policy | Pooled BA | Macro azimuth-bin BA | Worst azimuth-bin BA |
| --- | --- | --- | --- | --- |
{rows}

`reflect` retains the full normalized frame. `crop180` takes a fixed 180×180 central crop that is inside the original image support even at 45-degree rotation, then resizes to 224. `circle180` additionally applies an identical fixed circular mask. These policies prevent rotation-corner geometry from varying with azimuth; the full-frame reflect variant remains an explicit comparison.

## Existing systems

| Existing system | Pooled BA | Macro azimuth-bin BA | Worst azimuth-bin BA |
| --- | --- | --- | --- |
{comparisons}

The metadata and angle-only models remain shortcut diagnostics, not morphology baselines. Promotion is allowed only when the new candidate beats the existing image model on pooled grouped OOF and improves both macro and worst-bin robustness by the prespecified margins, while retaining at least 60% recall for both classes.

Gate checks: `{json.dumps(recommendation['gate_checks'], sort_keys=True)}`.

## Evaluation inference

Submission structure: {"PASS" if inference['submission']['valid'] else "FAIL"}. Mean predicted-label confidence: {pc(inference['mean_confidence'])}. Mean fold probability standard deviation: {inference['mean_probability_std']:.6f}. These are prediction diagnostics, not hidden-test accuracy. Actual hidden-test accuracy remains unknown until organizer evaluation.

Checkpoint files and SHA-256 values are recorded in `inference_manifest.json`. The candidate probabilities and genuine OOF predictions are saved alongside the report. The pre-existing submission remains unchanged as a fallback.

## Limitations

The 10,000-resample interval conditions on fixed OOF predictions and excludes model selection, retraining and evaluation-set shift. Architecture/policy selection used screening validation and therefore creates selection bias. Three-fold training is a time-bounded compromise. A model is not described as robust merely because pooled OOF improves.
'''
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
