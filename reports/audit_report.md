# Audit of the existing solution

## Scope and completion before this audit

The organizer's problem is a binary image-classification task, not a web application. No frontend or API is required. The earlier workspace already contained a trained HOG/logistic-regression model and a structurally valid 2,000-row submission. Its inference was rerun during this audit and reproduced the original CSV byte for byte (335 depth, 1,665 rise). The original files were preserved. Public repository/weights and this stricter audit were not part of that earlier completion.

The official PDF and supplied submission instructions determine the output format. Local discovery found all required material; see `discovery.md`. No dummy training/evaluation data or invented predictions were used.

## Data findings

All 7,854 training and 2,000 evaluation images decoded. Metadata/image ID sets match exactly, and original archive metadata agrees with extracted metadata. All images are 256 x 256, uint8 RGB containers with identical R/G/B channels, so conversion to grayscale loses no channel information. Overall pixel range is 0-255; no constant images were found. Metadata contains finite angles in range and no missing or duplicate IDs. Training labels comprise 2,854 depth and 5,000 rise examples.

Decoded-pixel SHA-256 found 2,385 additional copies beyond the unique images across both splits. A conservative raw/normalized pHash plus thumbnail-RMSE check added two near-duplicate connections. The resulting groups include 1,458 training groups containing both labels and 829 groups shared between training and evaluation. These counts describe duplicate groups, not necessarily erroneous labels: identical pixels may have different azimuth metadata, so interpretation can differ. For example, `train_00001.png` and `train_06657.png` are pixel-identical but have angles 77.69/120.29 and labels 1/0. No labels were changed and no training-label lookup was used for evaluation predictions.

Each raw duplicate group is kept wholly within a validation fold even when metadata differs. This is conservative. Unknown related scenes beyond these exact/near-duplicate checks may remain. Group construction uses evaluation pixels without labels only to conservatively identify shared scene groups; model fitting, calibration, thresholds and model selection use training labels only.

The azimuth distribution differs materially between splits (two-sample KS statistic 0.38418). In training, rise represents approximately 87-89% below 270 degrees but only 35-36% from 270-360 degrees. Evaluation is concentrated from 90-180 degrees. This makes direct angle features a potentially unreliable label shortcut.

## What the original pipeline did correctly

- Matched metadata IDs to image paths and used the supplied binary label convention.
- Applied the shared `normalize_solar_azimuth` implementation at training and inference, passing `-sun_azimuth_angle` to Pillow.
- Used balanced logistic-regression class weights, stratified five-fold validation and balanced accuracy.
- Saved a fitted model, feature settings and threshold together; loading it reproduces the original output.
- Produced exactly `image_id,label` and the correct evaluation ordering.

The saved out-of-fold predictions independently reproduce balanced accuracy 0.7538424667 at threshold 0.555, depth recall 0.804485, rise recall 0.703200 and confusion matrix [[2296,558],[1484,3516]]. This is a historical score, not an unbiased estimate of evaluation performance.

## Weaknesses and repairs

| Finding | Repair or treatment |
| --- | --- |
| Original random stratified folds did not keep identical images together. | StratifiedGroupKFold for outer validation and inner calibration. |
| Threshold was maximized on the same pooled OOF labels used to report the score. | Threshold selection inside each outer training partition only. |
| Black fill creates artificial triangular rotation boundaries. | Reflect-pad, bicubic rotate, center-crop to 256 x 256, then resize to 48 for descriptors. |
| Selected model explicitly ingested sine/cosine azimuth features. | Retain it as a comparison; restrict final selection to image-only candidates and test held-out azimuth sectors. |
| Cached feature arrays lacked data/order provenance. | Fingerprint both metadata and image bytes, and reject stale feature caches. Resume also checks code, configuration and duplicate groups. |
| Integer conversion could silently truncate fractional submission labels. | Require integer binary labels and reject fractional values before writing. |
| Validation lacked an explicit expected-ID-order check. | Final validator compares the entire ordered ID list to test metadata. |
| Metadata did not reject all non-finite/out-of-range angles. | Add strict angle checks, path-safe IDs and binary-label checks. |
| Frontend could obscure the actual competition deliverable. | Isolate a portable ML-only repository and standalone upload CSV. |

The root preprocessing default stays constant-fill so the old checkpoint's predictions remain reproducible. New training/inference explicitly use reflect mode. Representative original/constant/reflect images were visually inspected and a negative-90-degree marker test verifies the sign. Reflect padding avoids newly introduced fill triangles, but preserves black regions already present in the original image and introduces mirrored edge content. It cannot guarantee removal of all angle-related cues.

## Validation design and modeling limits

Five grouped outer folds use seed 2026. Within each outer training partition, one grouped inner holdout fits a sigmoid calibrator and chooses a balanced-accuracy threshold. The base classifier is then refitted on the outer training partition. Calibration and threshold share an inner holdout, but never use outer validation labels. Refitting can shift calibration, which is evaluated by the outer fold. Reported probabilities are model estimates, not guarantees of calibrated confidence.

All candidates additionally undergo leave-one-90-degree-sector-out validation. Groups appearing in a held-out sector are removed from training entirely. Each sector has its own training-only calibration/threshold. This deliberately difficult distribution-shift test is not the official evaluation split.

Comparisons include the existing metadata logistic design, reflect-image logistic, balanced and unweighted reflect-image RBF SVMs, and an angle-only diagnostic. Final selection is restricted to image-only models, using the mean of grouped OOF balanced accuracy and mean sector-held-out balanced accuracy. Selection reuses these validation results; there is no untouched labeled test set.

No CUDA GPU was available (12 logical CPU cores, approximately 16.9 GB RAM). RBF SVMs offer a feasible nonlinear comparison with the linear baseline. No pretrained CNN, mixed precision, learning-rate schedule or early stopping is claimed. Those neural-network controls are not applicable to these converged sklearn estimators. No augmentation or test-time augmentation is used; in particular, normalized validation/evaluation images are never randomly flipped or rotated.

The final checkpoint bundles five fitted outer-fold members. Every training row participates in four member training partitions. OOF scores evaluate the one member that did not see each row; they are not an unbiased measured score of the five-member ensemble. Evaluation labels are unavailable, so leaderboard balanced accuracy is unknown.

## Execution checks

An isolated Python 3.13 environment was created and pinned dependencies installed. The full new training/inference commands use this environment. Root regression tests passed: 18/18; the portable repository's contract tests passed: 5/5. Initial sandbox runs encountered temporary-directory permission errors; rerunning outside the sandbox passed. Original inference also completed successfully and was byte-identical to the previous submission. A full five-fold rerun of original validation from fresh features reproduced the 75.3842% score with maximum probability difference 6.67e-16. The delivered extraction helper was tested on both original outer ZIP bundles in a fresh temporary directory; its full metadata/image byte fingerprint matched the working dataset. This temporary copy was removed, preserving the original archives and working data. Final run metrics, actual selected model, checkpoint hash and CSV validation are recorded in `final_results.md` after training.
