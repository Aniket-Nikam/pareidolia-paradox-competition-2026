# The Pareidolia Paradox

Reproducible CPU classification of 256 x 256 grayscale lunar surface crops. Class **0 (Depth)** means craters, holes and depressions; class **1 (Rise)** means mounds, hills, rocks and boulders. The competition deliverable is a prediction CSV. No frontend, API or dashboard is required.

**Current qualification: final selection gate NOT PASSED.** The expanded study found no defensible new winner. The previous CSV and checkpoint are preserved, not certified as robust. Read the [current reliability report](reports/reliability_report.md) and [reproduction commands](RELIABILITY.md) before using them. Its nested-ensemble OOF estimate supersedes the historical single-member estimate below. Actual hidden-test accuracy is unknown.

Current nested ensemble OOF balanced accuracy: **71.49%**, with a 10,000 group-bootstrap 95% interval of **70.46%–72.52%**. Depth recall is 74.81%; Rise recall is 68.18%. Worst sun-angle-bin balanced accuracy is only **47.68%**. The interval conditions on fitted OOF predictions and does not establish a minimum future test accuracy.

## Data

Use the organizer-provided files locally. Datasets are intentionally not included in this repository. Expected layout (nested original folder layouts also work):

```text
data/
  train_metadata.csv
  test_metadata.csv
  train_images/train_00001.png ... (7,854 images)
  eval_images/eval_00001.png ... (2,000 images)
```

Training columns: `image_id,sun_azimuth_angle,label`. Evaluation columns: `image_id,sun_azimuth_angle`. The evaluation archive may be named `eval_images.zip` or `test_images.zip`. Every basename and row order is validated. Point `--data-root` to a tree containing exactly one of each metadata CSV and one copy of every PNG.

## Installation

Tested with Python 3.13 on Windows, CPU only. Run from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux/macOS, create the environment with `python3.13 -m venv .venv` and replace `.\.venv\Scripts\python.exe` in the following commands with `.venv/bin/python`.

If using the supplied outer ZIP bundles, extract only the relevant data into a new directory:

```powershell
.\.venv\Scripts\python.exe prepare_data.py --sources "C:\path\Train-20260902T160709Z-1-001.zip" "C:\path\Test-20260902T160709Z-1-001.zip" --destination data
```

For individual archives, pass `train_images.zip`, `eval_images.zip`, `train_metadata.csv`, and `test_metadata.csv` together after `--sources`. Skip extraction when an existing data tree is already available; use its path instead of `data` below. The extractor rejects overwrites, unsafe archive members and unexpected sizes.

## Validate and train

```powershell
.\.venv\Scripts\python.exe data_checks.py --data-root data --artifacts artifacts
.\.venv\Scripts\python.exe train.py --data-root data --artifacts artifacts --config config.yaml
```

This historical reproduction run compares five methods with five-fold grouped validation and four azimuth-sector stress tests. It saves `artifacts/pareidolia_final_model.joblib`, aggregate/per-fold metrics, OOF predictions and fingerprinted feature caches. Its historical composite selection rule is not the new qualification gate or a reliability percentage. CPU runtime is several minutes or longer depending on hardware. Use the identical command with `--resume` to reuse completed experiments. Partial folds rerun. Changed data, configuration or model source requires a fresh artifacts directory; do not relabel old caches as current. The expanded study and current selection decision are reproduced separately using `RELIABILITY.md`.

## Inference and submission validation

Place the downloaded checkpoint at `artifacts/pareidolia_final_model.joblib`, or use the locally trained one:

```powershell
.\.venv\Scripts\python.exe inference.py --data-root data --checkpoint artifacts/pareidolia_final_model.joblib --artifacts artifacts --output output/submission.csv
.\.venv\Scripts\python.exe validate_submission.py --submission output/submission.csv --metadata data/test_metadata.csv --report artifacts/submission_validation.json
```

If metadata remains nested, give the validator its actual path. Inference currently validates the complete train/evaluation data layout, so keep both splits available. Only `output/submission.csv` is the prediction upload; it has exactly 2,000 data rows and two columns. Other artifacts are diagnostic, not submission columns. `example/submission_format.csv` is only a two-row format illustration, not an actual submission.

## Model and physics

Selected method: **reflect-rbf-balanced**. Image-only candidates standardize 444 descriptors: 144 downsampled intensity values, 288 unsigned HOG values, five quantiles and seven global statistics. The existing metadata comparator adds four angle features (448 total), while the angle-only diagnostic uses just those four features. Linear baselines use balanced logistic regression; nonlinear candidates use an RBF SVM with C=3 and gamma=`scale`, comparing balanced versus unweighted class losses. Direct sine/cosine angle features are used only by comparison/diagnostic models, not the final image-only model.

Each image is matched to its metadata and passed through `ml/preprocessing.py::normalize_solar_azimuth` with **`-sun_azimuth_angle`**. Pillow defines positive angles as counter-clockwise, so the negative value performs the required normalization. We reflect-pad each side by ceil(half the image diagonal)+4 pixels, rotate with bicubic interpolation about the same center, then center-crop to the original 256 x 256. The descriptor stage resizes to 48 x 48 with Lanczos. Padding avoids new black fill triangles but does not remove original black regions or all orientation cues. Identical deterministic preprocessing is used for training, validation and evaluation.

No stochastic augmentation, flips, arbitrary extra rotations or test-time augmentation are used in the deployed model. It uses no pretrained network or synthetic data. The expanded comparison also evaluates frozen ImageNet ResNet-18 and MobileNetV3-Small feature extractors; neither was deployed. Separate requested noise/interpolation diagnostic copies never replace primary validation or test inputs.

## Validation and reproducibility

Balanced accuracy is the mean of the two class recalls, giving depth and rise equal importance despite 2,854 versus 5,000 training examples. Seed 2026 and five StratifiedGroupKFold outer splits are fixed. Exact/near-duplicate groups stay together. Within each outer training partition, an inner grouped holdout fits sigmoid calibration and selects the balanced-accuracy threshold on a 0.20-0.80 grid (0.005 steps). The base classifier is then refitted on the outer training set. Neither thresholds nor calibration use outer validation labels.

The final model bundles five fold members. Each member's probability is converted to a threshold-adjusted decision score with cutoff 0.5. Final labels use the mean adjusted score at 0.5; the accompanying mean raw probability is diagnostic. Class imbalance is handled by validation-targeted thresholds and by class weighting where selected. Evaluation labels are never consulted or inferred from duplicate lookup.

| Method | Grouped OOF BA | Depth recall | Rise recall | Mean held-out sector BA |
| --- | --- | --- | --- | --- |
| existing-metadata-logistic | 75.42% | 80.41% | 70.42% | 50.27% |
| reflect-logistic | 69.45% | 76.59% | 62.30% | 50.06% |
| reflect-rbf-balanced | 70.93% | 74.77% | 67.08% | 50.57% |
| reflect-rbf-unbalanced | 70.82% | 71.93% | 69.70% | 49.75% |
| angle-only-diagnostic | 78.14% | 82.38% | 73.90% | 49.87% |

Historical single-held-out-member OOF BA: **70.93%**. Sector-held-out mean BA: **50.57%**. These are not the newer nested five-member ensemble estimate. The [current reliability report](reports/reliability_report.md) contains the primary estimate, 10,000 group-bootstrap interval, all candidate comparisons and failed selection gate. Earlier results remain in [historical results](reports/final_results.md); source/data risks are in [audit report](reports/audit_report.md).

## Exact model download

Public weights link pending. The exact checkpoint is ready locally; upload/share it before completing the competition form.

Filename: `pareidolia_final_model.joblib`. This is a sklearn ensemble, not a PyTorch `.pth` file. All five required members are bundled in one file.

SHA-256: `ce1bbc1fe409513104452082f305a0c0a62316e054a6f2b0c1895430829150cf`

```powershell
Get-FileHash artifacts/pareidolia_final_model.joblib -Algorithm SHA256
```

Only load a checkpoint from a trusted source after checking its hash: joblib/pickle can execute code. Use the pinned dependency versions for model compatibility.

## Limitations

The source data contains pixel-identical images with different labels/metadata, and azimuth is correlated with labels while its distribution changes in evaluation. Held-out-direction and within-angle-bin performance are near chance: robust generalization has not been demonstrated. Raw-group and azimuth-block validation help measure this risk but do not eliminate unknown related scenes. Reflect padding introduces mirrored content. Calibration and threshold share an inner holdout, and refitting may shift calibration. Model selection uses the reported validation results, so scores are not untouched test estimates. Historical OOF evaluates one held-out member per row. The newer nested OOF evaluates five-member ensembles with smaller training partitions than deployment. Confidence intervals exclude model selection and deployment shift. Official evaluation accuracy is unknown. CPU-only comparisons used frozen CNN features, not end-to-end fine-tuning.

Competition deadline supplied by the organizer: **21 September 2026, 11:59 PM IST**. Publishing code or generating predictions does not submit the entry.
