# Verified final results

Measured on the supplied official data. Balanced accuracy is (depth recall + rise recall) / 2. These are validation scores, not leaderboard scores.

## Comparison

| Method | Grouped OOF BA | Depth recall | Rise recall | Mean held-out sector BA |
| --- | --- | --- | --- | --- |
| existing-metadata-logistic | 75.42% | 80.41% | 70.42% | 50.27% |
| reflect-logistic | 69.45% | 76.59% | 62.30% | 50.06% |
| reflect-rbf-balanced | 70.93% | 74.77% | 67.08% | 50.57% |
| reflect-rbf-unbalanced | 70.82% | 71.93% | 69.70% | 49.75% |
| angle-only-diagnostic | 78.14% | 82.38% | 73.90% | 49.87% |

The historical random-stratified, pooled-threshold result was **75.38%**. A full rerun from freshly extracted features reproduced it (maximum OOF probability difference 6.67e-16). It is not directly comparable to these stricter grouped/inner-threshold scores.

## Selection and output

Selected **reflect-rbf-balanced**, using the predeclared mean of grouped OOF BA and mean held-out-sector BA among image-only candidates. Selection score: 0.607490. Grouped OOF BA: **70.93%**. Mean held-out-sector BA: **50.57%**. Confusion matrix (true rows/predicted columns: depth,rise): [[2134, 720], [1646, 3354]].

The checkpoint contains five trained fold members and their training-only sigmoid calibrators/thresholds. Final inference averages threshold-adjusted decision scores and applies 0.5. OOF measures a single held-out member per observation, not the full ensemble's unbiased performance. Model selection used these validation results, and sector performance must be considered alongside the ordinary OOF score.

Final CSV: exactly 2000 rows, header `image_id,label`, integer labels only, no nulls, no duplicate IDs and exact metadata ID/order match. Prediction counts: depth 376, rise 1624. This distribution is not an accuracy measurement.

CSV SHA-256: `d613c160a15a23089d800ea3b056afedc9167da696689c6a4daec8ebae81cc72`

Exact inference checkpoint: `pareidolia_final_model.joblib` (50,934,522 bytes).

Checkpoint SHA-256: `ce1bbc1fe409513104452082f305a0c0a62316e054a6f2b0c1895430829150cf`

Data fingerprint: `ef56180f7bedc1ebb902ba02043bc0b753705b20437ce57aae5c9dde6faccfd3`.

Full new training and inference ran in a newly created Python 3.13 virtual environment, installed only from the delivered requirements. Dependency check passed. Root regression tests: 18 passed; portable repository contract tests: 5 passed. OOF groups were independently checked to be disjoint across folds. Original model inference reproduced the prior CSV byte for byte. No CUDA GPU, no CNN, no augmentation, no test-time augmentation.

## Selected-model OOF performance by azimuth

| Azimuth | N | OOF BA | Depth recall | Rise recall |
| --- | --- | --- | --- | --- |
| 0-45 | 693 | 48.81% | 25.61% | 72.01% |
| 45-90 | 693 | 49.74% | 9.64% | 89.84% |
| 90-135 | 722 | 51.60% | 8.60% | 94.59% |
| 135-180 | 706 | 49.19% | 2.67% | 95.72% |
| 180-225 | 712 | 48.24% | 6.41% | 90.06% |
| 225-270 | 685 | 50.91% | 30.34% | 71.48% |
| 270-315 | 1818 | 50.16% | 88.45% | 11.87% |
| 315-360 | 1825 | 50.64% | 86.82% | 14.46% |

## Interpretation limits

The data has exact duplicates, metadata-dependent label conflicts and substantial azimuth/label correlation plus train/evaluation azimuth shift. Grouping and sector tests expose part of this risk but cannot prove physical feature understanding. Mirrored padding avoids new black fill triangles; original black image regions and orientation cues remain. Probabilities are not externally calibrated confidence guarantees. There is no evaluation ground truth, so official balanced accuracy is unknown. Public model/repository status is tracked separately in the final checklist; no form or LinkedIn post has been submitted automatically.

## existing-metadata-logistic

Five-fold CV computation: 51.4 seconds (excludes feature extraction and sector tests).

| Fold | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| 1 | 75.40% | 77.23% | 73.57% | 0.635 | [[441, 130], [264, 735]] |
| 2 | 75.44% | 78.11% | 72.77% | 0.615 | [[446, 125], [272, 727]] |
| 3 | 75.20% | 84.76% | 65.63% | 0.675 | [[484, 87], [344, 657]] |
| 4 | 76.11% | 80.39% | 71.83% | 0.630 | [[459, 112], [282, 719]] |
| 5 | 74.94% | 81.58% | 68.30% | 0.680 | [[465, 105], [317, 683]] |


Held-out azimuth sectors (all duplicate groups touching validation are excluded from training):

| Sector | N | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- |
| 0-90 | 1386 | 49.72% | 35.15% | 64.29% | 0.660 | [[58, 107], [436, 785]] |
| 90-180 | 1428 | 49.23% | 7.74% | 90.71% | 0.690 | [[13, 155], [117, 1143]] |
| 180-270 | 1397 | 52.13% | 40.12% | 64.15% | 0.660 | [[67, 100], [441, 789]] |
| 270-360 | 3643 | 50.00% | 0.00% | 100.00% | 0.500 | [[0, 2354], [0, 1289]] |

## reflect-logistic

Five-fold CV computation: 18.8 seconds (excludes feature extraction and sector tests).

| Fold | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| 1 | 68.71% | 72.85% | 64.56% | 0.655 | [[416, 155], [354, 645]] |
| 2 | 69.08% | 82.31% | 55.86% | 0.685 | [[470, 101], [441, 558]] |
| 3 | 69.44% | 77.23% | 61.64% | 0.640 | [[441, 130], [384, 617]] |
| 4 | 70.55% | 75.66% | 65.43% | 0.640 | [[432, 139], [346, 655]] |
| 5 | 69.46% | 74.91% | 64.00% | 0.665 | [[427, 143], [360, 640]] |


Held-out azimuth sectors (all duplicate groups touching validation are excluded from training):

| Sector | N | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- |
| 0-90 | 1386 | 49.70% | 47.88% | 51.52% | 0.635 | [[79, 86], [592, 629]] |
| 90-180 | 1428 | 49.64% | 8.33% | 90.95% | 0.665 | [[14, 154], [114, 1146]] |
| 180-270 | 1397 | 50.91% | 53.29% | 48.54% | 0.645 | [[89, 78], [633, 597]] |
| 270-360 | 3643 | 50.00% | 0.00% | 100.00% | 0.500 | [[0, 2354], [0, 1289]] |

## reflect-rbf-balanced

Five-fold CV computation: 224.5 seconds (excludes feature extraction and sector tests).

| Fold | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| 1 | 70.69% | 69.00% | 72.37% | 0.635 | [[394, 177], [276, 723]] |
| 2 | 71.17% | 78.28% | 64.06% | 0.680 | [[447, 124], [359, 640]] |
| 3 | 70.21% | 74.08% | 66.33% | 0.650 | [[423, 148], [337, 664]] |
| 4 | 71.58% | 78.63% | 64.54% | 0.660 | [[449, 122], [355, 646]] |
| 5 | 70.98% | 73.86% | 68.10% | 0.660 | [[421, 149], [319, 681]] |


Held-out azimuth sectors (all duplicate groups touching validation are excluded from training):

| Sector | N | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- |
| 0-90 | 1386 | 50.15% | 50.91% | 49.39% | 0.645 | [[84, 81], [618, 603]] |
| 90-180 | 1428 | 49.44% | 1.19% | 97.70% | 0.625 | [[2, 166], [29, 1231]] |
| 180-270 | 1397 | 52.70% | 49.70% | 55.69% | 0.605 | [[83, 84], [545, 685]] |
| 270-360 | 3643 | 50.00% | 0.00% | 100.00% | 0.500 | [[0, 2354], [0, 1289]] |

## reflect-rbf-unbalanced

Five-fold CV computation: 246.6 seconds (excludes feature extraction and sector tests).

| Fold | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| 1 | 69.79% | 68.30% | 71.27% | 0.655 | [[390, 181], [287, 712]] |
| 2 | 71.56% | 74.26% | 68.87% | 0.655 | [[424, 147], [311, 688]] |
| 3 | 70.77% | 72.50% | 69.03% | 0.640 | [[414, 157], [310, 691]] |
| 4 | 71.21% | 74.78% | 67.63% | 0.645 | [[427, 144], [324, 677]] |
| 5 | 70.76% | 69.82% | 71.70% | 0.640 | [[398, 172], [283, 717]] |


Held-out azimuth sectors (all duplicate groups touching validation are excluded from training):

| Sector | N | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- |
| 0-90 | 1386 | 49.67% | 57.58% | 41.77% | 0.670 | [[95, 70], [711, 510]] |
| 90-180 | 1428 | 49.60% | 2.38% | 96.83% | 0.630 | [[4, 164], [40, 1220]] |
| 180-270 | 1397 | 49.73% | 56.29% | 43.17% | 0.655 | [[94, 73], [699, 531]] |
| 270-360 | 3643 | 50.00% | 0.00% | 100.00% | 0.500 | [[0, 2354], [0, 1289]] |

## angle-only-diagnostic

Five-fold CV computation: 20.5 seconds (excludes feature extraction and sector tests).

| Fold | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| 1 | 77.86% | 80.74% | 74.97% | 0.590 | [[461, 110], [250, 749]] |
| 2 | 77.79% | 82.31% | 73.27% | 0.580 | [[470, 101], [267, 732]] |
| 3 | 78.06% | 84.59% | 71.53% | 0.580 | [[483, 88], [285, 716]] |
| 4 | 79.16% | 83.19% | 75.12% | 0.565 | [[475, 96], [249, 752]] |
| 5 | 77.83% | 81.05% | 74.60% | 0.580 | [[462, 108], [254, 746]] |


Held-out azimuth sectors (all duplicate groups touching validation are excluded from training):

| Sector | N | BA | Depth recall | Rise recall | Threshold | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- |
| 0-90 | 1386 | 46.76% | 6.06% | 87.47% | 0.600 | [[10, 155], [153, 1068]] |
| 90-180 | 1428 | 50.00% | 0.00% | 100.00% | 0.620 | [[0, 168], [0, 1260]] |
| 180-270 | 1397 | 52.73% | 16.77% | 88.70% | 0.585 | [[28, 139], [139, 1091]] |
| 270-360 | 3643 | 50.00% | 0.00% | 100.00% | 0.500 | [[0, 2354], [0, 1289]] |
