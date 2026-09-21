# End-to-end GPU morphology training report

## Recommendation

**FALLBACK RETAINED.** The exact validated fallback remains the recommended competition CSV.

SHA-256: `d613c160a15a23089d800ea3b056afedc9167da696689c6a4daec8ebae81cc72`

The GPU candidate changes 220 of 2,000 labels relative to the fallback, but it failed every prespecified morphology-improvement threshold. Hidden evaluation accuracy remains unknown.

## Full grouped OOF result

The selected screening configuration was **EfficientNet-B0 with the `circle180` policy**. All 4,010,110 parameters were fine-tuned from ImageNet initialization with CUDA mixed precision, class-balanced cross entropy, AdamW, cosine decay, early stopping, and checkpointing. Three StratifiedGroupKFold models used detected duplicate groups. The threshold remained fixed at 0.5.

- Pooled balanced accuracy: **68.00%**
- 95% duplicate-group bootstrap interval: **66.96%–69.06%**
- Depth recall: **68.82%**
- Rise recall: **67.18%**
- Macro within-azimuth-bin balanced accuracy: **50.85%**
- Worst azimuth-bin balanced accuracy: **48.49%**
- Worst-fold balanced accuracy: **66.82%**

| Fold | Pooled BA | Macro azimuth-bin BA | Worst azimuth-bin BA |
| --- | ---: | ---: | ---: |
| 0 | 67.09% | 51.18% | 48.77% |
| 1 | 70.08% | 49.33% | 44.08% |
| 2 | 66.82% | 51.96% | 48.18% |

Every OOF row was predicted by a model that did not train on its duplicate group. Validation and evaluation used identical deterministic geometry and no augmentation. Training added only mild post-normalization brightness, contrast, and Gaussian-noise augmentation.

## Fast screening

One grouped fold was used for each architecture/policy screen. The fixed selection rule prioritized macro 45-degree-bin balanced accuracy, with pooled balanced accuracy as the tie-breaker.

| Architecture | Boundary policy | Pooled BA | Macro azimuth-bin BA | Worst azimuth-bin BA |
| --- | --- | ---: | ---: | ---: |
| EfficientNet-B0 | reflect | 71.50% | 51.49% | 46.07% |
| EfficientNet-B0 | crop180 | 68.93% | 50.42% | 47.76% |
| EfficientNet-B0 | circle180 | 66.78% | 53.98% | 47.97% |
| ResNet-18 | circle180 | 63.74% | 50.68% | 46.79% |

`reflect` keeps the full normalized frame. `crop180` uses a fixed central 180×180 crop that remains inside the original image support at every rotation, including 45 degrees, then resizes to 224×224. `circle180` adds the same fixed radius-86 circular mask to every image. ConvNeXt-Tiny was not run because the time-bounded screening results already failed to show convincing within-bin morphology signal.

## Direct comparison

| System | Pooled BA | Macro azimuth-bin BA | Worst azimuth-bin BA |
| --- | ---: | ---: | ---: |
| GPU EfficientNet-B0 circle180 | 68.00% | 50.85% | 48.49% |
| Existing metadata logistic | 75.42% | 50.13% | 47.60% |
| Existing reflect RBF image model | 70.93% | 49.91% | 48.24% |
| Angle-only diagnostic | 78.14% | 49.81% | 48.91% |

The metadata and angle-only models are shortcut diagnostics, not morphology models. Probability-level ensembling was not promoted because the GPU model did not first establish robust morphology signal; combining it with a known azimuth-confounded model would not satisfy the scientific gate.

## Gate decision

| Prespecified check | Result |
| --- | --- |
| All three grouped folds complete | Pass |
| Both class recalls at least 60% | Pass |
| Pooled BA exceeds the existing image model | **Fail** |
| Macro within-bin BA improves by at least 3 percentage points | **Fail** |
| Worst-bin BA improves by at least 2 percentage points | **Fail** |

The experiment therefore does not demonstrate dependable morphology recognition. The exact fallback CSV remains the recommendation.

## Reproducibility and outputs

- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- PyTorch: 2.14.0+cu130
- Full-run maximum epochs: 12; patience: 3; batch size: 32
- Selected epochs: fold 0 epoch 1, fold 1 epoch 3, fold 2 epoch 1
- Training: `gpu/train.py`
- Inference: `gpu/inference.py`
- Colab fallback: `notebooks/pareidolia_gpu_finetuning.ipynb`

The candidate passed the 2,000-row schema and exact-ID-order checks. Its mean predicted-label confidence was 72.73% and mean across-fold probability standard deviation was 0.083153; these are prediction diagnostics, not evaluation accuracy. Checkpoints, per-image predictions, and datasets remain intentionally untracked.

## Limitations

The 10,000-resample interval conditions on fixed OOF predictions and excludes model selection, retraining, and evaluation-set shift. Architecture and boundary-policy selection used screening validation and therefore introduces selection bias. Three-fold training was a deliberate time-bounded compromise. No claim of hidden-test accuracy or robust morphology performance is made.
