# Local source-material discovery

Discovery was read-only. Searched the workspace recursively (including ignored data/model folders), its parent under Documents/ChatGPT, all accessible Documents, Downloads, Desktop, and the Codex attachment directory. Checked user-level likely project/upload directories and mounted filesystem roots. Only C: and its temporary-directory alias were mounted; no external drive was available. OneDrive Desktop/Documents were not present. Protected system directories were not searched and unrelated content was not inspected.

57 ZIP/TAR/7Z archives were found in these user locations; 44 ZIP central directories were inspected for competition contents. No notebook was found in these searched locations. Two renamed outer archives matched the competition by their internal metadata/image archives. Other archives were unrelated and are not listed here.

## 1. Official source material

- Downloads/Problem Statement - The Pareidolia Paradox.pdf (two pages; page 2 blank).
- Downloads/Train-20260902T160709Z-1-001.zip: Train/train_metadata.csv and Train/train_images.zip.
- Downloads/Test-20260902T160709Z-1-001.zip: Test/test_metadata.csv and Test/eval_images.zip.
- Workspace data/source/train/Train/train_metadata.csv and data/source/test/Test/test_metadata.csv.
- Workspace data/raw/train_images (7,854 PNGs) and data/raw/eval_images (2,000 PNGs).

Nested archive PNG basename sets exactly matched metadata IDs. Both extracted CSVs were compared with original archive CSV contents and matched exactly. No additional extraction was necessary. The subsequent data audit fully decodes the extracted images.

## 2. Existing solution source

The actual solution root is ParadoxProject, an existing Git project. Relevant files: ml/preprocessing.py, ml/dataset.py, ml/features.py, ml/baseline.py, ml/submission.py, ml/deep_learning.py; scripts/train_baseline.py, scripts/generate_submission.py; requirements.txt, requirements-deep-learning.txt; tests/test_preprocessing.py, tests/test_dataset.py, tests/test_features.py, tests/test_submission.py. Existing backend and frontend files were inspected only where relevant to understanding the prior inference path. They are not required in the competition delivery.

## 3. Generated artifacts

- artifacts/models/solar_logistic.joblib
- artifacts/models/oof_predictions.csv
- artifacts/metrics/pareidolia-baseline-2026.json
- artifacts/cache/train_raw_features.npz
- artifacts/cache/train_solar_features.npz
- artifacts/test_predictions.csv
- Existing exported UI data and screenshots (not competition deliverables).

## 4. Existing final output

- artifacts/submission.csv
- public/submission.csv and dist/client/submission.csv (copies for the earlier interface).

All material necessary for audit, training, inference, and validation exists. No re-upload was requested.
