# Reproduce the reliability study

This study does not generate or promote a new final submission. Read the selection gate in [the reliability report](reports/reliability_report.md). The historical `train.py` reproduces the earlier five-candidate experiment and its composite selection rule; that rule is not the current scientific qualification gate or a reliability percentage.

Use Python 3.13 and install `requirements-reliability.txt` in a separate environment. The CPU run used torch 2.12.0+cpu and torchvision 0.27.0+cpu. A compatible CPU installation can be obtained from the official PyTorch CPU package index. No GPU, mixed precision, end-to-end CNN fine-tuning or neural early stopping was used. The two ImageNet networks are frozen feature extractors.

Run from the repository root. Replace `python` with the environment's executable and `data` with the official dataset root. Keep the same data and source versions when resuming; use new artifact directories after changes. Do not mix caches from different experiments.

```powershell
python -m pip install -r requirements-reliability.txt
python -m unittest discover -s tests -v
python data_checks.py --data-root data --artifacts artifacts/audit_v2
python train.py --data-root data --artifacts artifacts/audit_v2 --config config.yaml
python reliability_audit.py --data-root data --previous artifacts/audit_v2 --artifacts artifacts/reliability_v3
python reliability_experiments.py --data-root data --previous artifacts/audit_v2 --artifacts artifacts/reliability_v3 --family classical
python extract_transfer.py --data-root data --artifacts artifacts/reliability_v3 --download-only
python extract_transfer.py --data-root data --artifacts artifacts/reliability_v3
python extract_transfer.py --data-root data --artifacts artifacts/reliability_v3 --split test
python reliability_experiments.py --data-root data --previous artifacts/audit_v2 --artifacts artifacts/reliability_v3 --family transfer
python stress_diagnostics.py --data-root data --previous artifacts/audit_v2 --artifacts artifacts/reliability_v3
python validate_fold_ensemble.py --data-root data --previous artifacts/audit_v2 --artifacts artifacts/reliability_v3
python nested_diagnostics.py --data-root data --previous artifacts/audit_v2 --artifacts artifacts/reliability_v3
python profile_classical.py --data-root data --previous artifacts/audit_v2 --artifacts artifacts/reliability_v3
python build_reliability_report.py --data-root data --previous artifacts/audit_v2 --artifacts artifacts/reliability_v3 --reports reports --submission output/submission.csv
```

The final command requires the preserved submission being audited. Do not regenerate a CSV just to bypass a failed selection gate. `--legacy-metrics` optionally imports the older random-split experiment log; it is not required to reproduce the grouped study.

The report writer saves primary nested-ensemble OOF to `artifacts/reliability_v3/oof_predictions.csv`. The audit saves checkpoint evaluation diagnostics to `artifacts/reliability_v3/evaluation_probabilities.csv`. For the requested top-level delivery paths, copy these two files into `artifacts/` unchanged. They are local research outputs, excluded from Git, not organizer upload formats.

Each nested outer fold excludes the entire validation group from every member. All thresholds and calibrators are fit on its training side. Bootstrap resampling is paired, group-aware and composition-stratified with 10,000 resamples. The confidence interval is conditional on fitted OOF predictions, not a guaranteed hidden-test performance range. Separate noise/interpolation diagnostic copies do not replace primary validation or test observations.

Resuming completed candidate/fold artifacts avoids repeating work. An interrupted candidate with completed CV can resume its sector checks; its recorded peak memory then covers only the resumed stage. Timing and memory coverage limitations are recorded, not imputed. Partial nested-fold files are not treated as complete unless the fold's prediction and metadata files were saved.
