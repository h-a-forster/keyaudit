# results

Small derived results of finished studies, one folder per study (written by `keyaudit report`).
No raw data, no per-model responses, no GPQA question text. `simulation-v1` is the
simulation study; `items-cov-v1` and `items-cov-v2` are item-bootstrap coverage runs
(`keyaudit simulate --profile coverage`, seeds 20261024 and 20261025). `calibrated-cov-v1`
is the coverage of the label-calibrated interval on the real labeled design
(`scripts/calibrated_coverage.py`, seed 20261030).
