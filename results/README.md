# results

Small derived results of finished studies, one folder per study (written by `keyaudit report`).
No raw data, no per-model responses, no GPQA question text. `simulation-v2` is the
simulation study with the current code; `simulation-v1` (2026-10-24) predates the
one-function calibration and tie-aware AP fixes (`15d5b7a`, `593ed9f`) and is kept only as
the record REPORT 4.3 cites for the pigeonhole bootstrap. `items-cov-v1` and `items-cov-v2`
are item-bootstrap coverage runs (`keyaudit simulate --profile coverage`, seeds 20261024 and
20261025). `calibrated-cov-v1`
is the coverage of the label-calibrated interval on the real labeled design
(`scripts/calibrated_coverage.py`, seed 20261030). `option-mapping-check` compares study-v1
with a refit under the fixed option mapping (`scripts/option_mapping_check.py`).
`e6-benchmarks` holds the compact results of E6 (REPORT 3.8; `scripts/e6/`).
