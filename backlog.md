# Backlog

## In progress

## Waiting on me

- Branch `polish-2026-10-30` is merged to main through a pull request with CI green, and v0.1.1 is tagged with `docs/release-notes-0.1.1.md` as its release notes.
- Private vulnerability reporting is enabled on the repository, so the route in `SECURITY.md` works.
- A stronger, less correlated panel is approved or rejected as a source (for example `evaleval/EEE_datastore`); it is the only way the model-only rate could become informative.

## Decided

## Proposed

- `results/study-v1/` comes from the fixed option mapping: stages from `build` on are rerun (offline, about 90 min on 3 workers) and the numbers in README, REPORT and the release notes follow.
- `results/simulation-v1/` is rerun with the current code (one-function calibration, tie-aware AP).
- The simulation has a scenario with one dominant family like the real panel, and item-bootstrap coverage is measured in the `weak` and `families` scenarios.
- Option hashes are case-sensitive (needs a re-fetch), which also recovers the GPQA item dropped by case-folding.
- The Platinum MMLU high-school maths overlap is evaluated.
- The source of the 9 GPQA duplicate-option items is known: GPQA itself or the harness.
