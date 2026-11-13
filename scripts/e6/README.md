# E6: keyaudit on real key defects (2026-11-09)

REPORT.md section 3.8. These scripts are kept as they ran (outside ruff), except that the
boards folder is now an argument instead of a fixed relative path. They call keyaudit as a
library and make no model calls.

## Committed results (`results/e6-benchmarks/`)

| File | What |
|---|---|
| `summary.json` | every number in REPORT 3.8; rebuilt by `score.py` from the files below |
| `labels.json` | census truth for geometric_shapes and date_understanding, the solver census per BBH subtask, the ARC and TruthfulQA sample designs (seeds, selected items), fit summaries |
| `flags-*.csv` | per item: key, proposed option, posterior `s`, q-value, flag (no question text, no responses) |
| `claims_arc_tqa.json` | the 12 ARC and TruthfulQA claims put to the second pass, with reasons and sources: `cls` is the final class that `score.py` scores ("wrong key" counts as wrong, every class but "refuted" as a defect), `first_pass_cls` the class it replaced, `second_pass` the refutation pass's verdict |
| `*_answers.txt`, `date_table.md` | the blind first-pass answers, by sheet number, and the date census table |

`claims_arc_tqa.json` quotes those items: ARC-Challenge text is CC-BY-SA-4.0 (Ai2,
`allenai/ai2_arc`), TruthfulQA text Apache-2.0. `date_table.md` quotes BBH stems (MIT).

```sh
uv run python scripts/e6/score.py          # rewrites results/e6-benchmarks/summary.json
```

## Rebuilding the raw boards (about 104 MB, not committed)

Leaderboard responses are not redistributed, so the boards are rebuilt locally into a folder
of your choice (`BOARDS` below). The study's gated v2 details need the same HF token and
accepted gates as `keyaudit fetch`; the v1 details are public.

| Data | Source | Revision |
|---|---|---|
| `BOARDS/bbh/*.json` | github.com/suzgunmirac/BIG-Bench-Hard | `9ee07bd481feebf959a6b59d61ea57bdcf30964d` |
| `BOARDS/bbh/upstream/` | github.com/google/BIG-bench | `092b196c1f8f14a54bbc62f24759d43bde46dd3b` |
| `BOARDS/arc_challenge/test.parquet` | huggingface.co/datasets/allenai/ai2_arc | `210d026faf9955653af8916fad021475a3f00453` |
| `BOARDS/truthfulqa/mc.parquet`, `generation.parquet` | huggingface.co/datasets/truthfulqa/truthful_qa | `741b8276f2d1982aa3d5b832d3ee81ed3b896490` |

```sh
# responses (one unit file per model, skipped on rerun; progress.log per benchmark)
uv run python scripts/e6/fetch_v2.py BOARDS/bbh . geometric_shapes,date_understanding,disambiguation_qa,ruin_names,sports_understanding 100
uv run python scripts/e6/fetch_v1.py BOARDS scripts/e6/panel_v1_final.txt
# fits and samples
uv run python scripts/e6/fit_v2.py BOARDS/bbh geometric_shapes,date_understanding,disambiguation_qa,ruin_names,sports_understanding
uv run python scripts/e6/fit_v1.py BOARDS arc_challenge      # and truthfulqa
uv run python scripts/e6/sample_sheets.py BOARDS             # seeded blinded sheets
uv run python scripts/e6/bbh_sheet.py BOARDS                 # judgement-subtask sheet
# census
uv run python scripts/e6/bbh_solvers.py BOARDS/bbh BOARDS/bbh/_solver_results.json
uv run python scripts/e6/bbh_date_census.py BOARDS/bbh/date_understanding.json
uv run python scripts/e6/geo.py BOARDS
uv run python scripts/e6/compare_upstream.py BOARDS/bbh
uv run python scripts/e6/compare_blind.py BOARDS arc_challenge results/e6-benchmarks/arc_answers.txt
# compact results
uv run python scripts/e6/derive_results.py BOARDS results/e6-benchmarks
uv run python scripts/e6/score.py
```

`fetch_v2.py` reads the panel from `runs/study-v1/` (the study's local run folder), so it
needs a completed `keyaudit fetch`. `list_v1.py`, `probe_v1.py` and `schema.py` were used to
choose the v1 panel and read the details schema; `census.py` and `estimates.py` print the
counts and estimates `score.py` now computes.
