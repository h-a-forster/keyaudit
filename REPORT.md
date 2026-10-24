# keyaudit report (v0.1.0)

Date: 2026-10-24. Every number here comes from two committed result sets:
`results/study-v1/` (the real study, which stopped before the model stages) and
`results/simulation-v1/` (synthetic panels with known truth). Estimands and methods are
defined in [docs/DESIGN.md](docs/DESIGN.md). Deviations from the design are listed in DESIGN
section 14.

**Summary.**

- The model-based study on Open LLM Leaderboard responses has **not been run**. All 200
  candidate details repositories were unreadable (194 gated, 6 broken), and the study needs
  at least 60 models. So there is no model-based key-error rate for any real benchmark yet.
- Label-only estimates for the 6,810 MMLU-derived MMLU-Pro test items, from the MMLU-Redux
  2.0 relabels:
  - **1.48% [0.83%, 2.13%]** have a wrong key (narrow estimand).
  - **3.97% [2.68%, 5.25%]** have a wrong key, no correct option or several correct options
    (broad estimand).
  - Chemistry stands out: 17 of 115 labeled items, 14.8% [8.9%, 22.6%], Holm p < 0.001.
- On synthetic panels the model ranks planted mis-keys well when its assumptions hold: AP
  0.95, against 0.89 for the best baseline. Its rate without labels is biased low, and it is
  inflated by about 2 points when correctly keyed "trick" items are present. With 25% of
  items human-labeled, the calibrated estimator's bias stayed between -0.30 and +0.07 points
  in every scenario.
- The model-based bootstrap intervals in simulation-v1 are **not valid** (section 3.3). The
  bootstrap was changed after that run, and the coverage of the replacement has not been
  measured.
- This build has **not** had an independent audit.

## 1. What was run

| | study-v1 | simulation-v1 |
|---|---|---|
| Config | `configs/study.toml`, hash `a2773441...` | profile `default` (`keyaudit.study.simulation.PROFILES`) |
| Seed | 20261024 | 20261024 |
| Units | fetch 1/195, labels 1/1; `build` stopped (exit 3) | 70/70 (40 scenario panels, 30 coverage panels) |
| Wall time | 12.5 min on 1 worker (mostly about 3.5 s per denied request) | 38.6 min on 3 workers |
| Code | keyaudit 0.1.0.dev0 (M5) | keyaudit 0.1.0.dev0 (M5) |

`results/study-v1/provenance.json` records the data revisions (19 MMLU-Pro revisions,
MMLU-Redux 2.0, Platinum and the leaderboard contents) and the library versions (numpy
2.4.6, scipy 1.18.1, pyarrow 26.0.0, huggingface-hub 2.2.0).

**Panel access.** `fetch` tried 200 candidate models: 100 targets plus reserves, by accuracy
quintile.

- 194 returned HTTP 403. The account behind the token had not accepted those repositories'
  conditions. Only the account owner can accept them, and there is no API for it.
- 6 failed: 5 returned HTTP 400 (no parquet conversion) and 1 returned 404 (repository
  missing).
- 0 were readable, so `build` raised its precondition error.

The full list is in `results/study-v1/access.csv`. The details adapter has therefore never
read a real repository. It is tested only on synthetic samples shaped like the documented
schema.

## 2. Label-only results (MMLU-Pro, MMLU-derived items)

**Population:** the 6,810 MMLU-Pro test items whose source is `ori_mmlu`, at the canonical
revision `014c8cf5`. That is the revision that was current when the leaderboard stopped
evaluating. Without a panel there is nothing to fingerprint, so this choice is a fallback
(DESIGN 14, deviation 15).

**Labels:** MMLU-Redux 2.0 relabels, joined by question text.

| Join step | Count |
|---|---|
| Redux rows | 5,700 |
| Unique matches | 2,916 |
| Ambiguous matches (dropped) | 9 |
| Key text differs (excluded) | 23 |
| Labeled, key text agrees | 2,893 |
| Subject strata | 56 |

**Estimator:** the classical stratified estimator with a finite-population correction (the
difference estimator with no model predictions, f = 0). Redux sampled items per subject, so
each subject is weighted by its share of the population. This is why the stratified and raw
rates differ.

**Variance floor (changed after the audit).** With about 1.5% positives, 35 of the 56 subject
strata have no labeled positive and an estimated variance of zero. In a simulation of the
actual design (real per-subject population and labeled counts, true rates = observed subject
rates shrunk toward the overall rate) the original interval, 1.48% [0.95%, 2.01%], covered
only about 86% for the narrow estimand (93% for broad). Each stratum's variance is now floored
at the pooled labeled variance of its category; the point estimates are unchanged and the
same simulation gives about 98% coverage for narrow and 95% for broad (600-1,000 replicates;
the floored narrow interval is conservative). `tests/test_label_coverage.py` reruns a seeded
200-replicate version of this check (plain about 82%, floored about 98%).

| Estimand | Positives / labeled | Stratified estimate (95% CI) | Raw labeled rate (CP 95%) |
|---|---|---|---|
| `E_label` narrow (Redux `wrong_groundtruth`) | 56 / 2,893 | 1.48% [0.83%, 2.13%] | 1.94% [1.47%, 2.51%] |
| `E_label` broad (adds no correct and multiple correct answers) | 97 / 2,893 | 3.97% [2.68%, 5.25%] | 3.35% [2.73%, 4.08%] |

At the latest revision (`24ac2da5`, 2,883 labeled) the estimates agree to two decimals.

**By category** (narrow estimand, canonical revision). Each category is tested against all
others with a Fisher exact test, Holm-adjusted over 14 categories:

- Chemistry: 17 / 115 = 14.8% [8.85%, 22.61%], Holm p < 0.001.
- No other category differs after adjustment. Rates range from 0.0% (business, 0 / 104) to
  2.9% (health, 11 / 385).

These tests treat the labeled items in a category as a simple random sample and ignore the
subject strata. The chemistry effect is large enough that this is unlikely to change the
conclusion, but no stratified test has been run.

**Maintainer corrections.** After the canonical revision the MMLU-Pro maintainers changed 23
keys (21 from `ori_mmlu`, 2 from `stemez`). Redux had labeled 10 of these items `ok` and had
not labeled the other 13. None of the 56 Redux `wrong_groundtruth` items has been corrected.
The two efforts may have found different errors, or they may define a key error differently.
These counts alone cannot tell which.

**Platinum.** Only aggregates are committed, because the per-item labels are CC-BY-SA. Of the
200 reviewed items in each BBH subtask, 10 `object_counting` items were rejected. None were
rejected in `navigate` or `logical_deduction_three_objects` (`labels.json`,
`platinum_counts`).

**Not available without a panel:**

- the label-calibrated estimate, which needs model scores;
- model-based and transfer estimates for GPQA and BBH;
- validation of the ranking against human labels;
- the suspect-item list.

## 3. Simulation study (known truth)

The sizes follow DESIGN 7.2: 100 models, 3,000 items, K = 10 options, a true key-error rate
of 3% and 10 item groups. Each scenario has 10 replicate panels. Intervals are 95% intervals
for the mean over replicates. The scenarios are:

- `design`: independent models.
- `weak`: a panel about as weak as the 2024 leaderboard (median accuracy about 0.35, best
  model about 0.70).
- `families`: 10 model families that share item effects (sd 1.5).
- `tricks`: 3% of items have a correct key but violate A1-A2. Stronger models are drawn to
  one wrong option on them, which is the shared blind spot of DESIGN 5.2.

### 3.1 Rates (percent)

| Scenario | True rate | `E_model`, all items | Bias of `E_model` | Region share | `E_model` in region (true) | Calibrated, 25% labeled | Calibrated CI coverage |
|---|---|---|---|---|---|---|---|
| design | 2.87 | 2.45 [2.20, 2.69] | -0.42 [-0.48, -0.36] | 86% | 2.38 (2.44) | 2.86 [2.57, 3.15] | 10/10 |
| weak | 2.91 | 2.43 [2.30, 2.56] | -0.48 [-0.57, -0.40] | 68% | 2.29 (2.33) | 2.98 [2.76, 3.21] | 10/10 |
| families | 3.04 | 1.93 [1.79, 2.07] | -1.11 [-1.20, -1.02] | 86% | 1.76 (1.83) | 3.06 [2.90, 3.22] | 9/10 |
| tricks | 3.02 | 5.05 [4.70, 5.40] | +2.03 [+1.82, +2.23] | 86% | 4.93 (2.70) | 2.73 [2.26, 3.20] | 8/10 |

**Inside the competence region, without trick items, the model rate is close to the truth.**
The in-region bias is -0.05 to -0.06 points in `design`, `weak` and `families`. Outside the
region the panel cannot see mis-keys, so the rate over all items is biased low. In `families`
the -1.1 point bias comes mostly from outside the region: inside it the model gives 1.76%
against a true 1.83%.

**Shared blind spots break the model rate, as the identifiability result predicts.**

- At q <= 0.10, trick items are flagged in 86% [82%, 90%] of cases. True mis-keys are
  flagged at the same rate, 86% [81%, 90%]. Only 0.21% of other clean items are flagged.
- The model rate becomes 5.05% against a true 3.02%, and precision at q <= 0.10 falls from
  0.97 to 0.51.

No refinement of the model can fix this from responses alone. This is why the headline
estimand is calibrated against human labels.

**The calibrated estimator holds up in every scenario.**

- Its bias is between -0.30 and +0.07 points.
- Its standard error (0.24 to 0.44 points) is about half that of the labels-only estimator
  on the same labels (0.51 to 0.56 points).
- Interval coverage was 37/40 across scenarios, and 8/10 in `tricks`. With 10 replicates per
  scenario, these counts cannot tell 95% coverage apart from about 75%.

### 3.2 Ranking and operating point

Average precision against the planted mis-keys, and the operating point at q <= 0.10:

| Scenario | keymix | keymix in region | top-decile disagreement | item-total | scalability | 2PL sign LRT | one-coin DS | Precision / recall at q <= 0.10 |
|---|---|---|---|---|---|---|---|---|
| design | 0.947 | 0.999 | 0.889 | 0.542 | 0.541 | 0.554 | 0.425 | 0.97 / 0.82 |
| weak | 0.866 | 0.995 | 0.639 | 0.370 | 0.369 | 0.348 | 0.267 | 0.95 / 0.66 |
| families | 0.914 | 0.998 | 0.862 | 0.472 | 0.469 | 0.472 | 0.541 | 1.00 / 0.57 |
| tricks | 0.549 | 0.577 | 0.546 | 0.366 | 0.365 | 0.370 | 0.328 | 0.51 / 0.86 |

- keymix has the highest mean AP in every scenario. Its margin over top-decile disagreement
  is large for a weak panel (0.87 against 0.64) and small for families (0.91 against 0.86).
- In the three scenarios without tricks, the model's own FDR estimate (0.095 to 0.096) is
  conservative: the realised false discovery proportion is 0.00 to 0.05.
- With tricks the FDR estimate is badly anti-conservative: the realised proportion is 0.49.

These figures are means over 10 panels. Replicate intervals are in `summary.json`, and no
paired test between methods was run on the simulation.

### 3.3 Bootstrap intervals for the model rate are not valid in this run

The coverage panels (30 panels of 1,000 items and 60 models, B = 40) used the pigeonhole
bootstrap, which resamples both items and models. It reported coverage of 29/30 overall and
22/30 inside the competence region. **The 29/30 is not evidence that the interval works:**

- The median interval is 22 points wide, for a rate near 3%.
- The point estimate lies outside its own interval in 19 of 30 panels.

The cause: a model drawn twice in a replicate counts as two independent confirmations of the
same answer. This inflates the refitted rate by about 2-3x (DESIGN 14, deviation 20, now
covered by a regression test).

After this run the default became an item bootstrap with the panel held fixed. Its coverage
has **not** been measured. A smoke run on the quick profile (2 panels, B = 4) shows only that
it runs. The estimator is also biased low by about 1.0 point on these smaller panels, so an
interval that ignores the bias will under-cover even if the bootstrap itself is sound.

## 4. Limitations

- **Identifiability.** The model-based rate is valid only under A1 (truth curves rise with
  ability) and A2 (choice among distractors does not depend on ability).
  - A correctly keyed item that stronger models increasingly answer with one shared wrong
    option produces exactly the same responses as a mis-keyed item (DESIGN 5.2; the
    construction is checked to 1e-12).
  - The `tricks` scenario shows the size of the effect: at a 3% trick rate the model rate is
    about 2 points too high.
  - Outside the competence region only worst-case bounds are possible.
  - On real data the share of such items is unknown. A model-based rate without labels is
    therefore an estimate under assumptions that cannot be checked from responses, not a
    measurement.
- **Shared blind spots.** Leaderboard models share training data, base models and possibly
  contaminated corpora. A misconception common to most strong models is exactly the trick
  case. Families of fine-tunes also shrink the effective panel size. Real panels may be more
  correlated than the `families` scenario, which already cut recall at q <= 0.10 from 0.82
  to 0.57.
- **Sample of models.** The planned panel is the open-weight models that the Open LLM
  Leaderboard evaluated until early 2025. They are much weaker than current frontier models,
  which is why the `weak` scenario exists. A different panel sees different items, so
  competence-region shares and rates do not carry over to stronger panels. No panel was
  drawn in this run.
- **Human labels are not ground truth.**
  - The MMLU-Pro maintainers' 23 later corrections include 10 items that Redux labeled `ok`,
    and they corrected none of Redux's 56 `wrong_groundtruth` items.
  - The broad estimand is about 2.7 times the narrow one, so the "key error rate" depends on
    the definition used.
  - The Redux overlap covers only the MMLU-derived 57% of MMLU-Pro. The other 43% (STEM
    websites, TheoremQA, SciBench) has no human labels here, and nothing is claimed for it.
- **Simulation is not reality.** The simulated responses come from the model's own family,
  with deliberate violations in `weak`, `families` and `tricks`. The good results in 3.1 and
  3.2 are therefore a best case. With 10 replicates per scenario, the coverage statements are
  weak.
- **Untested on real data.** The leaderboard details adapter and the GPQA and BBH matrix
  construction have never read a real repository.
- **No independent audit.** The same build process wrote and checked the statistics, the code
  and this report. One bootstrap defect was found after the runs (section 3.3), and others
  may remain.

## 5. Reproducing

```sh
uv sync --group dev
uv run keyaudit simulate --profile default --out runs/simulation-v1 --results results/simulation-v1 --workers 3
uv run python scripts/run_study.py --config configs/study.toml   # needs HF_TOKEN with accepted gates
uv run keyaudit report --config configs/study.toml
```

Rerunning `simulate` into an existing run directory reuses its finished units, including the
pigeonhole coverage units. Use a new `--out` directory to measure item-bootstrap coverage.
