# keyaudit design

Status: v0.1 plan, written at scaffold time (2026-10-24). Module stubs in `src/keyaudit/`
follow this document. When code and this document disagree, fix one of them in the same
commit and log the deviation in section 14.

## 1. Question, estimands and claims

How many answer keys in widely used multiple-choice LLM benchmarks are wrong, and which
items are they? keyaudit answers this from existing per-item model responses (no new model
runs) and calibrates the answer against human relabels.

**Estimands** (per benchmark, and per reporting group g = MMLU-Pro category, source or MMLU
subject):

- `E_label(g)`: the share of items in g that human auditors following the MMLU-Redux
  protocol would mark `wrong_groundtruth` (primary, "narrow") or any of `wrong_groundtruth`,
  `no_correct_answer`, `multiple_correct_answers` ("broad"). This is the headline estimand
  where labels exist. It is operational: it inherits the auditors' own error.
- `E_model(g)`: the share of items whose key differs from the latent true option under the
  mis-keyed latent class model of section 5. Meaningful only under the identifying
  assumptions of section 5.2.

**Deliverables** (v0.1):

| Deliverable | Evidence |
|---|---|
| MMLU-Pro key-error rate with 95% intervals, for the MMLU-derived items (label-calibrated) and for the other sources (model-based and transfer, marked as extrapolation) | sections 6.7, 8 |
| Per-category rates with intervals and Holm-adjusted comparisons | sections 6.7, 6.8 |
| A ranked list of suspect items: item id, key, proposed option, posterior, q-value, panel competence, leave-one-family-out stability | sections 6.5, 7.4 |
| Precision and recall against MMLU-Redux 2.0 (and, smaller, Platinum and maintainer corrections) with intervals, against five baselines | section 7 |
| GPQA and BBH: the same outputs, with an explicit statement of where the panel is too weak to identify anything | sections 5.3, 8 |
| A stated identifiability limit: what responses alone cannot tell apart | section 5.2 |

**Claims we will not make**: a GPQA error rate presented as calibrated (no public human
relabel of the leaderboard's GPQA items); any statement that a flagged item is wrong without
a human check; precision on items outside the labeled overlap.

## 2. Prior art and the gap

| Work | What it does | What it leaves open |
|---|---|---|
| Northcutt, Athalye, Mueller (2021), confident learning; `cleanlab` (Apache-2.0) | label errors in classifier test sets from model out-of-sample probabilities | single-model scores, not a panel of independent LLMs; no calibrated rate |
| Gema et al. (2024/2025), MMLU-Redux | 5,700 MMLU items relabelled by experts (100 per subject); 6.5% non-`ok` overall | human labels only; 4-option MMLU, not MMLU-Pro |
| Vendrow et al. (2025), Platinum benchmarks | 15 benchmarks cleaned; flags items where frontier models disagree, then humans review | human review of model disagreements; rejected/revised status per item |
| Truong et al. (NeurIPS 2025), "Fantastic Bugs" | three measurement-theory statistics (inter-item tetrachoric correlation, scalability Z_j, item-total correlation) on HELM response matrices, up to 84% precision@50 by expert review | binary responses, no proposed answer, no rate estimate, recall only for GSM8K |
| Land and Bikel (EMNLP 2026, arXiv 2605.30504) | 4PL IRT over 114 models; indicator = log-lik(d_i=0) - log-lik(d_i=1); ~95% precision in the top 200 | ground truth is a GPT-5.4 consensus, not humans; no recall; no calibrated rate; GPQA items assumed correct |
| Nakamura, Tan, Yean (2026, arXiv 2609.22133) | LLM and expert coders disagree on the same ambiguous texts; ambiguity-aware bounds | political-science coding, not benchmark keys; supports the "shared ambiguity" concern |
| Dawid and Skene (1979); Whitehill et al. (2009), GLAD | latent true label from many annotators with ability and item difficulty | the template for section 5; not applied to LLM panels with a single key |

At scaffold time no public work, to our knowledge, combines (a) a calibrated benchmark-level
key-error rate with intervals, (b) validation of the ranking against independent human
relabels with both precision and recall, and (c) MMLU-Pro. Say "to our knowledge" in any
user-facing text.

## 3. Scope review (decisions taken at scaffold time)

Implementation budget: about 4-5 hours of agent time, CPU only, at most 3 worker processes.

1. **Data access needs a Hugging Face account.** Every `open-llm-leaderboard/*-details`
   repository is gated with automatic approval ("agree to share your contact information"),
   verified on 2026-10-24 for 1,000 of them. There is no public API to accept the conditions;
   the account owner clicks "Agree" once per repository. The study therefore runs on the
   repositories the token can read: `fetch` records each 401/403 as `denied` and continues;
   `run` needs at least `panel.min_models` readable models. Benchmark and label datasets
   need no login.
2. **MMLU-Pro is the headline benchmark.** It is the only leaderboard benchmark with a large
   random human-labelled overlap (about 2,880 items with MMLU-Redux labels, section 4.2).
   GPQA (546 extended items, panel accuracy at most about 0.42 against 0.25 chance) is
   reported as an identifiability case, not a calibrated rate. BBH is limited to the three
   subtasks Platinum covers.
3. **Column projection instead of full downloads.** The raw MMLU-Pro sample file is about
   400 MB per model and its parquet conversion about 80 MB; we read only the footer and the
   needed column chunks with HTTP range requests (expected 1-5 MB per model, enforced budget
   15 MB per model, 3 GB total).
4. **EM with a profile likelihood and a bootstrap, not MCMC.** Fully Bayesian IRT with a
   latent key over 1.2M responses does not fit the budget. Uncertainty comes from a
   pigeonhole bootstrap (items and models resampled) and, for the headline, from the
   design-based variance of the difference estimator.
5. **Argmax responses only.** Per-option log-likelihoods are stored, but v0.1 uses the
   chosen option. A soft-response variant is future work.
6. **Cut**: tetrachoric correlations (O(I^2) for 12k items), an LLM judge, re-running any
   model, a 4PL reproduction of Land and Bikel (we use a binary sign test in its spirit),
   Open LLM Leaderboard v1 (MMLU) details, the Every Eval Ever frontier-model samples
   (`evaleval/EEE_datastore`, MIT, 14 frontier models on MMLU-Pro): not a named source, so
   adding it needs explicit sign-off. It is the most promising extension, because strong
   models make mis-keys identifiable.
7. **GPQA validation** from the expert-validator columns, if the leaderboard details carry
   them, is a stretch goal in M4. GPQA question text never leaves the local cache.

## 4. Data

### 4.1 Sources, licences and pins

| Source | Use | Access | Licence | Pin |
|---|---|---|---|---|
| `open-llm-leaderboard/contents` | panel candidates, reported raw scores (adapter gate) | ungated | not stated | `9c09a7cae43334062a82cb164f2ef255013dafa2` |
| `open-llm-leaderboard/<org>__<model>-details` | per-item responses (`leaderboard_mmlu_pro`, `leaderboard_gpqa_extended`, three BBH subtasks), parquet conversion of the `latest` split | gated (auto) | not stated ("More Information Needed") | commit SHA recorded per repo at fetch time |
| `TIGER-Lab/MMLU-Pro` | item table, keys at every revision | ungated | MIT | 19 revisions in `configs/*.toml` (2024-06-11 to 2026-01-19) |
| `edinburgh-dawg/mmlu-redux-2.0` | human labels (57 subjects x 100) | ungated | CC-BY-4.0 | `372ea425445d51e1ba1188c56e5e893f8138621f` (parquet ref `db29904159aa684e8c75888bf0123d8fcf38131a`) |
| `madrylab/platinum-bench` | human labels for 3 BBH subtasks and MMLU HS maths | ungated | CC-BY-SA-4.0 | `51920a33bfb4620c789729ace14141e87a14969b` (parquet ref `167104616cd070bc8ae58b97404d714c7d225965`) |

Redistribution rules: per-model responses are never committed (licence not stated). Committed
results contain item ids, our scores and aggregates. MMLU-Pro item text may appear in the
suspect list (MIT). GPQA appears only as record ids and hashes (the GPQA authors ask that
questions not be published in plain text). Platinum-derived per-item labels are not
committed (ShareAlike); only aggregate metrics are. Redux-derived labels are committed with
attribution (CC-BY-4.0) only if needed for reproducibility.

### 4.2 Facts checked at scaffold time (2026-10-24)

- MMLU-Pro test: 12,032 items; `src` = `ori_mmlu` 6,810, `stemez` 4,083, `theoremQA` 598,
  `scibench` 541. Key-*text* changes between consecutive upload revisions (19 Jul 2024 ->
  19 Jan 2026): 25 (Aug 2024), 5 (Sep 2024), 1 (Nov 2024), 20 (Apr 2025), 2 (Aug 2025),
  1 (Oct 2025), 0 (Jan 2026); 54 from `80ab9608` to `24ac2da5` (44 of them `ori_mmlu`;
  24 math, 21 health). Index-only changes are larger (171 in Apr 2025) because options were
  edited, so the oracle must compare key text, not indices.
- The leaderboard task (`lm-evaluation-harness`, `leaderboard/mmlu_pro`) loads
  `TIGER-Lab/MMLU-Pro` without a pinned revision, 5-shot, letters A-J, metric `acc`. Runs
  from June 2024 to March 2025 may therefore have used different keys. The response (argmax
  over letter log-likelihoods) does not depend on the key, so we rescore every model against
  one canonical key per item, mapping options by text where they changed.
- MMLU-Redux 2.0: 5,700 items; `ok` 5,330, `bad_question_clarity` 132, `wrong_groundtruth`
  106, `multiple_correct_answers` 39, `no_correct_answer` 36, `expert` 32,
  `bad_options_clarity` 25.
- Join of MMLU-Pro `ori_mmlu-<subject>` to Redux by subject and normalised question text
  (revision `014c8cf5`): 2,893 labelled MMLU-Pro items in 56 subjects with the same key text
  in both (the scaffold first counted 2,902 / 2,879 per Redux row; corrected in section 14,
  item 2); among those `wrong_groundtruth` 56, `multiple_correct_answers` 24,
  `no_correct_answer` 17 (broad positives 97). For the 3 items Redux marks wrong whose
  MMLU-Pro key differs from MMLU's, MMLU-Pro adopted Redux's correction.
- Platinum: `bbh_logical_deduction_three_objects` 200 (0 rejected), `bbh_navigate` 200
  (0 rejected), `bbh_object_counting` 200 (10 rejected), `mmlu_math` 270 (3 rejected).
- Leaderboard contents: 4,576 evaluations; 470 from official providers; MMLU-Pro raw score
  median 0.34, maximum 0.73; GPQA raw maximum about 0.42.
- Byte-range projection works through the Hub (two small columns of the 4 MB MMLU-Pro
  parquet: 72 KB of column chunks).

### 4.3 Fetching, caching and checksums (`hub`, M2)

- All HTTP with a 60 s timeout, 3 retries with exponential backoff (2, 4, 8 s), a fixed
  user agent, and the token only in the `Authorization` header (never logged).
- Small files (MMLU-Pro revisions, Redux, Platinum, contents) are downloaded whole and
  verified against the LFS sha256 from the tree API; mismatch is an error.
- Details parquet: `RangeFile` + `pyarrow.parquet.ParquetFile.read(columns=...)`; bytes
  fetched are counted and the read aborts above `per_model_budget_mb`. The cache stores the
  derived arrays only (`data/units/<task>/<model>.npz`: item ids, per-option
  log-likelihoods as float32, the key embedded in the doc), with sha256 and provenance
  (repo, commit SHA, file path, remote size, remote LFS oid if visible, bytes fetched, time,
  library versions) appended to `data/manifest.jsonl`.
- `data/`, `cache/` and `runs/` are gitignored.

## 5. Model and identifiability

### 5.1 Notation and the mis-keyed latent class model

Items i = 1..I with K_i options and key k_i; models j = 1..J with chosen option r_ij
(missing allowed). Each item has a latent true option t_i.

    Prior:      P(t_i = k_i) = 1 - eps_g(i),   P(t_i = c) = eps_g(i) / (K_i - 1),  c != k_i
    Truth:      P(r_ij = c | t_i = c) = p_ij(c) = g_i + (1 - g_i) * sigmoid(a_ic (theta_j - b_ic))
                with g_i = 1 / K_i and a_ic >= 0
    Distractor: P(r_ij = o | t_i = c) = (1 - p_ij(c)) * pi_ic(o),  o != c,
                pi_ic(o) = (n_io + alpha) / (n_i - n_ic + alpha (K_i - 1)),  alpha = 0.5

n_io is the number of models choosing o on item i. This is the GLAD/Dawid-Skene structure
with the benchmark key as a prior rather than an annotator, a guessing floor, and an
item-specific, ability-independent distractor distribution.

### 5.2 What responses cannot identify (shared blind spots)

Take a mis-keyed item: truth c, key k, truth curve p(theta) increasing, distractor shares
pi. Define an alternative with truth = k:

    P'(r = k | theta) = (1 - p(theta)) * pi(k)                 (decreasing in theta)
    P'(r = c | theta) = p(theta)                                (an ability-dependent distractor)
    P'(r = o | theta) = (1 - p(theta)) * pi(o),  o not in {c, k}

Both produce exactly the same response distribution for every model. The alternative is a
correctly keyed item on which stronger models are increasingly drawn to one specific wrong
option, i.e. a shared misconception, a contaminated wrong answer, or an inverse-scaling
item. So:

1. The model separates the two only through two assumptions: (A1) the truth curve is
   non-decreasing in ability (`a >= 0`), and (A2) distractor choice does not depend on
   ability. A shared blind spot is precisely a violation of A1 and A2 on that item. No
   amount of response data from the same kind of panel removes it.
2. Without A1-A2 the response data are consistent with eps = 0 (every flag is a trick
   item) and with every flag being a mis-key, so `E_model` is not identified. A calibrated
   rate needs external labels, which is why the headline uses the human-labelled random
   sample (section 6.7) and why `E_model` is reported only "under A1-A2".
3. Correlated models (fine-tunes and merges of the same base) make blind spots shared and
   inflate the apparent evidence. We scale log-likelihoods by tau = J / J_eff (section 6.5),
   report J_eff, cluster models into families, and report leave-one-family-out stability.

`identifiability.equivalent_truth_k` builds the alternative parametrisation; a unit test
checks the two log-likelihoods agree to 1e-12 on random data (M3).

### 5.3 Where the panel has no information (competence region)

A mis-key is only visible when the panel knows the true answer. For each item, with c* the
most probable truth, competence `kappa_i` = mean over the top-decile models (by theta) of
`p_ij(c*)`. Items with `kappa_i < kappa_0` (prespecified 0.5) carry almost no information
about their key. Reported:

- the share of items in the competence region S per benchmark and group;
- recall against human labels inside and outside S (expected: near zero outside);
- a transfer check: Fisher exact test of the Redux error rate inside vs outside S on the
  labeled overlap (if they differ, extrapolating `E_model` from S to all items is unsafe);
- worst-case bounds for `E_model`: `[eps_S |S| / I, (eps_S |S| + |S^c|) / I]`. These will
  usually be wide; that is the honest statement for weak panels (GPQA).

### 5.4 Estimation

1. Abilities: penalised joint maximum likelihood of the section 5.1 truth curve on key
   correctness (theta ~ N(0,1), log a ~ N(0, 0.5^2), b ~ N(0, 2^2)); standardise theta.
   Key errors (a few percent) barely move theta; an outer loop refits theta on soft truth
   `y_ij = sum_c w_i(c) 1[r_ij = c]` (2 outer iterations).
2. Candidate log-likelihood `l_i(c)`: MAP (a >= 0, b) for the binary outcome 1[r_ij = c]
   plus the distractor multinomial term, for every option c (about 120k small fits for
   MMLU-Pro, solved as one vectorised projected-Newton problem).
3. EM: `w_i(c) ∝ prior_i(c; eps_g) * exp(l_i(c) / tau)`; `eps_g` = posterior mean under a
   Beta(alpha_0, beta_0) prior whose parameters are set by method of moments across groups
   (full pooling if the between-group variance estimate is not positive). Converge when
   max |delta eps| < 1e-6 or 500 iterations.
4. Outputs: `s_i = 1 - w_i(k_i)`, proposed option `argmax_{c != k_i} w_i(c)`, `kappa_i`,
   `E_model(g) = mean_{i in g} s_i`.

## 6. Module specifications

### 6.1 `checkpoint` (M1)

`UnitStore(root, config_hash)`: `done(unit)`, `save(unit, arrays, meta)` (write to a temp
file, fsync, `os.replace`), `load(unit)`, `manifest.json` (config hash, package version, git
SHA, start history), `progress.log` (one line per finished or failed unit with timestamp,
seconds, done/total, ETA). `run_units(store, units, workers<=3, retries)`: skips done units,
`ProcessPoolExecutor` with an initializer that sets BLAS threads to 1 (threadpoolctl and
`OMP_NUM_THREADS` etc.), retries failed units, records `denied` units without retrying,
returns statuses. A run directory refuses to resume when the config hash differs.
`scripts/run_study.py` wraps `keyaudit run` in a resume-on-failure loop (3 attempts) and
logs to `runs/<name>/progress.log`.

### 6.2 `sources.leaderboard` (M2)

- `select_panel` (section 8.2) from the pinned contents table; writes `configs/panel_v1.csv`
  (model name, details repo, MMLU-Pro raw, family hint, priority rank). Denied models are
  replaced by the next in priority order.
- `fetch_responses(model, task)`: list `refs/convert/parquet` for config
  `<org>__<model>__<task>` split `latest`; read `doc_id`, the doc's id field
  (`question_id` for MMLU-Pro, `Record ID` for GPQA, the input hash for BBH), the doc's key
  and options, and `filtered_resps`; parse per-option log-likelihoods. Exact column names
  are confirmed on the first readable repository and fixed in a test fixture.
- Response: argmax over options of log-likelihood (MMLU-Pro `acc`), or of log-likelihood
  divided by the continuation's byte length where the metric is `acc_norm` (GPQA choices
  "(A)".."(D)" have equal length, so it equals `acc`; BBH needs the normalisation).
- Adapter gate: our recomputed per-model correct count equals the leaderboard's reported raw
  score x item count exactly (contents table and the repo's `results_*.json`). A model that
  fails is excluded and listed; more than 5% failures stops the study.

### 6.3 `matrix` (M2)

`ResponseMatrix` (items x models int8, -1 missing). Canonical key per item: the
leaderboard-era revision that the majority of panel runs used (fingerprinted by matching
each run's embedded keys to the 19 pinned revisions); items whose options differ across runs
are mapped by option text or dropped (count reported). Near-duplicate models (response
agreement > 0.95 on MMLU-Pro) are dropped, keeping the higher-priority one. Families:
average-linkage hierarchical clustering of 1 - agreement, cut at 0.5 (prespecified).

### 6.4 `irt` (M3)

Vectorised penalised JML and the constrained item fit of section 5.4 (projected Newton with
a in [0, 8], step halving, at most 50 iterations, tolerance 1e-8 on the gradient norm).

### 6.5 `keymix` (M3)

`fit_keymix(matrix, tau=None, alpha_pi=0.5, outer_iter=2)` implementing sections 5.1 and
5.4. `tau = J / J_eff` where J_eff is the participation ratio of the eigenvalues of the
model-by-model agreement matrix (`identifiability.effective_panel_size`). Groups: MMLU-Pro
`category` (14) by default; `source` (4) and Redux `subject` (56) for secondary tables.

### 6.6 `baselines` (M3)

Five suspect scores (higher = more suspect): top-decile plurality disagreement; minus
item-total point-biserial correlation; minus scalability Z_j =
`sum_{k != j} Cov(X_j, X_k) sqrt(N - 1) / sqrt(sum_{k != j} V(X_j) V(X_k))` (Truong et al.
2025); binary sign likelihood ratio `l(a <= 0) - l(a >= 0)` of the section 5.1 curve on key
correctness (in the spirit of Land and Bikel 2026; not their 4PL); one-coin Dawid-Skene
posterior with models as annotators.

### 6.7 `estimate` (M4)

1. **Label-calibrated rate (headline)**, population P = MMLU-Pro `ori_mmlu` items (6,810),
   labeled sample L = the Redux overlap. Redux sampled 100 items per subject uniformly from
   the MMLU test set and MMLU-Pro filtered MMLU items without regard to Redux, so within each
   subject L is a simple random sample of P (the sampling fraction differs by subject; hence
   stratification by subject, collapsing subjects with fewer than 10 labeled items into
   their MMLU-Pro category). Stratified difference estimator (survey-sampling difference
   estimator = prediction-powered inference for a mean, with finite-population correction):

       theta_s = mean_{P_s} f_i + mean_{L_s}(Y_i - f_i)
       var_s   = (1 - n_s/N_s) S^2_{Y-f,s} / n_s
       theta   = sum_s (N_s/N) theta_s,   var = sum_s (N_s/N)^2 var_s

   with `f_i` the cross-fitted Platt calibration of `s_i` to Y (5 folds: the f_i of a labeled
   item never uses its own label; unlabeled items get the average of the 5 fold maps, see
   section 14, deviation 21). Also reported: the classical stratified estimate (f = 0),
   the variance ratio, and a stratified bootstrap interval as a check on the normal
   approximation (positives are rare). Implemented in `stats.stratified_difference_estimate`.
2. **Model-based rate** `E_model(g)` with pigeonhole bootstrap intervals (Owen 2007): each
   replicate resamples items and models independently with replacement and refits; 95%
   percentile intervals; B from the config.
3. **Transfer estimate** for `stemez`, `theoremQA`, `scibench` (no labels): mean calibrated
   score; the interval combines the pigeonhole bootstrap with a bootstrap of the calibration
   fit over L. Reported as an extrapolation that assumes the calibration transfers across
   sources.
4. **Bounds**: section 5.3.

### 6.8 Multiplicity and intervals (`stats`, implemented at scaffold)

Clopper-Pearson for proportions (precision@k, recall at the operating point); Wilson as a
cross-check; Holm across the five baseline comparisons and across the 14 per-category
"differs from pooled" tests; Bayesian FDR q-values (Mueller et al. 2004) from `1 - s_i` for
the suspect list, operating point q <= 0.10 (model-based FDR, checked empirically against
labels).

## 7. Validation and test strategy

### 7.1 Reference values (tests/test_stats.py; recomputed by scripts/reference_values.py)

| Quantity | Value |
|---|---|
| Clopper-Pearson 95%, 106/5700 | [0.015250, 0.022448] |
| Clopper-Pearson 95%, 57/2900 | [0.014920, 0.025391] |
| Clopper-Pearson 95%, 5/100 | [0.016432, 0.112835] |
| Clopper-Pearson 95%, 0/50 | [0, 0.071122] |
| Clopper-Pearson 95%, 10/200 | [0.024234, 0.090028] |
| Wilson 95%, 106/5700 | [0.015400, 0.022442] |
| Holm / BH for p = (0.01, 0.04, 0.03, 0.005) | (0.03, 0.06, 0.06, 0.02) / (0.02, 0.04, 0.04, 0.02) |
| Difference estimator, f = (0.1, 0.2, 0.9, 0.4, 0, 0.5), labels 0/1/0 on items 1, 3, 4 | 0.216667, se 0.102740 |
| Data facts (section 4.2): Redux counts, join counts, revision key-text diffs, Platinum counts | fixed in M2 tests against the pinned revisions |

### 7.2 Oracles for the model (M3)

- Exact observational equivalence of section 5.2 (log-likelihoods equal to 1e-12).
- Synthetic recovery: simulate J = 100 models, I = 3,000 items, K = 10, eps = 0.03, abilities
  N(0,1), item parameters from the priors, distractor shares Dirichlet(1); check eps recovery
  (|bias| < 0.005 over 20 seeds) and planted-item AP > 0.8 inside S. Slow test: pigeonhole
  bootstrap coverage of eps over 100 synthetic datasets is at least 0.90 for a nominal 0.95
  (lower 99.9% Clopper-Pearson bound reported).
- Misspecification check (slow): synthetic trick items that violate A1-A2 are flagged at the
  same rate as true mis-keys (demonstrates section 5.2, not a bug).
- Baselines on a hand-sized matrix against direct numpy formulas.

### 7.3 Validation against humans (M4)

- **MMLU-Redux 2.0 overlap** (2,893 items, 56 narrow and 97 broad positives): AP and
  ROC AUC with stratified bootstrap intervals (2,000 resamples); precision@k and recall@k,
  k in {25, 50, 100, 200}; precision and recall at q <= 0.10 with Clopper-Pearson intervals;
  paired bootstrap AP difference vs each baseline, Holm-adjusted. Breakdowns: recall by
  Redux error type, inside vs outside S, and precision split by whether the proposed option
  is one of the original four MMLU options or a distractor added by MMLU-Pro (an added
  distractor that is actually correct is an MMLU-Pro error Redux could not see, so precision
  against Redux understates true precision there).
- **Maintainer corrections**: items whose key text changed after the canonical revision
  (up to 54). Recall only (they are not a random sample, and unfixed errors exist).
- **Platinum**: BBH (600 items, 10 rejected) and MMLU HS maths overlap (3 rejected):
  false-positive rate among Platinum-clean items with Clopper-Pearson intervals, and
  detection of the few rejected items. Too few positives for precision/recall claims.
- **Semi-synthetic planted keys**: on Redux-`ok` items, swap 2% of keys to (a) a random
  distractor and (b) the most chosen distractor; refit; recall vs competence and the shift
  in estimated rate vs the planted rate. 20 replicates (study), 100 (full).
- **Stability**: leave-one-family-out refits for the largest 5 (10) families; each suspect
  item reports the share of refits in which it stays at q <= 0.10.

### 7.4 Test rules

Tests run offline, write only under `tmp_path`, use fixed seeds and pass on Windows.
Network-dependent checks live in `scripts/` and are run by hand. Statistical assertions use
explicit exact intervals or seeded tolerances stated in the test. Simulations longer than a
few seconds are `@pytest.mark.slow` (`uv run pytest --runslow`).

## 8. Study design

### 8.1 What is run

Benchmarks: MMLU-Pro (12,032), GPQA extended (546), BBH logical_deduction_three_objects,
navigate, object_counting (250 each). Seed 20261024; every stochastic unit derives its seed
from `(seed, crc32(unit_id))`.

### 8.2 Panel rule (prespecified)

From the pinned contents table: available on the Hub, not flagged, not a merge, type not
"multimodal" or "other", MMLU-Pro raw >= 0.15 (chance is 0.10). Family key = the `Base
Model` field (or the model itself); keep at most 2 per family, preferring official
providers, then higher MMLU-Pro. Strong models carry the signal, so take the top quintile
(by MMLU-Pro raw) up to `top_quintile_cap`, then `per_other_quintile` per remaining
quintile by seeded sampling. The candidate list is ordered by priority; denied repositories
are skipped and replaced from the same quintile. The realised panel (names only) is committed
as `configs/panel_v1.csv` with the reason for every exclusion.

### 8.3 Sizes and run time

| Config | Models | Bootstrap B | LOFO | Planted reps | Projected wall (3 workers) |
|---|---|---|---|---|---|
| `configs/study.toml` | 100 (min 60) | 200 | 5 | 20 | 20-30 min incl. ~10-15 min fetch |
| `configs/study_full.toml` | 300 (min 150) | 1000 | 10 | 100 | 2-3 h, ~1-1.5 GB |

These are projections; M2 measures bytes and seconds per fetch unit and M5 the fit and
bootstrap times, and the table is updated with measurements. If the default study is
projected over 45 minutes, reduce B before reducing the panel.

### 8.4 Commands (after M5)

```sh
uv run keyaudit fetch  --config configs/study.toml      # needs HF_TOKEN with accepted gates
uv run keyaudit run    --config configs/study.toml --workers 3
uv run keyaudit status --config configs/study.toml
uv run keyaudit report --config configs/study.toml      # results/study-v1/ (REPORT.md by hand)
uv run python scripts/run_study.py --config configs/study_full.toml   # full size, with retries
```

## 9. Reporting rules

- Report what was run: panel size and list, denied and excluded models with reasons, item
  counts after every filter, revisions, seeds, library versions, wall times.
- Every rate has an interval and names its estimand (`E_label` narrow/broad or `E_model`)
  and its population. Model-based numbers always carry "under A1-A2".
- Extrapolations (transfer estimates, GPQA) are labelled as such in tables and text.
- Multiple comparisons: Holm for baseline comparisons and per-category tests; q-values for
  the suspect list. No unadjusted "significant" category claims.
- A flagged item is "suspect", never "wrong", unless a human label says so. A claimed
  third-party data error in the report cites the item id, the evidence and a minimal check
  that it is not an adapter mistake (wrong revision, option mapping, normalisation).

## 10. Architecture

```
hub.py               HTTP: listing, pinned revisions, RangeFile, verified downloads, manifest
sources/leaderboard  panel rule, details adapter, adapter gate
sources/benchmarks   MMLU-Pro revisions, key-text diffs, revision fingerprinting
sources/labels       Redux and Platinum joins, label tiers
matrix.py            ResponseMatrix, canonical keys, dedupe, families
irt.py               truth curve, penalised JML, vectorised constrained item fits
keymix.py            mis-keyed latent class EM (sections 5.1, 5.4)
baselines.py         five comparison scores
identifiability.py   equivalence construction, J_eff, competence region, bounds
estimate.py          pigeonhole bootstrap, calibration, difference estimator, transfer
validate.py          ranking metrics, operating point, planted keys, LOFO stability
stats.py             intervals, Holm/BH, Bayesian FDR, difference estimator   [implemented]
checkpoint.py        UnitStore, run_units (<= 3 processes, BLAS threads 1)
study/config.py      TOML schema, validation, config hash                     [implemented]
study/runner.py      stages as units: fetch, build, fit, boot, lofo, planted, validate
study/report.py      results/<study>/ estimates.json, validation.json, suspects.csv, tables.md
cli.py               keyaudit fetch | run | status | report | check-config | versions
```

Data flow: contents -> panel -> fetch units -> response matrices (+ canonical keys) ->
keymix fit -> scores -> {bootstrap, LOFO, planted} units -> estimates and validation ->
report.

## 11. Milestones and done-checks

| Milestone | Scope | Done when |
|---|---|---|
| M0 (done) | scaffold, design, `stats`, `config`, CI | `uv sync`, `pytest`, `ruff check`, `ruff format --check` pass |
| M1 (~30 min) | `checkpoint`, `scripts/run_study.py` | tests: atomic save, skip on rerun, config-hash refusal, `denied` status, <= 3 processes, BLAS threads 1 in workers, retry on a flaky unit |
| M2 (~70 min) | `hub`, `sources.*`, `matrix`, panel | offline tests on recorded fixtures (a few rows, no question text for GPQA); section 4.2 facts reproduced by a script against pinned revisions; on real data: adapter gate passes for >= 95% of fetched models, bytes per model <= budget, revision fingerprint table, Redux join counts as in 4.2. **Blocked on a token with accepted gates**; everything else proceeds on fixtures |
| M3 (~60 min) | `irt`, `keymix`, `baselines`, `identifiability` | equivalence oracle to 1e-12; synthetic recovery and AP checks; slow coverage test; baselines vs hand formulas |
| M4 (~50 min) | `estimate`, `validate` | difference estimator reference values; cross-fitting leaves own label out (test); bootstrap determinism by seed; planted-key and LOFO units resumable |
| M5 (~50 min) | `study.runner`, `study.report`, CLI, first real run, REPORT.md | `keyaudit run` on `configs/study.toml` finishes in < 45 min wall and resumes after a kill; report regenerated from units; REPORT.md written by hand with sizes, seeds, versions, intervals, limits |
| M6 (optional) | GPQA expert-validator check; soft responses; EEE frontier panel (needs sign-off) | only after M5 |

## 12. Risks

1. **Gated data.** No token, or too few accepted repositories, blocks every real-data
   milestone. Mitigation: the code and tests run on fixtures; `fetch` reports exactly which
   repositories need acceptance; the study refuses to run below `min_models`.
2. **Projection fails** (row groups dominated by full-text input columns, or a server that ignores
   Range): bytes per model far above the budget. Mitigation: the budget aborts the read;
   fall back to fewer models (raw parquet is about 80 MB per model, so at most ~35 models in
   3 GB).
3. **Key revisions differ across runs.** Handled by rescoring against one canonical key;
   items with incompatible option lists are dropped and counted.
4. **Weak panel.** 2024 open-weight models top out at 0.73 on MMLU-Pro and 0.42 on GPQA; many
   items lie outside the competence region, so recall is bounded. Reported, not hidden.
5. **Correlated panel.** Many leaderboard models share a base; tau = J/J_eff, dedupe, family
   caps and LOFO mitigate but do not remove shared blind spots (section 5.2).
6. **Label noise and label mismatch.** Redux labels the 4-option MMLU item; MMLU-Pro added six
   distractors. Precision against Redux can understate true precision (section 7.3).
7. **Few positives.** About 56 narrow positives give recall intervals of roughly +/- 13
   points; Platinum's 13 positives support no precision/recall claim.
8. **Profile-likelihood overconfidence.** Raw posteriors can be near 0/1; tau and the
   calibration step address the rate estimate; rankings are unaffected.

## 13. References

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., Zrnic, T. (2023).
  Prediction-powered inference. *Science* 382(6671), 669-674.
- Angelopoulos, A. N., Duchi, J. C., Zrnic, T. (2023). PPI++: Efficient prediction-powered
  inference. arXiv:2311.01453.
- Dawid, A. P., Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates
  using the EM algorithm. *JRSS C* 28(1), 20-28.
- Fourrier, C., et al. (2024). Open LLM Leaderboard v2 (archived 2025). Hugging Face.
- Gema, A. P., et al. (2024). Are we done with MMLU? arXiv:2406.04127 (NAACL 2025).
- Land, S., Bikel, D. M. (2026). Auditing LLM benchmarks with item response theory.
  arXiv:2605.30504 (EMNLP 2026).
- Manski, C. F. (2003). *Partial Identification of Probability Distributions*. Springer.
- Mueller, P., Parmigiani, G., Robert, C., Rousseau, J. (2004). Optimal sample size for
  multiple testing: the case of gene expression microarrays. *JASA* 99(468), 990-1001.
- Nakamura, K., Tan, J. L., Yean, G. (2026). Observational equivalence of LLM and human
  annotation. arXiv:2609.22133.
- Northcutt, C. G., Athalye, A., Mueller, J. (2021). Pervasive label errors in test sets
  destabilize machine learning benchmarks. arXiv:2103.14749 (NeurIPS 2021 D&B).
- Owen, A. B. (2007). The pigeonhole bootstrap. *Annals of Applied Statistics* 1(2), 386-411.
- Polo, F. M., et al. (2024). tinyBenchmarks: evaluating LLMs with fewer examples.
  arXiv:2402.14992.
- Rein, D., et al. (2023). GPQA: a graduate-level Google-proof Q&A benchmark.
  arXiv:2311.12022.
- Saerndal, C.-E., Swensson, B., Wretman, J. (1992). *Model Assisted Survey Sampling*.
  Springer.
- Truong, S., et al. (2025). Fantastic bugs and where to find them in AI benchmarks.
  arXiv:2511.16842 (NeurIPS 2025 D&B).
- Vendrow, J., Vendrow, E., Beery, S., Madry, A. (2025). Do large language model benchmarks
  test reliability? arXiv:2502.03461.
- Wang, Y., et al. (2024). MMLU-Pro: a more robust and challenging multi-task language
  understanding benchmark. arXiv:2406.01574 (NeurIPS 2024 D&B).
- Whitehill, J., Ruvolo, P., Wu, T., Bergsma, J., Movellan, J. (2009). Whose vote should
  count more: optimal integration of labels from labelers of unknown expertise. NeurIPS 22.
- Zrnic, T., Candes, E. J. (2024). Cross-prediction-powered inference. *PNAS* 121(15).

Software consulted (not dependencies): `crowd-kit` 1.4.2 (Apache-2.0; pulls in
`transformers`), `ppi-python` 0.2.3 (MIT; pulls in `numba`), `py-irt` 0.7.1 (MIT; Python
< 3.12), `girth` 0.8.0 (MIT), `cleanlab` 2.9.0 (Apache-2.0), `fantastic-bugs` code
(GitHub `sangttruong/fantastic-bugs`). None is needed at run time; any may serve as an
offline oracle through `uv run --with <pkg>==<version>` in a fixture script.

## 14. Implementation notes and deviations

Logged in milestone order. "Measured" means checked by `scripts/check_data_facts.py`
against the pinned revisions on 2026-10-24 (output: `results/data-facts.json`, locked by
`tests/test_data_facts.py`).

**M2 (data layer)**

1. Section 4.2 facts reproduced exactly: 12,032 items and source counts; key-text changes
   per upload 0, 0, 25, 5, 1, 20, 1, 1, 1, 0 from `80ab9608` to `24ac2da5` (54 in total,
   44 `ori_mmlu`, 24 math, 21 health; 171 index-only changes in April 2025); Redux counts;
   Platinum counts; contents (4,576 evaluations, 470 official, MMLU-Pro raw median 0.3447,
   maximum 0.7303).
2. Corrections to section 4.2: the GPQA raw maximum in the contents table is 0.4706, not
   about 0.42. (It is pooled over the three GPQA subsets, so it is not the extended-set
   accuracy.) The Redux join counts per MMLU-Pro item: 2,916 MMLU-Pro items match a Redux
   row, 9 are dropped because the matching Redux rows disagree, 23 have a different key text,
   leaving **2,893** labelled items (the scaffold's 2,902 / 2,879 counted differently; 70
   groups of MMLU-Pro items share a question text with another item in the same subject).
   The positive counts are unchanged: 56 `wrong_groundtruth`, 24 `multiple_correct_answers`,
   17 `no_correct_answer` (97 broad).
3. Redux is read from the arrow files of the pinned main revision (`372ea425`, LFS sha256
   verified) rather than from the parquet conversion branch; the rows are the same 5,700.
4. The details adapter (`sources.leaderboard.parse_samples`, `fetch_responses`) follows
   the lm-evaluation-harness per-sample layout and the leaderboard task definitions as
   understood at the time of writing (MMLU-Pro letters A-J scored by `acc`; GPQA "(A)"-"(D)";
   BBH choice lists with `acc_norm` by byte length including the leading space). **It has not
   been run on a real details repository**: on 2026-10-24 the available token could list
   repository files but every content request returned 403 (conditions not accepted). The
   adapter gate is the safeguard; `scripts/probe_details.py` prints the real schema once a
   repository is readable. A response letter beyond an item's options (MMLU-Pro items with
   fewer than 10 options) is recorded as invalid (missing), not as a choice.
5. Fetch units store option-text hashes (crc32 of the normalised text) instead of option
   text, so options are mapped across key revisions without keeping any item text in the run
   directory. GPQA record ids and BBH inputs are stored as truncated sha256 ids.
6. Panel rule details fixed in code: one evaluation per model (the best MMLU-Pro raw score,
   since details repositories are per model); "merge" types and the `Merged` flag both
   exclude; quintiles are computed after the family cap; reserves follow the selected models
   in priority order within their quintile.

**M3 (model and baselines)**

7. `tau` is estimated from the correlation matrix of IRT residuals of key-correctness
   (a Yen Q3-type matrix) instead of the raw response-agreement matrix: models that answer
   independently given the items still agree on most items because they share the truth, so
   the participation ratio of raw agreement would report J_eff of about 5-10 for an
   independent panel and flatten every likelihood ratio. Each off-diagonal squared
   correlation is reduced by its null expectation `1/n_ij` before the ratio
   `J^2 / sum R_ij^2`. Measured: tau = 1.02 for an independent synthetic panel (J = 100,
   I = 3,000) and 1.5 for six families with a shared item effect of sd 2.
8. Constrained candidate fits use the prior a ~ N(1, 2^2) restricted to [0, 8] (a
   log-normal prior cannot reach the a = 0 bound) and b ~ N(0, 2^2). They start from the best
   point of a fixed (a, b) grid (31 x 49 points, two matrix products for all rows) and take at
   most 3 projected Fisher-scoring steps with backtracking. Tested: within 0.1 nats per row of
   a 300-iteration solution. JML stops after 30 alternations or when no parameter moves by
   1e-3; ability estimates no longer change after about 10 (the JML scale drifts slowly).
   A full fit takes about 15 s for 3,000 items x 100 models on one core.
9. Candidates are the key and every option chosen by at least one model (zero weight
   otherwise).
10. `sign_lrt` fits a 2PL curve (no guessing floor). With the 1/K floor a mis-keyed item's
    key-correctness, which falls towards 0, cannot be fitted by either sign and the baseline
    is uninformative (AP 0.22-0.27 on synthetic panels).
11. **Competence is endogenous.** kappa_i is computed for the most probable truth c*. When a
    mis-key is not detected, c* is the wrong key and kappa_i is low, so mis-keys concentrate
    outside the competence region S. Measured on synthetic panels (J = 60, I = 1,500,
    eps = 0.04): true rate 0.026-0.030 inside S (estimated 0.025-0.029) but 0.09-0.14
    outside (estimated 0.04-0.05). The overall `E_model` is therefore biased low; the
    recovery check of section 7.2 is applied inside S, and the transfer check of section 5.3
    (Redux rate inside vs outside S) is essential before extrapolating.
12. The one-coin Dawid-Skene baseline uses the key as its prior (P(t = key) = 1 - eps), like
    the main model, so the comparison isolates item difficulty and ability.

**M4 (estimation and validation)**

13. Pigeonhole replicates keep the full-sample tau: resampled duplicate models would
    otherwise look like a correlated panel and inflate tau.
14. Platt calibration regresses the label on logit(s_i), with s_i clipped to [1e-6, 1 - 1e-6]
    and a ridge of 1e-2 on both coefficients, fitted by Newton steps with backtracking (plain
    Newton diverged with 4 positives in a simulation). Bootstrap p-values are floored at 1/B.

**M5 (runner, report, first runs)**

15. Stages are `fetch, labels, build, fit, boot, lofo, planted, validate`. The new `labels`
    stage computes label-only estimates that need no panel: the classical stratified
    estimator (f = 0) of `E_label` for the MMLU-derived items, per-category Clopper-Pearson
    intervals, and Fisher exact tests of each category against the rest, Holm-adjusted. These
    tests treat the labeled items as a simple random sample and ignore the subject strata.
    Without a panel to fingerprint the runs, the canonical revision falls back to `014c8cf5`,
    the revision current when the leaderboard stopped evaluating.
16. `fetch` tries at most `2 * n_models` candidates. A denied model is replaced by the next
    reserve in its quintile. HTTP 400 (no parquet conversion for that config) and 404
    (repository missing) count as failed, not denied.
17. **First real run (2026-10-24, `configs/study.toml`)**: 200 candidates tried, 194 denied
    (403: conditions not accepted), 6 failed (5 x HTTP 400, 1 x 404), 0 readable. `build`
    stopped with exit code 3, so no model-based number exists. Wall time: 12.5 minutes,
    mostly about 3.5 s per denied request, plus 39 s for the labels unit.
18. The simulation study (`keyaudit simulate`, `results/simulation-v1/`) was added to test
    the method where the truth is known, because the real panel was unavailable. Its sizes
    are fixed in `keyaudit.study.simulation.PROFILES`.
19. GPQA and BBH matrices take items and keys from the details. Items whose embedded key
    differs between runs are dropped. Models are gated against the task accuracy in their
    `results_*.json`.

20. **Bootstrap over items, panel fixed (after the M5 runs).** In a pigeonhole replicate a
    model drawn twice counts as two independent confirmations of the same answer, so the
    refitted rate is inflated (about 2-3x on a 1000 x 60 panel; regression test
    `test_item_bootstrap_not_inflated_but_pigeonhole_is`). The default is now items only,
    conditional on the panel (`resample_models=False`); intervals no longer reflect panel
    sampling. The coverage units in `results/simulation-v1/` were computed before this change
    with the pigeonhole bootstrap: their median overall interval is about 22 points wide and
    does not contain its own point estimate in 19 of 30 panels, so the 29/30 "coverage" there
    is not evidence of a working interval. Coverage units now record their mode
    (`bootstrap`), and the summary refuses to mix modes. Item-bootstrap coverage was measured
    afterwards (deviation 22).
21. **One score function in the calibrated estimator (audit finding 5).** Unlabeled items
    used a Platt map fitted on all labels, labeled items cross-fitted maps, so the population
    mean and the labeled residuals used different functions. Unlabeled items now get the
    average of the fold maps (cross-prediction-powered inference). Effect: with 50 labeled
    items out of 1,000 (about 2.5 positives) the old mix was biased +0.14 [+0.08, +0.20]
    points over 6,000 seeded draws, the new form +0.03 [-0.02, +0.08]
    (`test_calibrated_estimator_unbiased_with_few_labels`, slow). With 25% labels the effect
    is negligible: on 22 `tricks` panels with 100 label draws each, old -0.00 [-0.02, +0.02],
    new +0.00 [-0.02, +0.02] points. The audit's -0.30 [-0.57, -0.03] in simulation-v1
    reproduces exactly from those 10 panels' label draws and is unchanged (-0.30) under the
    new form, so it was sampling noise in the label draws, not the mixed scores.
    simulation-v1 has not been rerun with the new form.
22. **Item-bootstrap coverage and the overall interval (audit finding 6).** Profile
    `coverage` (72 panels of 1,000 items x 60 independent models, eps = 0.03, B = 20), run
    with seeds 20261024 (`results/items-cov-v1`) and 20261025 (`results/items-cov-v2`).
    Percentile interval for the overall rate: 34/72 (CP 95% 0.35-0.59) and 35/72
    (0.37-0.61). The overall rate is -0.97 and -0.94 points low because mis-keys outside the
    competence region are invisible, and the replicates share that bias (replicate mean minus
    estimate about 0.0), so normal, basic and bias-corrected percentile forms computed from
    the same replicates do not repair it (24/72 to 46/72). Inside the region (bias -0.09
    points) the percentile interval covered 72/72 in both runs (CP 95% 0.95-1.00; it may be
    conservative). Fix: the overall rate is reported with a region-bounded interval,
    `[lo_S * share, hi_S * share + (1 - share)]` (`estimate.region_bounded_interval`), which
    covers whenever the in-region interval does: 72/72 on the fresh seed, median width 15.8
    points, so it is valid but wide. The report leads with the in-region rate. Measured only
    for independent models without trick items; in `tricks` the in-region rate itself is
    biased about +2 points, and no bootstrap repairs that.
