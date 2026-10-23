# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Project scaffold: package layout, `docs/DESIGN.md` (estimands, mis-keyed latent class model,
  identifiability limits, validation and study design, milestones), CI, configs for the
  default and full-size study.
- `keyaudit.stats`: Clopper-Pearson and Wilson intervals, Holm and Benjamini-Hochberg
  adjustments, Bayesian FDR q-values, and the stratified difference estimator
  (prediction-powered inference for a mean with finite-population correction).
- `keyaudit.study.config`: validated TOML study configs with a stable hash.
- CLI: `keyaudit --version`, `versions`, `check-config`; study commands scaffolded.
