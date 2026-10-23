# Contributing

Bug reports, new benchmark adapters, human-label sources and documentation fixes are welcome.
Read [docs/DESIGN.md](docs/DESIGN.md) first: it fixes the estimands, the model, the validation
gates and the reporting rules.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git. From a clone:

```sh
uv sync
uv run keyaudit versions
```

## Tests and lint

```sh
uv run pytest              # fast suite, offline
uv run pytest --runslow    # plus simulation and recovery checks
uv run ruff check .
uv run ruff format --check .
```

CI runs the fast suite on Ubuntu (Python 3.11 and 3.13) and Windows (Python 3.13), plus
lint. Tests must run offline, write only under pytest's `tmp_path`, use fixed seeds, and pass
on Windows (use `pathlib`). Mark tests that take more than a few seconds with
`@pytest.mark.slow`. Statistical assertions use explicit exact intervals or stated seeded
tolerances.

## Code

- Python 3.11+, typed. Runtime dependencies: numpy, scipy, pyarrow, huggingface-hub, httpx,
  threadpoolctl. Open an issue before adding one.
- Long computations go through `keyaudit.checkpoint` (one file per finished unit, skip on
  rerun, `progress.log`), with at most 3 worker processes.
- Match the surrounding style; ruff enforces the rest (line length 100). Write text files as
  UTF-8.

## Data

- Never commit downloaded data, caches or run directories (`data/`, `cache/`, `runs/`).
  Commit only small derived results under `results/`.
- Do not redistribute leaderboard per-sample responses (no stated licence). Never write GPQA
  question text to any committed file, issue or log.
- Respect each source's licence (DESIGN section 4.1); Platinum is ShareAlike.

## Reporting a benchmark error

A claimed answer-key error needs the item id, the evidence, and a check that it is not an
adapter mistake (wrong revision, option mapping, text normalisation). Report confirmed errors
to the benchmark's maintainers.

## Pull requests

- One change per pull request, with tests and docs.
- Add an entry under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
- Keep the "Status and next steps" section of [AGENTS.md](AGENTS.md) current.

Security issues: see [SECURITY.md](SECURITY.md). Do not open a public issue.
