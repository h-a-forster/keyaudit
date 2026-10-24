"""Command-line interface: ``keyaudit fetch | run | status | report | check-config | simulate |
versions``."""

from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from keyaudit import __version__
from keyaudit.checkpoint import MAX_WORKERS
from keyaudit.study.config import load_config
from keyaudit.study.runner import STAGES

_VERSIONED = ("numpy", "scipy", "pyarrow", "huggingface-hub", "httpx", "threadpoolctl")


def _stages(text: str) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in text.split(",") if p.strip())
    bad = [p for p in parts if p not in STAGES]
    if bad or not parts:
        raise argparse.ArgumentTypeError(f"unknown stage(s) {bad or [text]}; choose from {STAGES}")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyaudit",
        description="Estimate benchmark answer-key error rates from model-response panels.",
    )
    parser.add_argument("--version", action="version", version=f"keyaudit {__version__}")
    sub = parser.add_subparsers(dest="command")

    for name, help_text in (
        ("fetch", "download benchmark keys, human relabels and panel responses"),
        ("run", "run or resume the study stages"),
        ("status", "show done/total units per stage"),
        ("report", "write derived results for a finished run"),
        ("check-config", "validate a study config and print its hash"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, required=True)
        if name == "run":
            p.add_argument("--stages", type=_stages, default=STAGES)
            p.add_argument(
                "--workers", type=int, default=None, help=f"default from config; max {MAX_WORKERS}"
            )
    sim = sub.add_parser("simulate", help="run the synthetic-panel simulation study")
    sim.add_argument("--out", type=Path, default=Path("runs/simulation-v1"))
    sim.add_argument("--results", type=Path, default=Path("results/simulation-v1"))
    sim.add_argument("--profile", choices=("default", "quick", "coverage"), default="default")
    sim.add_argument("--seed", type=int, default=20261024)
    sim.add_argument("--workers", type=int, default=MAX_WORKERS)
    sub.add_parser("versions", help="print package and dependency versions")
    return parser


def _versions() -> None:
    print(f"keyaudit {__version__}")
    print(f"python {platform.python_version()}")
    for dist in _VERSIONED:
        try:
            print(f"{dist} {version(dist)}")
        except PackageNotFoundError:
            print(f"{dist} (not installed)")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "versions":
        _versions()
        return 0
    if args.command == "simulate":
        from keyaudit.study.simulation import run_simulation

        workers = min(args.workers, MAX_WORKERS)
        print(run_simulation(args.out, args.results, args.profile, args.seed, workers))
        return 0
    cfg = load_config(args.config)
    if args.command == "check-config":
        print(f"{cfg.study.name} {cfg.hash()}")
        return 0
    from keyaudit.study.runner import PreconditionFailed

    try:
        if args.command == "run":
            from keyaudit.study.runner import run_study

            workers = min(args.workers or cfg.run.workers, MAX_WORKERS)
            run_study(cfg, stages=args.stages, workers=workers)
        elif args.command == "status":
            from keyaudit.study.runner import status

            for stage, (done, total) in status(cfg).items():
                print(f"{stage:10s} {done}/{total}")
            log = Path(cfg.study.out_dir) / "progress.log"
            if log.exists():
                tail = log.read_text(encoding="utf-8").splitlines()[-1:]
                print(f"last log line: {tail[0] if tail else ''}")
        elif args.command == "report":
            from keyaudit.study.report import write_report

            print(write_report(cfg))
        elif args.command == "fetch":
            from keyaudit.study.runner import run_study

            run_study(cfg, stages=("fetch",), workers=cfg.run.workers)
    except PreconditionFailed as exc:
        print(f"keyaudit: precondition failed: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"keyaudit: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
