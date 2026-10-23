"""Run the simulation study with automatic resume-on-failure.

Usage: uv run python scripts/run_simulation.py [--profile default] [--out runs/simulation-v1]
       [--results results/simulation-v1] [--workers 3] [--attempts 3]

Wraps ``keyaudit simulate`` in up to ``--attempts`` attempts. Every finished unit is
checkpointed under ``<out>/units``, so a rerun only computes what is missing. Progress goes to
``<out>/progress.log``; this wrapper appends its attempt lines there too.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--profile", default="default", choices=("default", "quick"))
    ap.add_argument("--out", type=Path, default=Path("runs/simulation-v1"))
    ap.add_argument("--results", type=Path, default=Path("results/simulation-v1"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--pause", type=float, default=5.0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "progress.log"
    cmd = [sys.executable, "-m", "keyaudit", "simulate", "--profile", args.profile,
           "--out", str(args.out), "--results", str(args.results),
           "--workers", str(args.workers)]  # fmt: skip
    shown = " ".join(["python", *cmd[1:]])
    code = 1
    for attempt in range(1, args.attempts + 1):
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"{_stamp()} wrapper attempt {attempt}/{args.attempts}: {shown}\n")
        code = subprocess.call(cmd)
        if code == 0:
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"{_stamp()} wrapper finished\n")
            return 0
        time.sleep(args.pause)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"{_stamp()} wrapper gave up (exit {code})\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
