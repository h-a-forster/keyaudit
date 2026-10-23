"""Run a keyaudit study with automatic resume-on-failure.

Usage: uv run python scripts/run_study.py --config configs/study.toml [--attempts 3]
       [--workers 3] [--stages fetch,labels,...]

Wraps ``keyaudit run`` in up to ``--attempts`` attempts. Every finished unit is checkpointed,
so a rerun only computes what is missing. Progress goes to ``<out_dir>/progress.log``; this
wrapper appends its own attempt lines there too. Exit code 3 from ``keyaudit run`` means a
precondition failed (for example too few readable panel models) and is not retried.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def loggable_command(cmd: list[str]) -> str:
    """The command as logged: the interpreter's absolute path (which can contain a user
    name, and progress logs may be published) is replaced by its bare name."""
    return " ".join(["python", *cmd[1:]])


def _stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--stages", default=None, help="comma-separated stages (default: all)")
    ap.add_argument("--pause", type=float, default=5.0, help="seconds between attempts")
    args = ap.parse_args(argv)
    with open(args.config, "rb") as fh:
        out = Path(tomllib.load(fh)["study"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log = out / "progress.log"
    cmd = [sys.executable, "-m", "keyaudit", "run", "--config", str(args.config),
           "--workers", str(args.workers)]  # fmt: skip
    if args.stages:
        cmd += ["--stages", args.stages]
    shown = loggable_command(cmd)
    code = 1
    for attempt in range(1, args.attempts + 1):
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"{_stamp()} wrapper attempt {attempt}/{args.attempts}: {shown}\n")
        code = subprocess.call(cmd)
        if code == 0:
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"{_stamp()} wrapper finished\n")
            return 0
        if code == 3:
            break
        time.sleep(args.pause)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"{_stamp()} wrapper gave up (exit {code})\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
