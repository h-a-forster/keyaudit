"""Resumable unit store and bounded process pool (docs/DESIGN.md section 6.1, milestone M1).

Every long computation is split into units (one model fetch, one bootstrap chunk, one
leave-one-family-out refit, ...). A finished unit is written atomically to
``<root>/units/<unit_id>.npz`` plus ``<unit_id>.json`` (status, config hash, package version,
seconds, unit metadata), and a line is appended to ``<root>/progress.log``. A rerun skips
units whose files exist. A run directory whose ``manifest.json`` records a different config
hash refuses to open. At most 3 worker processes, BLAS threads pinned to 1 in each worker.

Unit functions are plain module-level callables (picklable) that take keyword arguments and
return ``{"arrays": {name: ndarray}, "meta": {...}}``. A function that raises
:class:`AccessDenied`-like errors (any exception whose class sets ``denied = True``) is
recorded as ``denied`` and not retried.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

MAX_WORKERS = 3
_UNIT_RE = re.compile(r"^[A-Za-z0-9._=+\-]+(/[A-Za-z0-9._=+\-]+)*$")
_BLAS_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

UnitFn = Callable[..., dict[str, Any]]
Unit = tuple[str, UnitFn, dict[str, Any]]


class ConfigMismatch(RuntimeError):
    """The run directory was created with a different configuration."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parent,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else None


def _package_version() -> str:
    from keyaudit import __version__

    return __version__


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


class UnitStore:
    """Atomic per-unit result files, a manifest and ``progress.log`` under ``root``."""

    def __init__(self, root: Path | str, config_hash: str, read_only: bool = False) -> None:
        self.root = Path(root)
        self.config_hash = config_hash
        self.units_dir = self.root / "units"
        self.log_path = self.root / "progress.log"
        self.manifest_path = self.root / "manifest.json"
        self.read_only = read_only
        if read_only:  # inspection only (status, report): no manifest entry, no directories
            return
        self.units_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any]
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_hash") != config_hash:
                raise ConfigMismatch(
                    f"{self.root} was created with config hash "
                    f"{str(manifest.get('config_hash'))[:12]}..., not {config_hash[:12]}...; "
                    "use a new out_dir or restore the original config"
                )
        else:
            manifest = {"config_hash": config_hash, "created": _now(), "starts": []}
        manifest["starts"].append(
            {"time": _now(), "package_version": _package_version(), "git_sha": _git_sha()}
        )
        _atomic_write_bytes(
            self.manifest_path, json.dumps(manifest, indent=2).encode("utf-8") + b"\n"
        )

    # -- paths ------------------------------------------------------------------------------
    def _paths(self, unit_id: str) -> tuple[Path, Path]:
        if not _UNIT_RE.match(unit_id) or ".." in unit_id.split("/"):
            raise ValueError(f"invalid unit id {unit_id!r}")
        base = self.units_dir / unit_id
        return base.with_name(base.name + ".npz"), base.with_name(base.name + ".json")

    # -- queries ----------------------------------------------------------------------------
    def status(self, unit_id: str) -> str | None:
        """``"ok"``, ``"denied"`` or ``None`` (not finished)."""
        npz, meta = self._paths(unit_id)
        if not meta.exists():
            return None
        info = json.loads(meta.read_text(encoding="utf-8"))
        st = info.get("status")
        if st == "ok" and not npz.exists():
            return None
        return st if st in ("ok", "denied") else None

    def done(self, unit_id: str) -> bool:
        return self.status(unit_id) is not None

    def list_units(self, prefix: str = "") -> list[str]:
        """Finished unit ids under ``prefix`` (sorted)."""
        out = []
        if not self.units_dir.exists():
            return out
        for p in sorted(self.units_dir.rglob("*.json")):
            uid = p.relative_to(self.units_dir).with_suffix("").as_posix()
            if uid.startswith(prefix) and self.done(uid):
                out.append(uid)
        return out

    # -- writes -----------------------------------------------------------------------------
    def _writable(self) -> None:
        if self.read_only:
            raise PermissionError("UnitStore opened read-only")

    def save(self, unit_id: str, arrays: dict[str, Any], meta: dict[str, Any]) -> None:
        """Write the arrays, then the metadata (the metadata file marks the unit done)."""
        self._writable()
        npz, meta_path = self._paths(unit_id)
        npz.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=npz.parent, prefix=f".{npz.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, **{k: np.asarray(v) for k, v in arrays.items()})
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, npz)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        info = {
            "unit": unit_id,
            "status": "ok",
            "config_hash": self.config_hash,
            "package_version": _package_version(),
            "finished": _now(),
            "meta": _jsonable(meta),
        }
        _atomic_write_bytes(meta_path, json.dumps(info, indent=1).encode("utf-8") + b"\n")

    def mark_denied(self, unit_id: str, reason: str) -> None:
        self._writable()
        _, meta_path = self._paths(unit_id)
        info = {
            "unit": unit_id,
            "status": "denied",
            "config_hash": self.config_hash,
            "finished": _now(),
            "meta": {"reason": reason},
        }
        _atomic_write_bytes(meta_path, json.dumps(info, indent=1).encode("utf-8") + b"\n")

    def load(self, unit_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return ``(arrays, meta)`` of a finished ``ok`` unit."""
        npz, meta_path = self._paths(unit_id)
        if self.status(unit_id) != "ok":
            raise FileNotFoundError(f"unit {unit_id} is not finished")
        info = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return arrays, info["meta"]

    def log(self, line: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {line}\n")


def _init_worker() -> None:
    """Pin BLAS and OpenMP threads to one per worker process."""
    for var in _BLAS_ENV:
        os.environ[var] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # pragma: no cover - threadpoolctl missing or no BLAS found
        pass


def _call(fn: UnitFn, kwargs: dict[str, Any]) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    out = fn(**kwargs)
    return out, time.perf_counter() - t0


def _is_denied(exc: BaseException) -> bool:
    return bool(getattr(exc, "denied", False))


def _fmt_eta(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "?"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def run_units(
    store: UnitStore,
    units: Iterable[Unit],
    workers: int = MAX_WORKERS,
    retries: int = 2,
) -> dict[str, str]:
    """Run pending units with at most ``min(workers, 3)`` processes; return unit -> status.

    Statuses: ``ok`` (now or earlier), ``denied`` (now or earlier; never retried) and
    ``failed`` (after ``retries`` extra attempts; not saved, so a rerun tries again).
    ``workers=1`` runs units in the calling process (no pool).
    """
    all_units = list(units)
    ids = [u[0] for u in all_units]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate unit ids")
    statuses: dict[str, str] = {}
    pending: list[Unit] = []
    for u in all_units:
        st = store.status(u[0])
        if st is None:
            pending.append(u)
        else:
            statuses[u[0]] = st
    total, n_done = len(all_units), len(statuses)
    if not pending:
        return statuses
    workers = max(1, min(int(workers), MAX_WORKERS, len(pending)))
    store.log(f"start: {len(pending)} pending of {total} units, workers={workers}")
    t_start = time.perf_counter()
    n_new = 0
    attempts: dict[str, int] = {u[0]: 0 for u in pending}
    by_id = {u[0]: u for u in pending}

    def finish(uid: str, result: dict[str, Any] | None, secs: float, exc: BaseException | None):
        nonlocal n_done, n_new
        attempts[uid] += 1
        if exc is None and result is not None:
            store.save(uid, result.get("arrays", {}), {**result.get("meta", {}), "seconds": secs})
            statuses[uid] = "ok"
        elif exc is not None and _is_denied(exc):
            store.mark_denied(uid, f"{type(exc).__name__}: {exc}")
            statuses[uid] = "denied"
        elif attempts[uid] <= retries:
            store.log(
                f"retry {uid} attempt {attempts[uid]}/{retries + 1}: {type(exc).__name__}: {exc}"
            )
            return True
        else:
            tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            store.log(f"FAILED {uid} after {attempts[uid]} attempts: {tb}")
            statuses[uid] = "failed"
        n_done += 1
        n_new += 1
        elapsed = time.perf_counter() - t_start
        remaining = total - n_done
        eta = elapsed / n_new * remaining / 1.0 if n_new else float("nan")
        store.log(
            f"{statuses[uid]} {uid} secs={secs:.1f} done={n_done}/{total} eta={_fmt_eta(eta)}"
        )
        return False

    if workers == 1:
        queue = list(pending)
        while queue:
            uid, fn, kw = queue.pop(0)
            try:
                res, secs = _call(fn, kw)
                again = finish(uid, res, secs, None)
            except Exception as exc:  # noqa: BLE001 - recorded and retried
                again = finish(uid, None, 0.0, exc)
            if again:
                queue.append(by_id[uid])
        return statuses

    # Workers inherit the BLAS limits from the environment at spawn: numpy and scipy are
    # imported (with this module) before ``_init_worker`` runs, and OpenBLAS sizes its
    # buffers at import, so setting them only in the initializer left each worker with
    # about 1 GB of committed memory and a full-width thread pool until the first unit.
    saved_env = {var: os.environ.get(var) for var in _BLAS_ENV}
    os.environ.update(dict.fromkeys(_BLAS_ENV, "1"))
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futs: dict[Future[tuple[dict[str, Any], float]], str] = {}
            for uid, fn, kw in pending:
                futs[pool.submit(_call, fn, kw)] = uid
            while futs:
                finished, _ = wait(futs, return_when=FIRST_COMPLETED)
                for fut in finished:
                    uid = futs.pop(fut)
                    try:
                        res, secs = fut.result()
                        again = finish(uid, res, secs, None)
                    except Exception as exc:  # noqa: BLE001 - recorded and retried
                        again = finish(uid, None, 0.0, exc)
                    if again:
                        _, fn, kw = by_id[uid]
                        futs[pool.submit(_call, fn, kw)] = uid
    finally:
        for var, val in saved_env.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val
    return statuses
