"""Hugging Face Hub access: listing, pinned revisions, byte-range parquet reads, checksums.

docs/DESIGN.md section 4.3 (milestone M2). No login is needed for the benchmark and label
datasets. The Open LLM Leaderboard per-sample details repositories are gated (automatic
approval after the user accepts the conditions in a browser, once per repository), so reading
them needs a token, resolved by ``huggingface_hub.get_token()`` (``HF_TOKEN`` or a cached
login). The token is never logged, printed or written to provenance files; it only travels in
the ``Authorization`` header.

Column projection: pyarrow reads only the parquet footer and the column chunks it needs
through :class:`RangeFile`, which issues HTTP ``Range`` requests with timeouts and retries
and counts every byte fetched, so the per-model download budget can be enforced.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

TIMEOUT_S = 60.0
RETRIES = 3
BACKOFF_S = (2.0, 4.0, 8.0)
USER_AGENT = "keyaudit (+https://github.com/h-a-forster/keyaudit)"
HF_ENDPOINT = "https://huggingface.co"

# Tests replace this with an ``httpx.MockTransport``; ``None`` means the real network.
_transport: httpx.BaseTransport | None = None
_sleep: Callable[[float], None] = time.sleep
# One keep-alive client per process: a fresh TLS connection (plus the Hub's redirect to its
# CDN) costs about a second per request, which dominated the per-sample reads.
_shared_client: httpx.Client | None = None
_client_lock = threading.Lock()
PREFETCH_WORKERS = 8


class AccessDenied(RuntimeError):
    """HTTP 401/403 on a gated repository: the user has not accepted its conditions."""

    denied = True  # checkpoint.run_units records the unit as denied and does not retry


class BudgetExceeded(RuntimeError):
    """A range read fetched more bytes than its budget allows."""


class ChecksumMismatch(RuntimeError):
    """A downloaded file does not match the sha256 published by the Hub."""


@dataclass(frozen=True)
class RemoteFile:
    repo_id: str
    repo_type: str
    revision: str  # full commit SHA, never a branch name, in anything that is recorded
    path: str
    size: int | None
    sha256: str | None  # LFS oid when the Hub exposes it

    @property
    def url(self) -> str:
        kind = "datasets/" if self.repo_type == "dataset" else ""
        rev = quote(self.revision, safe="")
        return f"{HF_ENDPOINT}/{kind}{self.repo_id}/resolve/{rev}/{quote(self.path)}"


def get_token() -> str | None:
    """The user's Hugging Face token (``HF_TOKEN`` or cached login), or ``None``."""
    from huggingface_hub import get_token as _get

    return _get()


def set_transport(transport: httpx.BaseTransport | None) -> None:
    """Route all HTTP through ``transport`` (tests); ``None`` restores the network."""
    global _transport, _shared_client
    with _client_lock:
        if _shared_client is not None:
            _shared_client.close()
        _shared_client = None
        _transport = transport


def _client() -> httpx.Client:
    """The process's shared keep-alive client (thread-safe; rebuilt by :func:`set_transport`)."""
    global _shared_client
    with _client_lock:
        if _shared_client is None:
            _shared_client = httpx.Client(
                transport=_transport,
                timeout=TIMEOUT_S,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
                limits=httpx.Limits(max_connections=2 * PREFETCH_WORKERS),
            )
        return _shared_client


def _headers(token: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    h = dict(extra or {})
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def request(
    url: str,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> httpx.Response:
    """HTTP request with timeout, retries on transport errors / 429 / 5xx, typed errors.

    401/403 raise :class:`AccessDenied`, 404 raises ``FileNotFoundError``. The error
    messages contain the URL but never the token.
    """
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            resp = _client().request(method, url, headers=_headers(token, headers))
        except httpx.TransportError as exc:
            last = exc
        else:
            if resp.status_code in (401, 403):
                raise AccessDenied(f"HTTP {resp.status_code} for {url}")
            if resp.status_code == 404:
                raise FileNotFoundError(f"HTTP 404 for {url}")
            if resp.status_code == 429 or resp.status_code >= 500:
                last = RuntimeError(f"HTTP {resp.status_code} for {url}")
            else:
                resp.raise_for_status()
                return resp
        if attempt < RETRIES:
            _sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
    raise RuntimeError(f"giving up after {RETRIES + 1} attempts: {last}")


def list_files(
    repo_id: str, revision: str, repo_type: str = "dataset", token: str | None = None
) -> list[RemoteFile]:
    """All files of a repository at ``revision`` with sizes and LFS sha256 (tree API)."""
    kind = "datasets" if repo_type == "dataset" else "models"
    rev = quote(revision, safe="")
    commit = resolve_revision(repo_id, revision, repo_type, token)
    out: list[RemoteFile] = []
    url: str | None = f"{HF_ENDPOINT}/api/{kind}/{repo_id}/tree/{rev}?recursive=true&expand=false"
    while url:
        resp = request(url, token)
        for entry in resp.json():
            if entry.get("type") != "file":
                continue
            lfs = entry.get("lfs") or {}
            out.append(
                RemoteFile(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=commit,
                    path=entry["path"],
                    size=entry.get("size"),
                    sha256=lfs.get("oid"),
                )
            )
        nxt = resp.links.get("next") if hasattr(resp, "links") else None
        url = nxt.get("url") if nxt else None
    return out


def resolve_revision(
    repo_id: str, revision: str, repo_type: str = "dataset", token: str | None = None
) -> str:
    """Full commit SHA of ``revision`` (a SHA is returned unchanged)."""
    if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision):
        return revision
    kind = "datasets" if repo_type == "dataset" else "models"
    rev = quote(revision, safe="")
    info = request(f"{HF_ENDPOINT}/api/{kind}/{repo_id}/revision/{rev}", token).json()
    sha = info.get("sha")
    if not sha:
        raise RuntimeError(f"could not resolve {repo_id}@{revision}")
    return str(sha)


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests that counts fetched bytes.

    Reads are served in blocks of at least ``min_block`` bytes (one request each) so that
    pyarrow's many small footer reads do not become many requests. :meth:`prefetch` fetches
    known byte ranges (the projected column chunks) concurrently up front; reads inside them
    are then served from memory. ``budget`` (bytes) aborts with :class:`BudgetExceeded`
    before a request would exceed it.
    """

    def __init__(
        self,
        url: str,
        size: int,
        token: str | None = None,
        budget: int | None = None,
        min_block: int = 64 * 1024,
    ) -> None:
        super().__init__()
        self.url = url
        self.size = int(size)
        self.bytes_fetched = 0
        self.requests = 0
        self.budget = budget
        self.min_block = min_block
        self._token = token
        self._pos = 0
        self._buf_start = 0
        self._buf = b""
        self._segments: list[tuple[int, bytes]] = []  # prefetched (start, data), sorted
        # The Hub answers each request with two redirects to a signed CDN URL; after the first
        # read the CDN URL is used directly (no token sent there) until it stops working.
        self._direct: str | None = None
        self._lock = threading.Lock()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def _check_budget(self, n: int) -> None:
        if self.budget is not None and self.bytes_fetched + n > self.budget:
            raise BudgetExceeded(
                f"reading {n} more bytes would exceed the budget of {self.budget} bytes "
                f"({self.bytes_fetched} already fetched) for {self.url}"
            )

    def _fetch(self, start: int, end: int, checked: bool = False) -> bytes:
        """Bytes [start, end) from the server (thread-safe)."""
        n = end - start
        if not checked:
            self._check_budget(n)
        rng = {"Range": f"bytes={start}-{end - 1}"}
        direct = self._direct
        resp = None
        if direct is not None:
            try:
                resp = request(direct, None, rng)
            except (AccessDenied, FileNotFoundError, httpx.HTTPStatusError):
                self._direct = None  # signature expired: go through the Hub again
        if resp is None:
            resp = request(self.url, self._token, rng)
            final = str(resp.url)
            if final != self.url and resp.status_code == 206:
                self._direct = final
        data = resp.content
        if resp.status_code != 206 and len(data) != n:
            raise RuntimeError(f"server ignored the Range header for {self.url}")
        if len(data) != n:
            raise RuntimeError(f"short range read: wanted {n} bytes, got {len(data)}")
        with self._lock:
            self.bytes_fetched += n
            self.requests += 1
        return data

    def prefetch(
        self, ranges: Sequence[tuple[int, int]], gap: int = 64 * 1024, workers: int = 0
    ) -> None:
        """Fetch the byte ranges ``[start, end)`` concurrently and keep them in memory.

        Ranges closer than ``gap`` bytes are merged into one request. The whole prefetch is
        checked against the budget before anything is requested.
        """
        merged: list[list[int]] = []
        buf_a, buf_b = self._buf_start, self._buf_start + len(self._buf)
        for a, b in sorted((max(0, a), min(self.size, b)) for a, b in ranges):
            if buf_a <= a and b <= buf_b:
                continue  # already read with the footer (small files)
            if b <= a:
                continue
            if merged and a - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        if self._buf and not any(a == buf_a for a, _ in self._segments):
            self._segments = sorted([*self._segments, (buf_a, self._buf)], key=lambda t: t[0])
        if not merged:
            return
        self._check_budget(sum(b - a for a, b in merged))
        n_workers = min(workers or PREFETCH_WORKERS, len(merged))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            datas = list(pool.map(lambda r: self._fetch(r[0], r[1], checked=True), merged))
        segs = self._segments + [(a, d) for (a, _), d in zip(merged, datas, strict=True)]
        self._segments = sorted(segs, key=lambda t: t[0])

    def _from_segments(self, start: int, n: int) -> bytes | None:
        for a, d in self._segments:
            if a <= start and start + n <= a + len(d):
                return d[start - a : start - a + n]
            if a > start:
                break
        return None

    def readinto(self, b: Any) -> int:
        view = memoryview(b).cast("B")
        want = min(len(view), max(0, self.size - self._pos))
        if want == 0:
            return 0
        seg = self._from_segments(self._pos, want) if self._segments else None
        if seg is not None:
            view[:want] = seg
            self._pos += want
            return want
        buf_end = self._buf_start + len(self._buf)
        if not (self._buf_start <= self._pos and self._pos + want <= buf_end):
            start = self._pos
            end = min(self.size, start + max(want, self.min_block))
            self._buf = self._fetch(start, end)
            self._buf_start = start
        off = self._pos - self._buf_start
        view[:want] = self._buf[off : off + want]
        self._pos += want
        return want


def read_parquet_columns(
    remote: RemoteFile,
    columns: list[str],
    budget_bytes: int,
    token: str | None = None,
    url: str | None = None,
) -> tuple[Any, dict[str, int]]:
    """Read ``columns`` of a remote parquet file; raise if more than ``budget_bytes`` fetched.

    Nested struct leaves can be named with dots (``"doc.question_id"``). Returns the
    ``pyarrow.Table`` and ``{"bytes_fetched", "requests", "remote_size"}``. ``url``
    overrides ``remote.url`` (the parquet listing API returns direct URLs).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    url = url or remote.url
    if remote.size is None:
        head = request(url, token, {"Range": "bytes=0-0"})
        rng = head.headers.get("Content-Range", "")
        if "/" not in rng:
            raise RuntimeError(f"cannot determine the size of {url}")
        size = int(rng.rsplit("/", 1)[1])
    else:
        size = remote.size
    rf = RangeFile(url, size, token=token, budget=budget_bytes)
    with pa.PythonFile(rf, mode="r") as fh:
        pf = pq.ParquetFile(fh)
        rf.prefetch(column_chunk_ranges(pf.metadata, columns))
        table = pf.read(columns=columns)
    return table, {"bytes_fetched": rf.bytes_fetched, "requests": rf.requests, "remote_size": size}


def column_chunk_ranges(metadata: Any, columns: Sequence[str]) -> list[tuple[int, int]]:
    """Byte ranges ``[start, end)`` of every column chunk that reading ``columns`` touches.

    A requested name selects the leaf columns equal to it or below it (``"doc.options"``
    selects ``doc.options.list.element``), in every row group.
    """
    out = []
    for g in range(metadata.num_row_groups):
        rg = metadata.row_group(g)
        for c in range(rg.num_columns):
            col = rg.column(c)
            path = col.path_in_schema
            if not any(path == name or path.startswith(name + ".") for name in columns):
                continue
            start = col.data_page_offset
            if col.has_dictionary_page and col.dictionary_page_offset is not None:
                start = min(start, col.dictionary_page_offset)
            out.append((start, start + col.total_compressed_size))
    return out


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_verified(remote: RemoteFile, dest_dir: str | Path, token: str | None = None) -> str:
    """Download a whole (small) file, verify its sha256 against the LFS oid, return path.

    The file lands at ``<dest_dir>/<repo_id with / as __>/<revision>/<path>``. An existing
    file with the expected checksum is reused. Without a published LFS oid the checksum is
    computed and recorded but cannot be verified.
    """
    dest = Path(dest_dir) / remote.repo_id.replace("/", "__") / remote.revision / remote.path
    if dest.exists() and remote.sha256 and sha256_file(dest) == remote.sha256:
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = request(remote.url, token)
    data = resp.content
    digest = hashlib.sha256(data).hexdigest()
    if remote.sha256 and digest != remote.sha256:
        raise ChecksumMismatch(f"{remote.repo_id}/{remote.path}: sha256 {digest} != expected")
    if remote.size is not None and len(data) != remote.size:
        raise ChecksumMismatch(f"{remote.repo_id}/{remote.path}: size {len(data)} != expected")
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)
    return str(dest)


def library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for dist in ("keyaudit", "numpy", "scipy", "pyarrow", "huggingface-hub", "httpx"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = "not installed"
    return out


def provenance(remote: RemoteFile, **extra: Any) -> dict[str, Any]:
    """A manifest record for a remote file (never contains the token)."""
    rec = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        **asdict(remote),
        **extra,
    }
    return rec


def append_manifest(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON line to a manifest file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
