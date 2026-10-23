"""Hugging Face Hub access: listing, pinned revisions, byte-range parquet reads, checksums.

docs/DESIGN.md section 4.3 (milestone M2). No login is needed for the benchmark and label
datasets. The Open LLM Leaderboard per-sample details repositories are gated (automatic
approval after the user accepts the conditions in a browser, once per repository), so reading
them needs a token, resolved by ``huggingface_hub.get_token()`` (``HF_TOKEN`` or a cached
login). The token is never logged, printed or written to provenance files.

Column projection: pyarrow reads only the parquet footer and the column chunks it needs
through :class:`RangeFile`, which issues HTTP ``Range`` requests with timeouts and retries
and counts every byte fetched, so the per-model download budget can be enforced.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from keyaudit._stub import not_yet

TIMEOUT_S = 60.0
USER_AGENT = "keyaudit (+https://github.com/h-a-forster/keyaudit)"


class AccessDenied(RuntimeError):
    """HTTP 401/403 on a gated repository: the user has not accepted its conditions."""


@dataclass(frozen=True)
class RemoteFile:
    repo_id: str
    repo_type: str
    revision: str  # full commit SHA, never a branch name, in anything that is recorded
    path: str
    size: int | None
    sha256: str | None  # LFS oid when the Hub exposes it


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests that counts fetched bytes."""

    def __init__(self, url: str, size: int, token: str | None = None) -> None:
        super().__init__()
        self.url = url
        self.size = size
        self.bytes_fetched = 0
        self._token = token
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


def read_parquet_columns(remote: RemoteFile, columns: list[str], budget_bytes: int) -> object:
    """Read ``columns`` of a remote parquet file; raise if more than ``budget_bytes`` fetched."""
    not_yet("M2", "read_parquet_columns")


def download_verified(remote: RemoteFile, dest_dir: str) -> str:
    """Download a whole (small) file, verify its sha256 against the LFS oid, return path."""
    not_yet("M2", "download_verified")
