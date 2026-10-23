"""HTTP layer: retries, typed errors, byte-range parquet projection, budgets, checksums."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from keyaudit import hub

TOKEN = "hf_test_secret_value"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(hub, "_sleep", lambda s: None)
    yield
    hub.set_transport(None)


def _serve(handler: Callable[[httpx.Request], httpx.Response]) -> None:
    hub.set_transport(httpx.MockTransport(handler))


def _range_server(data: bytes, seen: list[httpx.Request]) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        rng = req.headers.get("Range")
        if not rng:
            return httpx.Response(200, content=data)
        a, b = rng.removeprefix("bytes=").split("-")
        start, end = int(a), min(int(b), len(data) - 1)
        return httpx.Response(
            206,
            content=data[start : end + 1],
            headers={"Content-Range": f"bytes {start}-{end}/{len(data)}"},
        )

    _serve(handler)


def _parquet_bytes(n: int = 5000) -> bytes:
    rng = np.random.default_rng(0)
    # a wide, incompressible text column we never read (like the harness's full prompts)
    big = [rng.bytes(200).hex() for _ in range(n)]
    table = pa.table(
        {
            "doc_id": np.arange(n),
            "doc": pa.array(
                [{"question_id": int(i), "text": big[i]} for i in range(n)],
            ),
            "filtered_resps": [[[str(v), "False"] for v in rng.normal(size=4)] for _ in range(n)],
            "arguments": big,
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf, row_group_size=2500)
    return buf.getvalue()


def test_retry_then_success() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(503) if len(calls) < 3 else httpx.Response(200, json={"ok": 1})

    _serve(handler)
    assert hub.request("https://example.test/x").json() == {"ok": 1}
    assert len(calls) == 3
    assert calls[0].headers["User-Agent"] == hub.USER_AGENT


def test_gives_up_after_retries() -> None:
    calls = []
    _serve(lambda req: calls.append(req) or httpx.Response(500))
    with pytest.raises(RuntimeError, match="giving up after 4 attempts"):
        hub.request("https://example.test/x")
    assert len(calls) == hub.RETRIES + 1


def test_transport_error_retried() -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        if len(calls) == 1:
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, text="fine")

    _serve(handler)
    assert hub.request("https://example.test/x").text == "fine"


@pytest.mark.parametrize("code", [401, 403])
def test_denied_is_typed_and_hides_token(code: int) -> None:
    seen: list[httpx.Request] = []
    _serve(lambda req: seen.append(req) or httpx.Response(code))
    with pytest.raises(hub.AccessDenied) as exc:
        hub.request("https://example.test/gated", token=TOKEN)
    assert TOKEN not in str(exc.value)
    assert getattr(exc.value, "denied", False) is True
    assert len(seen) == 1  # never retried
    assert seen[0].headers["Authorization"] == f"Bearer {TOKEN}"


def test_404_is_file_not_found() -> None:
    _serve(lambda req: httpx.Response(404))
    with pytest.raises(FileNotFoundError):
        hub.request("https://example.test/missing")


def test_range_projection_reads_a_fraction() -> None:
    data = _parquet_bytes()
    seen: list[httpx.Request] = []
    _range_server(data, seen)
    remote = hub.RemoteFile("org/repo", "dataset", "a" * 40, "x.parquet", len(data), None)
    table, st = hub.read_parquet_columns(
        remote, ["doc_id", "doc.question_id", "filtered_resps"], budget_bytes=len(data)
    )
    assert table.num_rows == 5000
    assert table.column("doc_id").to_pylist()[:3] == [0, 1, 2]
    assert st["bytes_fetched"] < 0.5 * len(data)
    assert st["requests"] == sum(1 for r in seen if "Range" in r.headers)
    assert all("Range" in r.headers for r in seen)


def test_range_size_probe_when_unknown() -> None:
    data = _parquet_bytes(100)
    _range_server(data, [])
    remote = hub.RemoteFile("org/repo", "dataset", "a" * 40, "x.parquet", None, None)
    table, st = hub.read_parquet_columns(remote, ["doc_id"], budget_bytes=10 * len(data))
    assert table.num_rows == 100 and st["remote_size"] == len(data)


def test_budget_enforced() -> None:
    data = _parquet_bytes()
    _range_server(data, [])
    remote = hub.RemoteFile("org/repo", "dataset", "a" * 40, "x.parquet", len(data), None)
    with pytest.raises(hub.BudgetExceeded):
        hub.read_parquet_columns(remote, ["arguments"], budget_bytes=200_000)


def test_rangefile_seek_and_read() -> None:
    data = bytes(range(256)) * 1000
    _range_server(data, [])
    rf = hub.RangeFile("https://example.test/f", len(data), min_block=1000)
    rf.seek(-10, io.SEEK_END)
    assert rf.read(10) == data[-10:]
    rf.seek(500)
    assert rf.read(3) == data[500:503]
    assert rf.read(0) == b""
    rf.seek(len(data))
    assert rf.read(5) == b""
    with pytest.raises(ValueError):
        rf.seek(-1)


def test_server_ignoring_range_is_an_error() -> None:
    data = b"x" * 10_000
    _serve(lambda req: httpx.Response(200, content=data))
    rf = hub.RangeFile("https://example.test/f", len(data), min_block=100)
    with pytest.raises(RuntimeError, match="ignored the Range"):
        rf.read(10)


def test_download_verified(tmp_path: Path) -> None:
    payload = b"hello parquet"
    digest = hashlib.sha256(payload).hexdigest()
    calls = []
    _serve(lambda req: calls.append(req) or httpx.Response(200, content=payload))
    remote = hub.RemoteFile("org/repo", "dataset", "b" * 40, "d/f.bin", len(payload), digest)
    path = hub.download_verified(remote, tmp_path)
    assert Path(path).read_bytes() == payload
    assert "org__repo" in path and "b" * 40 in path
    hub.download_verified(remote, tmp_path)  # cached: no second request
    assert len(calls) == 1


def test_download_checksum_mismatch(tmp_path: Path) -> None:
    _serve(lambda req: httpx.Response(200, content=b"tampered"))
    remote = hub.RemoteFile("org/repo", "dataset", "b" * 40, "f.bin", 8, "0" * 64)
    with pytest.raises(hub.ChecksumMismatch):
        hub.download_verified(remote, tmp_path)
    assert not list(tmp_path.rglob("f.bin"))


def test_list_files_and_resolve() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert "/tree/" in req.url.path
        return httpx.Response(
            200,
            json=[
                {"type": "directory", "path": "data"},
                {"type": "file", "path": "data/a.parquet", "size": 10, "lfs": {"oid": "c" * 64}},
                {"type": "file", "path": "README.md", "size": 3},
            ],
        )

    _serve(handler)
    files = hub.list_files("org/repo", "d" * 40)
    assert [f.path for f in files] == ["data/a.parquet", "README.md"]
    assert files[0].sha256 == "c" * 64 and files[1].sha256 is None
    assert files[0].revision == "d" * 40
    assert files[0].url.endswith("/datasets/org/repo/resolve/" + "d" * 40 + "/data/a.parquet")


def test_resolve_branch_name() -> None:
    _serve(lambda req: httpx.Response(200, json={"sha": "e" * 40}))
    assert hub.resolve_revision("org/repo", "refs/convert/parquet") == "e" * 40


def test_provenance_has_no_token(tmp_path: Path) -> None:
    remote = hub.RemoteFile("org/repo", "dataset", "b" * 40, "f.bin", 8, None)
    rec = hub.provenance(remote, bytes_fetched=8)
    hub.append_manifest(tmp_path / "m.jsonl", rec)
    text = (tmp_path / "m.jsonl").read_text(encoding="utf-8")
    assert "org/repo" in text and TOKEN not in text and "bytes_fetched" in text
