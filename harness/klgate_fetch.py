#!/usr/bin/env python3
"""Fetch a row subset of the BF16 teacher-logits dataset with HTTP range requests.

The dataset (default ``brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits``) stores, per sealed
2,048-token window, one safetensors file with a single F32 tensor ``logits`` of shape
[2047, 154880]: row r is the teacher's full-vocabulary logits after tokens[0..r], predicting
tokens[r + 1]. Each row is 619,520 bytes, so a subset of rows from every window can be fetched
with one range request per row instead of downloading 1.27 GB per window.

What this tool writes under ``--out`` (paths as in the dataset, so a full download and a subset
are read the same way by ``klgate.py``):

  README.md, dataset-manifest.json, capture-receipt.json, token-panel-receipt.json,
  backend.json, plan.json                          verbatim, verified against the Hub listing
  calibration/panel-v1/*.json                      verbatim (window metadata and receipts)
  calibration/panel-v1/arrays/<window>.tokens.npy  verbatim token ids, int32 [2048]
  teacher-rows/<window>.safetensors                the fetched rows: ``positions`` I32 [n] and
                                                   ``logits`` F32 [n, 154880], with the source
                                                   file's identity in ``__metadata__``
  FETCH-MANIFEST.json                              what was fetched (byte ranges, digests, checks)
  SHA256SUMS                                       sha256 of every file above (``sha256sum -c``)

Checks: every whole file against the Hub listing at the pinned revision (sha256 for LFS files,
the git blob id for the others); every token array against the dataset manifest; every window's
resolve headers (commit, sha256 and size of the linked file) and its safetensors header (dtype,
shape, offsets, token digest) against the dataset manifest; every range response's status,
Content-Range and length; every fetched value finite. A range of a file cannot be checked against
the file's sha256; the rows are bound to it through the resolve headers and recorded by their own
sha256.

One range request is latency-bound (a few hundred KB/s), so rows are fetched over several
connections; ``--rate`` caps their sum. An interrupted run resumes: a window in progress is
``<window>.safetensors.partial`` plus ``.partial.rows`` (the rows already written, one per line).

Standard library only. No token is needed for this public dataset; if ``HF_TOKEN`` is set it is
sent to the Hub host only (never to the CDN host a download redirects to) and never printed.

Rows: every window's first ``--head-rows`` rows (default 5: the KL per position is largest at the
start of a window and falls fastest there) and ``--rows-per-window`` evenly spaced rows (default
184, one at the middle of each of 184 equal strata). ``klgate.py`` weights each row by the
positions it stands for. Rows already in an earlier finished file for the same source are copied,
not fetched again.

Example (25 windows x 189 rows, about 2.93 GB at 3 MB/s, roughly 17 minutes):

  python3 harness/klgate_fetch.py --revision 95f4fdd94bf29989db2e0d1054e4931f55edb6aa \\
      --rows-per-window 184 --head-rows 5 --rate 3e6 --max-bytes 3e9 --connections 12 --out <dir>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import re
import struct
import sys
import threading
import time
import urllib.parse
from array import array

DEFAULT_REPO = "brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits"
USER_AGENT = "glm53f-klgate-fetch/1 (python-stdlib)"
CHUNK = 1 << 16
REDIRECTS = (301, 302, 303, 307, 308)

# Whole files fetched verbatim (besides one token array per selected window).
META_FILES = (
    "README.md",
    "dataset-manifest.json",
    "capture-receipt.json",
    "token-panel-receipt.json",
    "backend.json",
    "plan.json",
    "calibration/panel-v1/panel.json",
    "calibration/panel-v1/panel.receipt.json",
    "calibration/panel-v1/tokenizer.receipt.json",
    "calibration/panel-v1/corpus.receipt.json",
    "calibration/panel-v1/arrays/causal-mask-2048.npy",
)


class FetchError(Exception):
    pass


def log(msg: str) -> None:
    print(f"[fetch] {msg}", file=sys.stderr, flush=True)


def evenly_spaced(count: int, n: int) -> list[int]:
    """n distinct rows of 0..count-1, one at the middle of each of n equal strata."""
    if not 1 <= n <= count:
        raise FetchError(f"rows per window must be in 1..{count}, got {n}")
    return [((2 * i + 1) * count) // (2 * n) for i in range(n)]


def row_plan(count: int, n: int, head: int) -> tuple[list[int], str]:
    """The first ``head`` rows (where the KL per position is largest and falls fastest) and n
    evenly spaced rows; returns the sorted rows and the rule, for the file's metadata."""
    if not 0 <= head <= count:
        raise FetchError(f"--head-rows must be in 0..{count}")
    rows = sorted(set(range(head)) | set(evenly_spaced(count, n)))
    rule = f"evenly spaced: row_i = ((2i+1)*{count}) // (2*{n}), i < {n}"
    return rows, (f"rows 0..{head - 1}, and {rule}" if head else rule)


def git_blob_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def floats(buf: bytes) -> array:
    a = array("f")
    a.frombytes(buf)
    if sys.byteorder != "little":
        a.byteswap()
    return a


class Pacer:
    """Byte budget and a strict rate limit over all connections (no catch-up bursts)."""

    def __init__(self, rate: float, max_bytes: int):
        self.rate = rate
        self.max_bytes = max_bytes
        self.used = 0
        self.inflight = 0
        self.lock = threading.Lock()
        self.next_free = time.monotonic()
        self.t0 = time.monotonic()

    def reserve(self, n: int) -> None:
        with self.lock:
            if self.used + self.inflight + n > self.max_bytes:
                raise FetchError(
                    f"byte budget: {self.used:,} fetched + {self.inflight + n:,} requested "
                    f"> --max-bytes {self.max_bytes:,}"
                )
            self.inflight += n

    def release(self, n: int) -> None:
        with self.lock:
            self.inflight -= n

    def account(self, n: int) -> None:
        with self.lock:
            self.used += n
            now = time.monotonic()
            if self.rate > 0:
                self.next_free = max(self.next_free, now) + n / self.rate
            delay = self.next_free - now
        if delay > 0:
            time.sleep(delay)

    def summary(self) -> str:
        dt = time.monotonic() - self.t0
        return f"{self.used:,} bytes in {dt:.0f} s ({self.used / max(dt, 1e-9) / 1e6:.2f} MB/s)"


class Http:
    """One thread's persistent HTTPS connections, one per host; redirects are the caller's."""

    def __init__(self, pacer: Pacer, hub_host: str, token: str | None, timeout: float = 60.0):
        self.pacer = pacer
        self.hub_host = hub_host
        self.token = token
        self.timeout = timeout
        self.conns: dict[str, http.client.HTTPSConnection] = {}

    def _drop(self, host: str) -> None:
        c = self.conns.pop(host, None)
        if c is not None:
            c.close()

    def get(self, url: str, headers: dict[str, str] | None = None, expect: int = 1 << 16):
        """One GET; returns (status, headers, body), the body read in full under the pacer."""
        u = urllib.parse.urlsplit(url)
        if u.scheme != "https" or not u.hostname:
            raise FetchError(f"refusing non-https URL {u.scheme}://{u.hostname}")
        path = u.path + (f"?{u.query}" if u.query else "")
        h = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if headers:
            h.update(headers)
        if self.token and u.hostname == self.hub_host:
            h["Authorization"] = f"Bearer {self.token}"
        self.pacer.reserve(expect)
        try:
            last_err: Exception | None = None
            for _ in range(2):  # a kept-alive connection may have been closed by the server
                conn = self.conns.get(u.hostname)
                if conn is None:
                    conn = self.conns[u.hostname] = http.client.HTTPSConnection(
                        u.hostname, timeout=self.timeout
                    )
                try:
                    conn.request("GET", path, headers=h)
                    resp = conn.getresponse()
                    parts = []
                    while True:
                        b = resp.read(CHUNK)
                        if not b:
                            break
                        self.pacer.account(len(b))
                        parts.append(b)
                    if resp.getheader("Connection", "").lower() == "close":
                        self._drop(u.hostname)
                    hdrs = {k.lower(): v for k, v in resp.getheaders()}
                    return resp.status, hdrs, b"".join(parts)
                except (http.client.HTTPException, OSError) as e:
                    last_err = e
                    self._drop(u.hostname)
            raise ConnectionError(f"GET {u.hostname}{u.path}: {last_err}")
        finally:
            self.pacer.release(expect)


class Hub:
    """The dataset at one revision; each thread gets its own connections."""

    def __init__(self, pacer: Pacer, endpoint: str, repo: str, revision: str, token: str | None):
        self.pacer = pacer
        self.endpoint = endpoint.rstrip("/")
        self.repo = repo
        self.revision = revision
        self.token = token
        self.host = urllib.parse.urlsplit(self.endpoint).hostname or ""
        self.local = threading.local()

    @property
    def http(self) -> Http:
        h = getattr(self.local, "http", None)
        if h is None:
            h = self.local.http = Http(self.pacer, self.host, self.token)
        return h

    def listing(self) -> dict[str, dict]:
        url = f"{self.endpoint}/api/datasets/{self.repo}/revision/{self.revision}?blobs=true"
        status, _, body = self.http.get(url, expect=4 << 20)
        if status != 200:
            raise FetchError(f"listing: HTTP {status}")
        info = json.loads(body)
        if info.get("sha") != self.revision:
            raise FetchError(f"listing is for commit {info.get('sha')}, not {self.revision}")
        return {s["rfilename"]: s for s in info["siblings"]}

    def resolve_url(self, path: str) -> str:
        quoted = urllib.parse.quote(path)
        return f"{self.endpoint}/datasets/{self.repo}/resolve/{self.revision}/{quoted}"

    def whole_file(self, path: str, entry: dict) -> bytes:
        """A small file in full, following redirects (LFS files go to the CDN, others to a cache)."""
        size = int(entry["size"])
        url = self.resolve_url(path)
        for _ in range(4):
            status, hdrs, body = self.http.get(url, expect=size + 8192)
            if status not in REDIRECTS:
                break
            url = urllib.parse.urljoin(url, hdrs["location"])
        if status != 200:
            raise FetchError(f"{path}: HTTP {status}")
        if not matches_listing(body, entry):
            raise FetchError(f"{path}: content does not match the listing at {self.revision}")
        return body

    def resolve_linked(self, path: str) -> tuple[str, dict[str, str]]:
        """The CDN URL of an LFS file and the resolve headers that identify it."""
        url = self.resolve_url(path)
        status, hdrs, _ = self.http.get(url, expect=8192)
        if status not in REDIRECTS or "location" not in hdrs:
            raise FetchError(f"{path}: resolve returned HTTP {status}, expected a redirect")
        return urllib.parse.urljoin(url, hdrs["location"]), hdrs


def matches_listing(body: bytes, entry: dict) -> bool:
    if len(body) != int(entry["size"]):
        return False
    lfs = entry.get("lfs")
    if lfs:
        return hashlib.sha256(body).hexdigest() == lfs["sha256"]
    return git_blob_sha1(body) == entry["blobId"]


class RangeReader:
    """Byte ranges of one LFS file from its CDN URL, re-resolved when the signed URL expires."""

    def __init__(self, hub: Hub, path: str, sha256: str, size: int):
        self.hub, self.path, self.sha256, self.size = hub, path, sha256, size
        self.lock = threading.Lock()
        self.url = ""
        self.identity: dict[str, str] = {}
        self._resolve("")

    def _resolve(self, stale: str) -> None:
        with self.lock:
            if self.url != stale:  # another thread already re-resolved
                return
            url, hdrs = self.hub.resolve_linked(self.path)
            etag = hdrs.get("x-linked-etag", "").strip('"')
            if hdrs.get("x-repo-commit") != self.hub.revision:
                raise FetchError(f"{self.path}: resolved at commit {hdrs.get('x-repo-commit')}")
            if etag != self.sha256 or int(hdrs.get("x-linked-size", -1)) != self.size:
                raise FetchError(f"{self.path}: linked file {etag} is not {self.sha256}/{self.size}")
            self.url = url
            self.identity = {
                "x-repo-commit": hdrs["x-repo-commit"],
                "x-linked-etag": etag,
                "x-linked-size": hdrs["x-linked-size"],
                "x-xet-hash": hdrs.get("x-xet-hash", ""),
            }

    def read(self, start: int, length: int) -> bytes:
        end = start + length - 1
        if start < 0 or end >= self.size:
            raise FetchError(f"{self.path}: range {start}-{end} outside {self.size} bytes")
        want = f"bytes {start}-{end}/{self.size}"
        delay = 1.0
        for _ in range(8):
            url = self.url
            try:
                status, hdrs, body = self.hub.http.get(
                    url, headers={"Range": f"bytes={start}-{end}"}, expect=length + 8192
                )
            except ConnectionError as e:
                log(f"{self.path}: {e}; retrying in {delay:.0f} s")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if status == 206 and hdrs.get("content-range") == want and len(body) == length:
                return body
            if status in (401, 403, 404, 410):  # the signed URL expired
                log(f"{self.path}: HTTP {status} from the CDN, re-resolving")
                self._resolve(url)
                continue
            log(f"{self.path}: HTTP {status}, {hdrs.get('content-range')!r}, {len(body)} bytes for "
                f"{want}; retrying in {delay:.0f} s")
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise FetchError(f"{self.path}: range {start}-{end} failed after retries")


def st_header(tensors: dict[str, tuple[str, list[int], int]], metadata: dict[str, str]) -> bytes:
    """A safetensors header for tensors laid out in dict order: {name: (dtype, shape, nbytes)}."""
    obj: dict[str, object] = {"__metadata__": metadata}
    off = 0
    for name, (dtype, shape, nbytes) in tensors.items():
        obj[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
        off += nbytes
    text = json.dumps(obj, separators=(",", ":")).encode()
    text += b" " * (-(8 + len(text)) % 8)  # data starts 8-byte aligned
    return struct.pack("<Q", len(text)) + text


class Window:
    """One window's subset file being assembled: header first, rows written at fixed offsets."""

    def __init__(self, hub: Hub, out: str, w: dict, rows: list[int], rule: str, listing: dict):
        self.w, self.rows, self.out = w, rows, out
        self.wid, src = w["window_id"], w["path"]
        entry = listing.get(src)
        if entry is None or (entry.get("lfs") or {}).get("sha256") != w["sha256"]:
            raise FetchError(f"{self.wid}: {src} at this revision is not the manifest's {w['sha256']}")
        self.reader = RangeReader(hub, src, w["sha256"], int(w["bytes"]))
        head8 = self.reader.read(0, 8)
        (hlen,) = struct.unpack("<Q", head8)
        if not 2 <= hlen <= 1 << 20:
            raise FetchError(f"{self.wid}: implausible safetensors header length {hlen}")
        self.header_json = self.reader.read(8, hlen)
        header = json.loads(self.header_json)
        count = int(w["prediction_positions"])
        t = header.get("logits") or {}
        if t.get("dtype") != "F32" or len(t.get("shape", [])) != 2 or t["shape"][0] != count:
            raise FetchError(f"{self.wid}: unexpected logits tensor {t}")
        self.vocab = int(t["shape"][1])
        self.row_bytes = self.vocab * 4
        if t["data_offsets"] != [0, count * self.row_bytes] or 8 + hlen + count * self.row_bytes != int(w["bytes"]):
            raise FetchError(f"{self.wid}: logits offsets {t['data_offsets']} do not fill the file")
        meta = header.get("__metadata__") or {}
        if meta.get("token_ids_sha256") != w["token_ids_sha256"] or meta.get("window_id") != self.wid:
            raise FetchError(f"{self.wid}: header metadata {meta} does not match the dataset manifest")
        self.src_base = 8 + hlen
        self.src_header_sha256 = hashlib.sha256(head8 + self.header_json).hexdigest()
        n = len(rows)
        local_meta = {
            "format": "glm53f-teacher-rows.v1",
            "window_id": self.wid,
            "source_repo": hub.repo,
            "source_revision": hub.revision,
            "source_path": src,
            "source_sha256": w["sha256"],
            "source_bytes": str(w["bytes"]),
            "source_header_sha256": self.src_header_sha256,
            "token_ids_sha256": w["token_ids_sha256"],
            "prediction_positions": str(count),
            "vocab": str(self.vocab),
            "row_rule": rule,
        }
        hdr = st_header(
            {"positions": ("I32", [n], 4 * n), "logits": ("F32", [n, self.vocab], n * self.row_bytes)},
            local_meta,
        )
        self.dst = os.path.join(out, "teacher-rows", f"{self.wid}.safetensors")
        self.partial, self.sidecar = self.dst + ".partial", self.dst + ".partial.rows"
        self.data0 = len(hdr) + 4 * n  # file offset of the first row
        os.makedirs(os.path.dirname(self.dst), exist_ok=True)
        self.done: set[int] = set()
        if os.path.exists(self.partial) and os.path.exists(self.sidecar):
            with open(self.partial, "rb") as f:
                same = f.read(len(hdr)) == hdr
            if same:
                size = os.path.getsize(self.partial)
                with open(self.sidecar) as f:
                    for line in f:
                        k = line.strip()
                        if k.isdigit() and self.data0 + (int(k) + 1) * self.row_bytes <= size:
                            self.done.add(int(k))
        if not self.done:
            pos = array("i", rows)
            if sys.byteorder != "little":
                pos.byteswap()
            with open(self.partial, "wb") as f:
                f.write(hdr)
                f.write(pos.tobytes())
            open(self.sidecar, "w").close()
        self.fd = os.open(self.partial, os.O_WRONLY)
        self.side = open(self.sidecar, "a")
        self.lock = threading.Lock()
        self.reused = self._reuse_local()

    def _reuse_local(self) -> int:
        """Copy rows this plan shares with an earlier, finished file of the same source."""
        if not os.path.exists(self.dst):
            return 0
        try:
            with open(self.dst, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(n))
                a, b = hdr["positions"]["data_offsets"]
                f.seek(8 + n + a)
                pos = array("i")
                pos.frombytes(f.read(b - a))
                if sys.byteorder != "little":
                    pos.byteswap()
                if hdr["__metadata__"].get("source_sha256") != self.w["sha256"] or hdr["logits"]["shape"][1] != self.vocab:
                    return 0
                old = {p: i for i, p in enumerate(pos)}
                base = 8 + n + hdr["logits"]["data_offsets"][0]
                copied = 0
                for k, r in enumerate(self.rows):
                    if k in self.done or r not in old:
                        continue
                    f.seek(base + old[r] * self.row_bytes)
                    buf = f.read(self.row_bytes)
                    if len(buf) != self.row_bytes:
                        break
                    os.pwrite(self.fd, buf, self.data0 + k * self.row_bytes)
                    self.side.write(f"{k}\n")
                    self.done.add(k)
                    copied += 1
                self.side.flush()
                return copied
        except (OSError, ValueError, KeyError):
            return 0

    def todo(self) -> list[int]:
        return [k for k in range(len(self.rows)) if k not in self.done]

    def fetch_row(self, k: int) -> int:
        r = self.rows[k]
        buf = self.reader.read(self.src_base + r * self.row_bytes, self.row_bytes)
        if not math.isfinite(sum(floats(buf))):
            raise FetchError(f"{self.wid} row {r}: non-finite teacher logits")
        os.pwrite(self.fd, buf, self.data0 + k * self.row_bytes)
        with self.lock:
            self.side.write(f"{k}\n")
            self.side.flush()
            self.done.add(k)
        return len(buf)

    def finish(self) -> dict:
        os.close(self.fd)
        self.side.close()
        if len(self.done) != len(self.rows):
            raise FetchError(f"{self.wid}: {len(self.done)} of {len(self.rows)} rows")
        h = hashlib.sha256()
        with open(self.partial, "rb") as f:  # every row again: finite, and the digest of the rows
            f.seek(self.data0)
            for _ in self.rows:
                buf = f.read(self.row_bytes)
                if len(buf) != self.row_bytes or not math.isfinite(sum(floats(buf))):
                    raise FetchError(f"{self.wid}: a short or non-finite row in {self.partial}")
                h.update(buf)
            if f.read(1):
                raise FetchError(f"{self.wid}: trailing bytes in {self.partial}")
        os.replace(self.partial, self.dst)
        os.remove(self.sidecar)
        return {
            "window_id": self.wid,
            "source": {
                "path": self.w["path"],
                "sha256": self.w["sha256"],
                "bytes": int(self.w["bytes"]),
                "resolve": self.reader.identity,
                "header_bytes": self.src_base,
                "header_sha256": self.src_header_sha256,
                "header_json": self.header_json.decode(),
                "row_bytes": self.row_bytes,
                "byte_ranges": "row r: [header_bytes + r*row_bytes, header_bytes + (r+1)*row_bytes)",
            },
            "rows": self.rows,
            "rows_reused_from_an_earlier_local_file": self.reused,
            "rows_sha256": h.hexdigest(),
            "local": os.path.relpath(self.dst, self.out),
        }


def _ident(entry: dict) -> dict:
    lfs = entry.get("lfs")
    if lfs:
        return {"bytes": int(entry["size"]), "sha256": lfs["sha256"], "verified_by": "lfs sha256"}
    return {"bytes": int(entry["size"]), "git_blob": entry["blobId"], "verified_by": "git blob id"}


def _complete(path: str, w: dict, rows: list[int]) -> bool:
    """A finished subset file for this window and these rows."""
    try:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
            a, b = hdr["positions"]["data_offsets"]
            f.seek(8 + n + a)
            pos = array("i")
            pos.frombytes(f.read(b - a))
        if sys.byteorder != "little":
            pos.byteswap()
        return (
            hdr["__metadata__"].get("source_sha256") == w["sha256"]
            and list(pos) == rows
            and os.path.getsize(path) == 8 + n + hdr["logits"]["data_offsets"][1]
        )
    except (OSError, ValueError, KeyError):
        return False


def run(a: argparse.Namespace) -> int:
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        raise FetchError("--revision must be a full 40-hex commit")
    out = os.path.abspath(a.out)
    pacer = Pacer(a.rate, int(a.max_bytes))
    hub = Hub(pacer, a.endpoint, a.repo, a.revision, os.environ.get("HF_TOKEN") or None)

    listing = hub.listing()
    log(f"{a.repo}@{a.revision}: {len(listing)} files listed")
    local_manifest = os.path.join(out, "dataset-manifest.json")
    if os.path.exists(local_manifest) and matches_listing(
        open(local_manifest, "rb").read(), listing["dataset-manifest.json"]
    ):
        manifest_raw = open(local_manifest, "rb").read()
    else:
        manifest_raw = hub.whole_file("dataset-manifest.json", listing["dataset-manifest.json"])
    manifest = json.loads(manifest_raw)
    finals = [w for w in manifest["logit_files"] if w.get("role") == "final"]
    if a.windows:
        want = a.windows.split(",")
        finals = [w for w in finals if w["window_id"] in want]
        if len(finals) != len(want):
            raise FetchError(f"unknown window ids in --windows {a.windows}")
    counts = {int(w["prediction_positions"]) for w in finals}
    if len(counts) != 1:
        raise FetchError(f"windows with different position counts {counts}")
    rows, rule = row_plan(counts.pop(), a.rows_per_window, a.head_rows)
    vocab = int(manifest["vocab_size"])
    tokens = [f"calibration/panel-v1/arrays/{w['window_id']}.tokens.npy" for w in finals]
    whole = ["dataset-manifest.json"] + [p for p in META_FILES if p != "dataset-manifest.json"] + tokens
    missing = [p for p in whole if p not in listing]
    if missing:
        raise FetchError(f"not in the listing: {missing}")
    planned = sum(int(listing[p]["size"]) for p in whole) + len(finals) * (len(rows) * vocab * 4 + 2048)
    log(f"plan: {len(finals)} windows x {len(rows)} rows ({rule}) of {vocab * 4:,} bytes, plus {len(whole)} "
        f"whole files: {planned:,} bytes at most; {pacer.used:,} fetched so far; budget {pacer.max_bytes:,}")
    if a.plan_only:
        print(json.dumps({"windows": [w["window_id"] for w in finals], "rows": rows, "bytes": planned}))
        return 0

    os.makedirs(out, exist_ok=True)
    files = []
    for p in whole:
        dst = os.path.join(out, p)
        if not (os.path.exists(dst) and matches_listing(open(dst, "rb").read(), listing[p])):
            write_atomic(dst, manifest_raw if p == "dataset-manifest.json" else hub.whole_file(p, listing[p]))
        files.append({"path": p, **_ident(listing[p])})
    by_id = {w["window_id"]: w for w in finals}
    for p in tokens:
        wid = os.path.basename(p)[: -len(".tokens.npy")]
        if sha256_file(os.path.join(out, p)) != by_id[wid]["token_ids_sha256"]:
            raise FetchError(f"{p}: sha256 differs from the manifest's token_ids_sha256")
    log(f"whole files verified: {len(files)}; {pacer.summary()}")

    windows: list[Window] = []
    kept: list[dict] = []
    for w in finals:
        dst = os.path.join(out, "teacher-rows", f"{w['window_id']}.safetensors")
        if _complete(dst, w, rows):
            log(f"{w['window_id']}: complete from an earlier run")
            kept.append({"window_id": w["window_id"], "rows": rows, "local": os.path.relpath(dst, out),
                         "rows_sha256": None, "note": "complete from an earlier run of this tool"})
            continue
        windows.append(Window(hub, out, w, rows, rule, listing))
        if windows[-1].done:
            log(f"{w['window_id']}: {len(windows[-1].done)} of {len(rows)} rows already on disk")

    tasks = [(win, k) for win in windows for k in win.todo()]
    log(f"{len(tasks)} rows to fetch over {a.connections} connections")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.connections) as ex:
        futures = [ex.submit(win.fetch_row, k) for win, k in tasks]
        try:
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    left = (len(tasks) - done) * windows[0].row_bytes / max(a.rate, 1.0)
                    log(f"{done}/{len(tasks)} rows, {pacer.summary()}, about {left / 60:.0f} min left")
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    records = [win.finish() for win in windows] + kept

    sums = []
    for root, _, names in os.walk(out):
        for name in sorted(names):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, out)
            if rel in ("SHA256SUMS", "FETCH-MANIFEST.json") or ".partial" in rel:
                continue
            sums.append((rel, sha256_file(full), os.path.getsize(full)))
    sums.sort()
    report = {
        "schema": "glm53f-teacher-fetch.v1",
        "repo": a.repo,
        "revision": a.revision,
        "endpoint": a.endpoint,
        "vocab": vocab,
        "rows_per_window": len(rows),
        "row_rule": rule,
        "bytes_fetched_this_run": pacer.used,
        "fetch_rate_limit_bytes_per_second": a.rate,
        "whole_files": files,
        "windows": sorted(records, key=lambda r: r["window_id"]),
        "local_files": [{"path": p, "sha256": h, "bytes": n} for p, h, n in sums],
    }
    write_atomic(os.path.join(out, "FETCH-MANIFEST.json"), (json.dumps(report, indent=1) + "\n").encode())
    write_atomic(os.path.join(out, "SHA256SUMS"), "".join(f"{h}  {p}\n" for p, h, _ in sums).encode())
    log(f"done: {pacer.summary()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="destination directory (created)")
    ap.add_argument("--revision", required=True, help="dataset commit, 40 hex digits")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    ap.add_argument("--rows-per-window", type=int, default=184, help="evenly spaced rows per window")
    ap.add_argument("--head-rows", type=int, default=5, help="also every row 0..K-1 of each window")
    ap.add_argument("--windows", default="", help="comma-separated window ids (default: every final window)")
    ap.add_argument("--rate", type=float, default=3e6, help="bytes per second over all connections (0: unlimited)")
    ap.add_argument("--max-bytes", type=float, default=3e9, help="refuse to fetch more than this")
    ap.add_argument("--connections", type=int, default=12)
    ap.add_argument("--plan-only", action="store_true", help="print the plan and the byte count, fetch no rows")
    a = ap.parse_args()
    try:
        return run(a)
    except FetchError as e:
        print(f"klgate_fetch: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
