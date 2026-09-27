#!/usr/bin/env python3
"""Fetch a subset of tensors from a Hugging Face safetensors checkpoint by HTTP range.

A sharded checkpoint often scatters the tensors one machine needs (for example the
coordinator's non-expert weights) across most of its shards. Downloading whole
shards then costs far more than the tensors themselves. This tool reads each
shard's header, fetches only the selected tensors' byte ranges, and writes them
as new safetensors files with a subset index and a manifest that maps every
tensor back to its source shard and offsets. Standard library only.

Examples:
  # every non-expert text tensor (attention, shared and dense MLPs, routers,
  # norms, embeddings, LM head, MTP non-expert), no vision tower:
  fetch_tensors.py --repo zai-org/GLM-5.3-Flash --revision <sha> --out DIR --select nonexpert
  # the routed experts of layers 3 and 4:
  fetch_tensors.py --repo ... --revision ... --out DIR2 --select experts:3,4
  # any tensors matching a regular expression:
  fetch_tensors.py --repo ... --revision ... --out DIR3 --select 're:layers\\.0\\.'

A token, if needed, is read from the environment variable named by --token-env
(default HF_TOKEN) and never printed. Re-running resumes: finished chunks are
recorded in OUT/.fetch-progress.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

SMALL_FILES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "processor_config.json", "special_tokens_map.json",
    "model.safetensors.index.json", "README.md", "LICENSE",
)
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
               "U8": 1, "U16": 2, "U32": 4, "U64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "BOOL": 1}
MAX_OUT_FILE = 4 << 30


def url_for(repo: str, rev: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{rev}/{name}"


def request(url: str, token: str | None, byte_range: tuple[int, int] | None = None, tries: int = 8) -> bytes:
    headers = {"User-Agent": "glm53f-afd-fetch/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    delay = 2.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                data = r.read()
            if byte_range and len(data) != byte_range[1] - byte_range[0] + 1:
                raise IOError(f"short read {len(data)} for range {byte_range}")
            return data
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise
            if attempt == tries - 1:
                raise
            print(f"retry {attempt + 1}: {url.rsplit('/', 1)[-1]} {byte_range}: {e}", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def shard_header(repo: str, rev: str, shard: str, token: str | None) -> tuple[int, dict]:
    n = struct.unpack("<Q", request(url_for(repo, rev, shard), token, (0, 7)))[0]
    return 8 + n, json.loads(request(url_for(repo, rev, shard), token, (8, 8 + n - 1)))


def selector(spec: str):
    if spec == "nonexpert":
        return lambda name: "mlp.experts." not in name and not name.startswith(("model.visual", "visual"))
    if spec.startswith("experts:"):
        layers = {int(x) for x in spec.split(":", 1)[1].split(",") if x.strip()}
        pat = re.compile(r"layers\.(\d+)\.mlp\.experts\.")
        def pick(name: str) -> bool:
            m = pat.search(name)
            return bool(m) and int(m.group(1)) in layers
        return pick
    if spec.startswith("re:"):
        rx = re.compile(spec[3:])
        return lambda name: bool(rx.search(name))
    raise SystemExit(f"unknown --select {spec!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", required=True, action="append",
                    help="nonexpert | experts:L1,L2 | re:REGEX (repeatable; union)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--token-env", default="HF_TOKEN")
    ap.add_argument("--prefix", default="subset")
    args = ap.parse_args()

    token = os.environ.get(args.token_env) or None
    os.makedirs(args.out, exist_ok=True)
    picks = [selector(s) for s in args.select]

    for name in SMALL_FILES:
        dst = os.path.join(args.out, name if name != "model.safetensors.index.json" else "source.model.safetensors.index.json")
        if os.path.exists(dst):
            continue
        try:
            data = request(url_for(args.repo, args.revision, name), token, tries=3)
        except Exception:
            continue
        with open(dst, "wb") as f:
            f.write(data)

    index = json.loads(request(url_for(args.repo, args.revision, "model.safetensors.index.json"), token))
    wanted = sorted(n for n in index["weight_map"] if any(p(n) for p in picks))
    if not wanted:
        raise SystemExit("selection is empty")
    shards = sorted({index["weight_map"][n] for n in wanted})
    print(f"{len(wanted)} tensors from {len(shards)} shards", flush=True)

    plan = []  # (name, shard, abs_start, nbytes, dtype, shape)
    for shard in shards:
        data_start, hdr = shard_header(args.repo, args.revision, shard, token)
        for name in wanted:
            if index["weight_map"][name] != shard:
                continue
            meta = hdr[name]
            a, b = meta["data_offsets"]
            plan.append((name, shard, data_start + a, b - a, meta["dtype"], meta["shape"]))

    # group into output files of at most MAX_OUT_FILE bytes, in source order
    groups, cur, cur_bytes = [], [], 0
    for item in sorted(plan, key=lambda t: (t[1], t[2])):
        if cur and cur_bytes + item[3] > MAX_OUT_FILE:
            groups.append(cur)
            cur, cur_bytes = [], 0
        cur.append(item)
        cur_bytes += item[3]
    if cur:
        groups.append(cur)

    total = sum(t[3] for t in plan)
    print(f"{total / 1e9:.2f} GB in {len(groups)} output files", flush=True)

    progress_path = os.path.join(args.out, ".fetch-progress.json")
    done = set(json.load(open(progress_path))) if os.path.exists(progress_path) else set()
    lock = threading.Lock()
    chunk = args.chunk_mb << 20
    jobs, files = [], []
    for gi, group in enumerate(groups):
        out_name = f"{args.prefix}-{gi + 1:05d}-of-{len(groups):05d}.safetensors"
        header, off = {}, 0
        for name, shard, start, nbytes, dtype, shape in group:
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
            off += nbytes
        header["__metadata__"] = {"format": "pt", "source_repo": args.repo, "source_revision": args.revision}
        hbytes = json.dumps(header, separators=(",", ":")).encode()
        hbytes += b" " * (-len(hbytes) % 8)
        base = 8 + len(hbytes)
        path = os.path.join(args.out, out_name)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        os.pwrite(fd, struct.pack("<Q", len(hbytes)) + hbytes, 0)
        os.ftruncate(fd, base + off)
        files.append((out_name, path, fd, header, base))
        for name, shard, start, nbytes, dtype, shape in group:
            dst = base + header[name]["data_offsets"][0]
            for c in range(0, nbytes, chunk):
                key = f"{out_name}:{name}:{c}"
                if key not in done:
                    jobs.append((key, fd, url_for(args.repo, args.revision, shard), start + c,
                                 min(chunk, nbytes - c), dst + c))

    fetched = [0]
    t0 = time.time()

    def work(job):
        key, fd, url, src, n, dst = job
        data = request(url, token, (src, src + n - 1))
        os.pwrite(fd, data, dst)
        with lock:
            done.add(key)
            fetched[0] += n
            if len(done) % 16 == 0:
                json.dump(sorted(done), open(progress_path, "w"))
                rate = fetched[0] / max(time.time() - t0, 1e-6) / 1e6
                print(f"{fetched[0] / 1e9:7.2f} GB fetched this run, {rate:6.1f} MB/s", flush=True)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        for f in cf.as_completed([ex.submit(work, j) for j in jobs]):
            f.result()
    json.dump(sorted(done), open(progress_path, "w"))

    weight_map, manifest = {}, {}
    for out_name, path, fd, header, base in files:
        os.fsync(fd)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            h = hashlib.sha256()
            pos = base + a
            while pos < base + b:
                n = min(64 << 20, base + b - pos)
                h.update(os.pread(fd, n, pos))
                pos += n
            src = next(t for t in plan if t[0] == name)
            weight_map[name] = out_name
            manifest[name] = {"file": out_name, "sha256": h.hexdigest(), "dtype": meta["dtype"],
                              "shape": meta["shape"], "source_file": src[1],
                              "source_absolute_offsets": [src[2], src[2] + src[3]]}
        os.close(fd)
    json.dump({"metadata": {"total_size": total, "source_repo": args.repo, "source_revision": args.revision,
                            "select": args.select}, "weight_map": weight_map},
              open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=1)
    json.dump(manifest, open(os.path.join(args.out, "fetch-manifest.json"), "w"), indent=1)
    print(f"done: {len(weight_map)} tensors, {total / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
