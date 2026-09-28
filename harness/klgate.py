#!/usr/bin/env python3
"""KL-divergence quality gate against published BF16 teacher logits (GLM-5.3-Flash).

The published GLM-5.3-Flash fidelity figures (EXL3 K4, K6, official FP8, NVFP4 KV cache) are
teacher-forced KL divergences against a sealed panel of full-vocabulary FP32 logits from the BF16
model (``brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits``). This tool computes the same number for
an engine that writes its own logits for the same token windows. ``docs/KL-GATE.md`` describes
the method, the sources and the engine interface.

Method (as the published measurements):
  - per scored position r of a window (row r = logits after tokens[0..r], predicting tokens[r+1]):
    d_r = KL(teacher || engine) = sum_v p(v) [ln p(v) - ln q(v)] in nats, with log-softmax and
    sums in float64 over the same vocabulary columns on both sides; by default every stored
    column (154,880, the published "full stored vocab" policy), or the tokenizer's 154,856 with
    ``--vocab tokenizer`` (the joint protocol's masked policy);
  - the headline is the token mean over all positions of all windows; top-1 agreement is the
    fraction of positions where both argmaxes agree. When only some rows of a window are scored,
    each row stands for the positions nearest to it, so the estimate is still of the whole panel's
    mean (the number the published figures are), with the standard error of that subsample;
  - uncertainty: a window-level block bootstrap (B = 5000, seed 20260829, percentile and BCa
    intervals) and the cluster-robust SE with windows as clusters.

Commands:
  plan      --teacher DIR --out PLAN.json   the engine's input: token ids and the rows to write
  score     --teacher DIR --engine DIR      the KL report (and the gate, with --max-mean)
  compare   A.json B.json                   paired difference of two score reports (--margin)
  canary    --teacher DIR                   the teacher against itself (exactly 0) and shifted
  selftest                                  synthetic checks of every computation above

Files: the teacher directory has the dataset's layout (``dataset-manifest.json``,
``calibration/panel-v1/arrays/<window>.tokens.npy``, and the logits either as the dataset's
``logits/window-NNNN.safetensors`` or as ``teacher-rows/<window>.safetensors`` subsets written by
``klgate_fetch.py``). The engine directory has one ``<window>.safetensors`` per window: ``logits``
F32 [R, V] and ``positions`` I32 [R], metadata ``window_id`` and ``tokens_sha256``.

Standard library only (tested with Python 3.12).
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
import random
import struct
import sys
import tempfile
from array import array
from itertools import repeat
from math import exp, fsum, isfinite, log, sqrt
from operator import mul, sub
from statistics import NormalDist

SCHEMA_PLAN = "glm53f-kl-plan.v1"
SCHEMA_REPORT = "glm53f-kl-report.v1"
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260829
ALIGN_BAND = (0.2, 0.995)  # teacher top-1 == realized next token, per window
SHIFT_RATIO_MIN = 3.0  # shifted self-KL must reach this multiple of the teacher's entropy
POSITION_BUCKETS = ((0, 256), (256, 1024), (1024, 1 << 30))
PERCENTILE_MIN_EXCEEDANCES = 100
Z95 = NormalDist().inv_cdf(0.975)


class GateError(Exception):
    pass


# ---------------------------------------------------------------------------------- file formats


class SafeTensors:
    """Read-only access to a safetensors file: metadata, integer vectors, F32 matrix rows."""

    DTYPES = {"F32": ("f", 4), "I32": ("i", 4), "I64": ("q", 8), "U32": ("I", 4)}

    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(self.fd).st_size
            (n,) = struct.unpack("<Q", os.pread(self.fd, 8, 0))
            if n > min(size - 8, 100 << 20):
                raise GateError(f"{path}: not a safetensors file")
            hdr = json.loads(os.pread(self.fd, n, 8))
            self.meta = hdr.pop("__metadata__", None) or {}
            self.tensors = hdr
            self.base = 8 + n
            for name, t in hdr.items():
                a, b = t["data_offsets"]
                if t["dtype"] not in self.DTYPES or not 0 <= a <= b <= size - self.base:
                    raise GateError(f"{path}: tensor {name} {t['dtype']} {t['data_offsets']}")
                if b - a != self.DTYPES[t["dtype"]][1] * math.prod(t["shape"]):
                    raise GateError(f"{path}: tensor {name} size does not match its shape")
        except BaseException:
            os.close(self.fd)
            raise

    def close(self) -> None:
        os.close(self.fd)

    def __enter__(self) -> "SafeTensors":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _read(self, name: str, offset: int, nbytes: int, code: str) -> array:
        a = array(code)
        a.frombytes(os.pread(self.fd, nbytes, self.base + self.tensors[name]["data_offsets"][0] + offset))
        if len(a) * a.itemsize != nbytes:
            raise GateError(f"{self.path}: short read of {name}")
        if sys.byteorder != "little":
            a.byteswap()
        return a

    def ints(self, name: str) -> list[int]:
        t = self.tensors[name]
        if t["dtype"] not in ("I32", "I64", "U32") or len(t["shape"]) != 1:
            raise GateError(f"{self.path}: {name} is {t['dtype']} {t['shape']}, expected a 1-D integer vector")
        code, size = self.DTYPES[t["dtype"]]
        return list(self._read(name, 0, size * t["shape"][0], code))

    def matrix(self, name: str) -> tuple[int, int]:
        t = self.tensors.get(name)
        if t is None or t["dtype"] != "F32" or len(t["shape"]) != 2:
            raise GateError(f"{self.path}: expected {name} F32 [rows, vocab], found {t}")
        return t["shape"][0], t["shape"][1]

    def row(self, name: str, i: int) -> array:
        _, cols = self.tensors[name]["shape"]
        return self._read(name, i * cols * 4, cols * 4, "f")


def read_npy_ints(path: str) -> list[int]:
    """A 1-D little-endian integer .npy file (the panel's token arrays)."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:6] != b"\x93NUMPY":
        raise GateError(f"{path}: not a .npy file")
    if raw[6] == 1:
        (hl,), off = struct.unpack("<H", raw[8:10]), 10
    else:
        (hl,), off = struct.unpack("<I", raw[8:12]), 12
    header = ast.literal_eval(raw[off : off + hl].decode("latin1"))
    code = {"<i4": "i", "<i8": "q", "<u4": "I"}.get(header["descr"])
    shape = header["shape"]
    if code is None or header["fortran_order"] or len(shape) != 1:
        raise GateError(f"{path}: unsupported array {header}")
    a = array(code)
    a.frombytes(raw[off + hl :])
    if sys.byteorder != "little":
        a.byteswap()
    if len(a) != shape[0]:
        raise GateError(f"{path}: {len(a)} values for shape {shape}")
    return list(a)


def tokens_sha256(tokens: list[int]) -> str:
    """The digest an engine reports for the ids it fed: little-endian u32 per token."""
    a = array("I", tokens)
    if sys.byteorder != "little":
        a.byteswap()
    return hashlib.sha256(a.tobytes()).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Teacher:
    """The teacher panel: windows, token ids and where each window's logits are."""

    def __init__(self, root: str):
        self.root = root
        path = os.path.join(root, "dataset-manifest.json")
        if not os.path.exists(path):
            raise GateError(f"{root}: no dataset-manifest.json")
        with open(path, "rb") as f:
            raw = f.read()
        self.manifest = json.loads(raw)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.stored_vocab = int(self.manifest["vocab_size"])
        self.windows = {w["window_id"]: w for w in self.manifest["logit_files"]}
        self.tokenizer_vocab = None
        receipt = os.path.join(root, "calibration/panel-v1/tokenizer.receipt.json")
        if os.path.exists(receipt):
            with open(receipt) as f:
                self.tokenizer_vocab = int(json.load(f)["vocab_size"])
        self.fetch = None
        fm = os.path.join(root, "FETCH-MANIFEST.json")
        if os.path.exists(fm):
            with open(fm) as f:
                d = json.load(f)
            self.fetch = {"repo": d.get("repo"), "revision": d.get("revision")}

    def ids(self, windows: str, exclude: str) -> list[str]:
        ids = list(self.windows)
        if windows:
            ids = windows.split(",")
            unknown = [w for w in ids if w not in self.windows]
            if unknown:
                raise GateError(f"unknown windows {unknown}")
        drop = set(exclude.split(",")) if exclude else set()
        return [w for w in ids if w not in drop]

    def tokens(self, wid: str) -> list[int]:
        w = self.windows[wid]
        path = os.path.join(self.root, "calibration/panel-v1/arrays", f"{wid}.tokens.npy")
        if sha256_file(path) != w["token_ids_sha256"]:
            raise GateError(f"{path}: sha256 differs from the manifest's token_ids_sha256")
        toks = read_npy_ints(path)
        if len(toks) != int(w["prediction_positions"]) + 1:
            raise GateError(f"{wid}: {len(toks)} tokens for {w['prediction_positions']} positions")
        return toks

    def logits(self, wid: str) -> tuple[str, list[int]]:
        """The window's teacher file and the positions its rows hold."""
        w = self.windows[wid]
        subset = os.path.join(self.root, "teacher-rows", f"{wid}.safetensors")
        full = os.path.join(self.root, w["path"])
        if os.path.exists(subset):
            with SafeTensors(subset) as st:
                m = st.meta
                if m.get("source_sha256") != w["sha256"] or m.get("token_ids_sha256") != w["token_ids_sha256"]:
                    raise GateError(f"{subset}: metadata does not match the dataset manifest")
                rows, _ = st.matrix("logits")
                pos = st.ints("positions")
            path = subset
        elif os.path.exists(full):
            if os.path.getsize(full) != int(w["bytes"]):
                raise GateError(f"{full}: size differs from the manifest")
            with SafeTensors(full) as st:
                if st.meta.get("token_ids_sha256") not in (None, w["token_ids_sha256"]):
                    raise GateError(f"{full}: token digest differs from the manifest")
                rows, _ = st.matrix("logits")
                pos = list(range(rows))
            path = full
        else:
            raise GateError(f"{wid}: no teacher logits under {self.root}")
        count = int(w["prediction_positions"])
        if len(pos) != rows or len(set(pos)) != rows or any(not 0 <= p < count for p in pos):
            raise GateError(f"{path}: positions are not distinct rows of 0..{count - 1}")
        return path, pos


# ---------------------------------------------------------------------------------- the kernel


def kl_row(t, s, pad_from: int | None = None) -> tuple:
    """KL(softmax(t) || softmax(s)) in float64, and the row's diagnostics.

    Returns (kl, teacher entropy, lse_t, lse_s, teacher argmax, engine argmax, teacher mass at
    columns >= pad_from). Argmax is the first maximum, as torch.argmax.
    """
    if len(t) != len(s):
        raise GateError(f"row widths differ: {len(t)} vs {len(s)}")
    if not (isfinite(sum(t)) and isfinite(sum(s))):  # any NaN or infinity
        raise GateError("non-finite logits")
    mt, ms = max(t), max(s)
    et = list(map(exp, map(sub, t, repeat(mt))))
    zt = fsum(et)
    zs = fsum(map(exp, map(sub, s, repeat(ms))))
    lse_t, lse_s = mt + log(zt), ms + log(zs)
    kl = fsum(map(mul, et, map(sub, t, s))) / zt - (lse_t - lse_s)
    ent = lse_t - fsum(map(mul, et, t)) / zt
    pad = fsum(et[pad_from:]) / zt if pad_from is not None and pad_from < len(t) else 0.0
    return kl, ent, lse_t, lse_s, t.index(mt), s.index(ms), pad


def score_window(job: dict) -> dict:
    """Score one window (a worker: every argument is plain data)."""
    wid, cols, pad_from = job["window_id"], job["cols"], job["pad_from"]
    toks = job["tokens"]
    out = {k: [] for k in ("pos", "kld", "agree", "ent", "lp_t", "lp_s", "t_hit")}
    pad_max = 0.0
    with SafeTensors(job["teacher_path"]) as ts, SafeTensors(job["engine_path"]) as es:
        _, tv = ts.matrix("logits")
        _, ev = es.matrix("logits")
        if tv < cols or ev < cols:
            raise GateError(f"{wid}: {cols} columns requested, teacher has {tv}, engine {ev}")
        erow = job["engine_rows"]
        shift = job.get("shift", 0)
        teacher_pos = job["teacher_positions"]
        for i, r in enumerate(teacher_pos):
            if shift:  # canary: score row i against row i + shift of the same file
                if i + shift >= len(teacher_pos):
                    break
                j = i + shift
            else:
                j = erow[r]
            t = ts.row("logits", i)
            s = es.row("logits", j)
            if tv > cols:
                t = t[:cols]
            if ev > cols:
                s = s[:cols]
            kl, ent, lse_t, lse_s, tt, st, pad = kl_row(t, s, pad_from)
            x = toks[r + 1]
            out["pos"].append(r)
            out["kld"].append(kl)
            out["agree"].append(1 if tt == st else 0)
            out["ent"].append(ent)
            out["lp_t"].append(t[x] - lse_t if x < cols else float("nan"))
            out["lp_s"].append(s[x] - lse_s if x < cols else float("nan"))
            out["t_hit"].append(1 if tt == x else 0)
            pad_max = max(pad_max, pad)
    out["window_id"] = wid
    out["pad_mass_max"] = pad_max
    return out


# ---------------------------------------------------------------------------------- statistics


def quantile(xs: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default)."""
    s = sorted(xs)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (h - lo)


def clustered_se(sums: list[float], counts: list[int]) -> float | None:
    """Cluster-robust SE of the token mean: sqrt(g/(g-1) sum_c (T_c - n_c mean)^2) / N."""
    g, n = len(sums), sum(counts)
    if g < 2:
        return None
    mean = fsum(sums) / n
    return sqrt(g / (g - 1) * fsum((t - c * mean) ** 2 for t, c in zip(sums, counts))) / n


def block_bootstrap(stats: list[tuple[float, ...]], counts: list[int], b: int, seed: int, alpha: float = 0.05) -> list[dict]:
    """Window-level bootstrap of ratio-of-sums means, with percentile and BCa intervals.

    ``stats[w]`` holds window w's sums (one per statistic); each statistic's estimate is
    sum over sampled windows of the sums / sum of their counts.
    """
    g, k = len(stats), len(stats[0])
    total = sum(counts)
    observed = [fsum(s[j] for s in stats) / total for j in range(k)]
    rng = random.Random(seed)
    reps: list[list[float]] = [[] for _ in range(k)]
    for _ in range(b):
        idx = [rng.randrange(g) for _ in range(g)]
        n = sum(counts[i] for i in idx)
        for j in range(k):
            reps[j].append(fsum(stats[i][j] for i in idx) / n)
    out = []
    nd = NormalDist()
    for j in range(k):
        bs = sorted(reps[j])
        pct = (quantile(bs, alpha / 2), quantile(bs, 1 - alpha / 2))
        bca = pct
        if bs[0] != bs[-1] and g > 2:
            prop = min(max(sum(1 for x in bs if x < observed[j]) / len(bs), 1e-9), 1 - 1e-9)
            z0 = nd.inv_cdf(prop)
            jack = []
            for drop in range(g):
                n = total - counts[drop]
                jack.append((fsum(s[j] for s in stats) - stats[drop][j]) / n)
            jbar = fsum(jack) / g
            num = fsum((jbar - x) ** 3 for x in jack)
            den = 6.0 * fsum((jbar - x) ** 2 for x in jack) ** 1.5
            acc = num / den if den > 0 else 0.0
            ends = []
            for z in (nd.inv_cdf(alpha / 2), nd.inv_cdf(1 - alpha / 2)):
                a2 = nd.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
                ends.append(quantile(bs, min(max(a2, 0.0), 1.0)))
            bca = (ends[0], ends[1])
        out.append({"observed": observed[j], "percentile95": pct, "bca95": bca})
    return out


def mean_sd(xs: list[float]) -> tuple[float, float | None]:
    m = fsum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    return m, sqrt(fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def weighted_quantile(xs: list[float], ws: list[float], q: float) -> float:
    """numpy's linear rule when the weights are equal, else the weighted inverse CDF."""
    if all(w == ws[0] for w in ws):
        return quantile(xs, q)
    pairs = sorted(zip(xs, ws))
    target, acc = q * fsum(ws), 0.0
    for x, w in pairs:
        acc += w
        if acc >= target:
            return x
    return pairs[-1][0]


def cells(positions: list[int], count: int) -> list[float]:
    """How many of a window's positions 0..count-1 each scored row stands for.

    Every position goes to the nearest scored row; a position halfway between two rows is split
    between them. When every position is scored each row stands for itself (1.0). For evenly
    spaced rows the cells are the strata of a systematic sample; rows in a fully scored run
    (the window's first positions, where the KL changes fastest) are exact.
    """
    order = sorted(range(len(positions)), key=positions.__getitem__)
    ps = [positions[i] for i in order]
    w = [1.0] * len(ps)
    w[0] += ps[0]
    w[-1] += count - 1 - ps[-1]
    for k in range(len(ps) - 1):
        half = (ps[k + 1] - ps[k] - 1) / 2
        w[k] += half
        w[k + 1] += half
    out = [0.0] * len(ps)
    for i, v in zip(order, w):
        out[i] = v
    return out


def window_total_var(kld: list[float], cell: list[float]) -> float | None:
    """Sampling variance of a window's cell-weighted KL total (exact rows contribute nothing).

    The rows that stand for more than themselves are treated as a simple random sample of the
    positions they stand for: N^2 (1 - n/N) s^2 / n (systematic sampling approximated as SRS).
    """
    xs = [k for k, c in zip(kld, cell) if c > 1.0]
    if not xs:
        return 0.0
    n, big_n = len(xs), fsum(c for c in cell if c > 1.0)
    sd = mean_sd(xs)[1]
    if sd is None:
        return None
    return big_n ** 2 * (1 - n / big_n) * sd ** 2 / n


def summarize(results: list[dict], teacher: Teacher, b: int, seed: int) -> dict:
    """Position-weighted panel statistics; with every position scored, exactly the published ones."""
    wins = []
    xs: list[float] = []
    ws: list[float] = []
    for res in results:
        w = teacher.windows[res["window_id"]]
        count = int(w["prediction_positions"])
        cell = cells(res["pos"], count)
        lp = [a - c for a, c in zip(res["lp_t"], res["lp_s"])]
        tot = fsum(c * k for c, k in zip(cell, res["kld"]))
        wins.append({
            "window_id": res["window_id"],
            "domain": w.get("domain", ""),
            "prediction_positions": count,
            "n": len(res["kld"]),
            "mean": tot / count,
            "row_mean": fsum(res["kld"]) / len(res["kld"]),
            "sd": mean_sd(res["kld"])[1],
            "top1_agreement": fsum(c * g for c, g in zip(cell, res["agree"])) / count,
            "teacher_top1_is_next_token": sum(res["t_hit"]) / len(res["t_hit"]),
            "teacher_entropy_mean": fsum(c * e for c, e in zip(cell, res["ent"])) / count,
            "ln_ppl_ratio": fsum(c * v for c, v in zip(cell, lp)) / count,
            "max": max(res["kld"]),
            "total_kld": tot,
            "total_agree": fsum(c * g for c, g in zip(cell, res["agree"])),
            "total_lp": fsum(c * v for c, v in zip(cell, lp)),
            "var_total": window_total_var(res["kld"], cell),
            "positions": res["pos"],
            "cells": cell,
            "kld": res["kld"],
            "agree": res["agree"],
        })
        xs.extend(res["kld"])
        ws.extend(cell)
    counts = [w["prediction_positions"] for w in wins]
    total_p, rows = sum(counts), len(xs)
    mean = fsum(w["total_kld"] for w in wins) / total_p
    var = [w["var_total"] for w in wins]
    se_sub = None if any(v is None for v in var) else sqrt(fsum(var)) / total_p
    se_c = clustered_se([w["total_kld"] for w in wins], counts)
    sd_rows = mean_sd(xs)[1]
    boot = block_bootstrap([(w["total_kld"], w["total_agree"], w["total_lp"]) for w in wins], counts, b, seed)
    quant = {}
    for q, name in ((0.5, "p50"), (0.9, "p90"), (0.95, "p95"), (0.99, "p99"), (0.999, "p999")):
        if rows * (1 - q) >= PERCENTILE_MIN_EXCEEDANCES or q == 0.5:
            quant[name] = weighted_quantile(xs, ws, q)
    domains = {}
    for d in sorted({w["domain"] for w in wins}):
        sel = [w for w in wins if w["domain"] == d]
        dp = sum(w["prediction_positions"] for w in sel)
        domains[d] = {
            "windows": len(sel),
            "rows": sum(w["n"] for w in sel),
            "mean": fsum(w["total_kld"] for w in sel) / dp,
            "top1_agreement": fsum(w["total_agree"] for w in sel) / dp,
        }
    buckets = {}
    for lo, hi in POSITION_BUCKETS:
        sel = [(c, k) for w in wins for p, c, k in zip(w["positions"], w["cells"], w["kld"]) if lo <= p < hi]
        if sel:
            name = f"{lo}-{hi if hi < 1 << 29 else 'end'}"
            buckets[name] = {"rows": len(sel), "mean": fsum(c * k for c, k in sel) / fsum(c for c, _ in sel)}
    return {
        "windows": len(wins),
        "scored_rows": rows,
        "panel_positions": total_p,
        "mean_kld": mean,
        "row_mean_kld": fsum(xs) / rows,
        "se_subset": se_sub,
        "se_clustered_window": se_c,
        "deff_window": (se_c / (sd_rows / sqrt(rows))) ** 2 if se_c and sd_rows else None,
        "bootstrap": {"b": b, "seed": seed, "rng": "python random.Random",
                      "mean_kld": boot[0], "top1_agreement": boot[1], "ln_ppl_ratio": boot[2]},
        "top1_agreement": fsum(w["total_agree"] for w in wins) / total_p,
        "ln_ppl_ratio": fsum(w["total_lp"] for w in wins) / total_p,
        "quantiles": quant,
        "max_kld": max(xs),
        "min_kld": min(xs),
        "per_domain": domains,
        "per_position_bucket": buckets,
        "per_window": wins,
    }


# ---------------------------------------------------------------------------------- commands


def _jobs(n: int | None) -> int:
    return max(1, n if n else min(8, os.cpu_count() or 1))


def _run_jobs(jobs: list[dict], workers: int) -> list[dict]:
    if workers == 1 or len(jobs) == 1:
        return [score_window(j) for j in jobs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(score_window, jobs))


def _columns(teacher: Teacher, policy: str) -> tuple[int, int | None]:
    """(columns scored, first padded column for the diagnostic)."""
    tok = teacher.tokenizer_vocab
    if policy == "stored":
        return teacher.stored_vocab, tok
    if policy == "tokenizer":
        if tok is None:
            raise GateError("--vocab tokenizer needs calibration/panel-v1/tokenizer.receipt.json")
        return tok, None
    if not policy.isdigit() or not 0 < int(policy) <= teacher.stored_vocab:
        raise GateError(f"--vocab {policy!r}: expected stored, tokenizer or a column count")
    cols = int(policy)
    return cols, tok if tok is not None and tok < cols else None


def cmd_plan(a) -> int:
    teacher = Teacher(a.teacher)
    windows = []
    for wid in teacher.ids(a.windows, a.exclude):
        toks = teacher.tokens(wid)
        _, pos = teacher.logits(wid)
        windows.append({
            "window_id": wid,
            "tokens_sha256": tokens_sha256(toks),
            "token_ids_npy_sha256": teacher.windows[wid]["token_ids_sha256"],
            "tokens": toks,
            "positions": sorted(pos),
        })
    plan = {
        "schema": SCHEMA_PLAN,
        "row_semantics": "row r = logits after tokens[0..r] (inclusive), predicting tokens[r+1]",
        "dataset_sha256": teacher.manifest.get("dataset_sha256"),
        "dataset_manifest_file_sha256": teacher.manifest_sha256,
        "teacher_model_revision": teacher.manifest.get("model_revision"),
        "vocab": teacher.stored_vocab,
        "windows": windows,
    }
    with open(a.out, "w") as f:
        json.dump(plan, f, separators=(",", ":"))
        f.write("\n")
    rows = sum(len(w["positions"]) for w in windows)
    print(f"plan: {len(windows)} windows, {rows} rows to write ({rows * teacher.stored_vocab * 4 / 1e9:.2f} GB of F32 logits) -> {a.out}")
    return 0


def _score(teacher: Teacher, engine_dir: str | None, ids: list[str], cols: int, pad_from, workers: int, shift: int = 0) -> tuple[list[dict], dict]:
    jobs, engine_meta = [], {}
    for wid in ids:
        toks = teacher.tokens(wid)
        tpath, tpos = teacher.logits(wid)
        if engine_dir is None:  # canary: the teacher is its own engine
            epath, erows = tpath, {p: i for i, p in enumerate(tpos)}
        else:
            epath = os.path.join(engine_dir, f"{wid}.safetensors")
            if not os.path.exists(epath):
                raise GateError(f"{wid}: no engine output {epath}")
            with SafeTensors(epath) as es:
                m = es.meta
                if m.get("window_id") != wid:
                    raise GateError(f"{epath}: window_id {m.get('window_id')!r}")
                if m.get("tokens_sha256") != tokens_sha256(toks):
                    raise GateError(f"{epath}: tokens_sha256 does not match the teacher's token ids")
                rows, _ = es.matrix("logits")
                epos = es.ints("positions") if "positions" in es.tensors else list(range(rows))
                engine_meta[wid] = m
            if len(epos) != rows or len(set(epos)) != rows:
                raise GateError(f"{epath}: {len(epos)} positions for {rows} rows, or repeated positions")
            erows = {p: i for i, p in enumerate(epos)}
            missing = [p for p in tpos if p not in erows]
            if missing:
                raise GateError(f"{wid}: the engine output lacks {len(missing)} teacher rows, e.g. {missing[:5]}")
        jobs.append({
            "window_id": wid, "tokens": toks, "teacher_path": tpath, "teacher_positions": tpos,
            "engine_path": epath, "engine_rows": erows, "cols": cols, "pad_from": pad_from, "shift": shift,
        })
    return _run_jobs(jobs, workers), engine_meta


def _check_alignment(results: list[dict]) -> list[str]:
    bad = []
    for res in results:
        hit = sum(res["t_hit"]) / max(len(res["t_hit"]), 1)
        if not ALIGN_BAND[0] <= hit <= ALIGN_BAND[1]:
            bad.append(f"{res['window_id']}: teacher top-1 equals the next token at {hit:.3f} of positions")
    return bad


def _fmt(x, digits=5):
    return "n/a" if x is None else f"{x:.{digits}f}"


def print_report(rep: dict) -> None:
    s = rep["summary"]
    boot = s["bootstrap"]
    subset = s["scored_rows"] < s["panel_positions"]
    print(f"KL(teacher || engine), nats; vocabulary: {rep['vocab_policy']} ({rep['columns']:,} columns)")
    print(f"windows {s['windows']}, scored rows {s['scored_rows']:,} standing for {s['panel_positions']:,} positions")
    line = f"  mean KLD                    {s['mean_kld']:.6f}"
    if subset:
        se = s["se_subset"]
        line += f"  (position-weighted; SE of the subsample {_fmt(se)}"
        if se is not None:
            line += f", 95% [{s['mean_kld'] - Z95 * se:.5f}, {s['mean_kld'] + Z95 * se:.5f}]"
        line += f"; unweighted row mean {s['row_mean_kld']:.6f})"
    print(line)
    b = boot["mean_kld"]
    print(f"  window bootstrap 95%        percentile [{b['percentile95'][0]:.5f}, {b['percentile95'][1]:.5f}]  "
          f"BCa [{b['bca95'][0]:.5f}, {b['bca95'][1]:.5f}]  (B={boot['b']}, seed {boot['seed']})")
    print(f"  clustered SE (windows)      {_fmt(s['se_clustered_window'])}  design effect {_fmt(s['deff_window'], 1)}")
    t = boot["top1_agreement"]
    print(f"  top-1 agreement             {s['top1_agreement']:.4f}  window bootstrap 95% "
          f"[{t['percentile95'][0]:.4f}, {t['percentile95'][1]:.4f}]")
    print(f"  ln(PPL_engine / PPL_teacher) {s['ln_ppl_ratio']:.5f}")
    q = s["quantiles"]
    print("  quantiles                   " + "  ".join(f"{k} {v:.5f}" for k, v in q.items()) + f"  max {s['max_kld']:.4f}")
    if rep.get("teacher_pad_mass_max") is not None:
        print(f"  teacher probability on the padded columns, max over rows: {rep['teacher_pad_mass_max']:.3e}")
    print("  per domain: " + "; ".join(f"{d} {v['mean']:.5f} (top-1 {v['top1_agreement']:.3f})" for d, v in s["per_domain"].items()))
    print("  per position: " + "; ".join(f"{k} {v['mean']:.5f} ({v['rows']} rows)" for k, v in s["per_position_bucket"].items()))
    print("  per window: window  domain  rows  mean  sd  top-1  teacher-top1=next")
    for w in s["per_window"]:
        print(f"    {w['window_id']}  {w['domain']:<28} {w['n']:5d}  {w['mean']:.5f}  {_fmt(w['sd'], 4)}  "
              f"{w['top1_agreement']:.3f}  {w['teacher_top1_is_next_token']:.3f}")
    if rep.get("gate"):
        g = rep["gate"]
        print(f"gate: {'PASS' if g['passed'] else 'FAIL'}: " + "; ".join(g["reasons"]))


def cmd_score(a) -> int:
    teacher = Teacher(a.teacher)
    ids = teacher.ids(a.windows, a.exclude)
    cols, pad_from = _columns(teacher, a.vocab)
    results, engine_meta = _score(teacher, a.engine, ids, cols, pad_from, _jobs(a.jobs))
    bad = _check_alignment(results)
    if bad:
        raise GateError("teacher rows look misaligned (alignment canary):\n  " + "\n  ".join(bad))
    summary = summarize(results, teacher, a.bootstrap, a.seed)
    pad = max(r["pad_mass_max"] for r in results) if pad_from is not None else None
    rep = {
        "schema": SCHEMA_REPORT,
        "direction": "KL(teacher || engine), nats",
        "estimator": ("float64 log-softmax and sums over the same columns on both sides; mean over the panel's "
                      "positions, each scored row weighted by the positions it stands for (1 when all are scored)"),
        "vocab_policy": a.vocab,
        "columns": cols,
        "teacher": {
            "dataset_sha256": teacher.manifest.get("dataset_sha256"),
            "dataset_manifest_file_sha256": teacher.manifest_sha256,
            "model_revision": teacher.manifest.get("model_revision"),
            "fetch": teacher.fetch,
        },
        "engine": {wid: m for wid, m in sorted(engine_meta.items())},
        "harness_sha256": sha256_file(os.path.abspath(__file__)),
        "teacher_pad_mass_max": pad,
        "summary": summary,
    }
    reasons, passed = [], True
    if a.max_mean is not None:
        se = summary["se_subset"]
        if se is None:
            raise GateError("--max-mean needs the SE of the subsample (every window needs two sampled rows)")
        upper = summary["mean_kld"] + Z95 * se
        ok = upper < a.max_mean
        passed &= ok
        reasons.append(f"mean {summary['mean_kld']:.5f} + 1.96 SE = {upper:.5f} {'<' if ok else '>='} {a.max_mean}")
    if a.min_top1 is not None:
        ok = summary["top1_agreement"] >= a.min_top1
        passed &= ok
        reasons.append(f"top-1 {summary['top1_agreement']:.4f} {'>=' if ok else '<'} {a.min_top1}")
    if reasons:
        rep["gate"] = {"passed": passed, "reasons": reasons}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
            f.write("\n")
    print_report(rep)
    return 0 if passed else 3


def cmd_canary(a) -> int:
    teacher = Teacher(a.teacher)
    ids = teacher.ids(a.windows, a.exclude)
    cols, pad_from = _columns(teacher, a.vocab)
    workers = _jobs(a.jobs)
    same, _ = _score(teacher, None, ids, cols, pad_from, workers)
    nonzero = sum(1 for r in same for k in r["kld"] if k != 0.0)
    rows = sum(len(r["kld"]) for r in same)
    ent = fsum(e for r in same for e in r["ent"]) / rows
    shifted, _ = _score(teacher, None, ids, cols, pad_from, workers, shift=1)
    sk = [k for r in shifted for k in r["kld"]]
    smean = fsum(sk) / len(sk)
    bad = _check_alignment(same)
    print(f"self: KL(teacher || teacher) is exactly 0 at {rows - nonzero} of {rows} positions")
    if pad_from is not None:
        print(f"padding: teacher probability on columns {pad_from} and above, max over rows "
              f"{max(r['pad_mass_max'] for r in same):.3e}")
    print(f"shift: KL(row i || row i+1 of the same window) mean {smean:.4f} nats; teacher entropy mean "
          f"{ent:.4f}; ratio {smean / ent:.1f} (needs >= {SHIFT_RATIO_MIN})")
    for msg in bad:
        print(f"alignment: {msg}")
    ok = nonzero == 0 and smean >= SHIFT_RATIO_MIN * ent and not bad
    print(f"canary: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 3


def _pairs(rep: dict) -> dict[tuple[str, int], tuple[float, int, float]]:
    out = {}
    for w in rep["summary"]["per_window"]:
        for p, k, g, c in zip(w["positions"], w["kld"], w["agree"], w["cells"]):
            out[(w["window_id"], p)] = (k, g, c)
    return out


def cmd_compare(a) -> int:
    """Paired difference A - B over identical rows, position-weighted as in score."""
    with open(a.a) as f:
        ra = json.load(f)
    with open(a.b) as f:
        rb = json.load(f)
    for key in ("vocab_policy", "columns"):
        if ra[key] != rb[key]:
            raise GateError(f"the reports differ in {key}: {ra[key]} vs {rb[key]}")
    if ra["teacher"]["dataset_sha256"] != rb["teacher"]["dataset_sha256"]:
        raise GateError("the reports are against different teacher panels")
    pa, pb = _pairs(ra), _pairs(rb)
    if set(pa) != set(pb):
        raise GateError(f"the reports score different rows ({len(pa)} vs {len(pb)}, {len(set(pa) ^ set(pb))} differ)")
    wins = sorted({w for w, _ in pa})
    stats, counts, xs, ys = [], [], [], []
    a_only = b_only = 0
    for wid in wins:
        keys = sorted(k for k in pa if k[0] == wid)
        stats.append((fsum(pa[k][2] * (pa[k][0] - pb[k][0]) for k in keys),
                      fsum(pa[k][2] * pa[k][0] for k in keys), fsum(pa[k][2] * pb[k][0] for k in keys)))
        counts.append(round(fsum(pa[k][2] for k in keys)))
        xs.extend(pa[k][0] for k in keys)
        ys.extend(pb[k][0] for k in keys)
        a_only += sum(1 for k in keys if pa[k][1] and not pb[k][1])
        b_only += sum(1 for k in keys if pb[k][1] and not pa[k][1])
    total = sum(counts)
    boot = block_bootstrap(stats, counts, a.bootstrap, a.seed)
    mean_d, mean_a, mean_b = (fsum(s[j] for s in stats) / total for j in range(3))
    ratio_reps = []
    rng = random.Random(a.seed)
    for _ in range(a.bootstrap):
        idx = [rng.randrange(len(wins)) for _ in wins]
        den = fsum(stats[i][2] for i in idx)
        if den > 0:
            ratio_reps.append(fsum(stats[i][1] for i in idx) / den)
    mx, my = fsum(xs) / len(xs), fsum(ys) / len(ys)
    cov = fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = fsum((x - mx) ** 2 for x in xs) * fsum((y - my) ** 2 for y in ys)
    rho = cov / sqrt(var) if var > 0 else None
    chi2 = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only) if a_only + b_only else 0.0
    p_mcnemar = math.erfc(sqrt(chi2 / 2)) if a_only + b_only else 1.0
    se_c = clustered_se([s[0] for s in stats], counts)
    lo, hi = boot[0]["percentile95"]
    print(f"paired over {len(xs):,} rows in {len(wins)} windows: A {mean_a:.6f}, B {mean_b:.6f}")
    print(f"  mean difference A - B       {mean_d:+.6f}  window bootstrap 95% [{lo:+.6f}, {hi:+.6f}]  "
          f"clustered SE {_fmt(se_c, 6)}")
    if ratio_reps and mean_b > 0:
        print(f"  ratio A / B                 {mean_a / mean_b:.4f}  95% "
              f"[{quantile(ratio_reps, 0.025):.4f}, {quantile(ratio_reps, 0.975):.4f}]")
    print(f"  per-row correlation         {_fmt(rho, 4)}")
    print(f"  top-1 (rows): A agrees and B not {a_only}, B agrees and A not {b_only}; McNemar p {p_mcnemar:.3g}")
    if a.margin is None:
        return 0
    ok = hi < a.margin
    print(f"gate: {'PASS' if ok else 'FAIL'}: upper 95% bound of A - B {hi:+.6f} {'<' if ok else '>='} margin {a.margin}")
    return 0 if ok else 3


# ---------------------------------------------------------------------------------- self-test


def _write_st(path: str, tensors: dict[str, tuple[str, list[int], bytes]], meta: dict[str, str]) -> None:
    obj: dict[str, object] = {"__metadata__": meta}
    off = 0
    for name, (dtype, shape, data) in tensors.items():
        obj[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    text = json.dumps(obj).encode()
    text += b" " * (-(8 + len(text)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text)
        for _, _, data in tensors.values():
            f.write(data)


def _f32(xs) -> bytes:
    a = array("f", xs)
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def _i32(xs) -> bytes:
    a = array("i", xs)
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def _npy(path: str, xs: list[int]) -> None:
    header = "{'descr': '<i4', 'fortran_order': False, 'shape': (%d,), }" % len(xs)
    header += " " * (-(10 + len(header) + 1) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + _i32(xs))


def _panel(root: str, rng: random.Random, windows: int, count: int, vocab: int, tok_vocab: int, subset: dict[str, list[int]]) -> dict:
    """A synthetic panel in the dataset's layout; teacher logits in float32."""
    arrays = os.path.join(root, "calibration/panel-v1/arrays")
    os.makedirs(arrays)
    os.makedirs(os.path.join(root, "logits"))
    os.makedirs(os.path.join(root, "teacher-rows"))
    entries, logits, tokens = [], {}, {}
    for k in range(windows):
        wid = f"final-{k:04d}"
        rows = []
        toks = [rng.randrange(tok_vocab) for _ in range(count + 1)]
        for r in range(count):
            row = [rng.gauss(0, 2) for _ in range(vocab)]
            row[toks[r + 1]] += 9.0 if r % 4 else 0.0  # the next token is usually the argmax
            for c in range(tok_vocab, vocab):
                row[c] = -3.0
            rows.append(list(array("f", row)))  # float32-rounded
        npy = os.path.join(arrays, f"{wid}.tokens.npy")
        _npy(npy, toks)
        path = f"logits/window-{k:04d}.safetensors"
        entry = {"window_id": wid, "path": path, "role": "final", "domain": f"d{k % 2}",
                 "prediction_positions": count, "token_ids_sha256": sha256_file(npy)}
        meta = {"window_id": wid, "token_ids_sha256": entry["token_ids_sha256"]}
        if wid in subset:
            pos = subset[wid]
            # the full file is absent; only the subset exists, as klgate_fetch.py writes it
            _write_st(os.path.join(root, "teacher-rows", f"{wid}.safetensors"),
                      {"positions": ("I32", [len(pos)], _i32(pos)),
                       "logits": ("F32", [len(pos), vocab], b"".join(_f32(rows[p]) for p in pos))},
                      {**meta, "source_sha256": "0" * 64})
            entry["sha256"] = "0" * 64
            entry["bytes"] = 0
        else:
            full = os.path.join(root, path)
            _write_st(full, {"logits": ("F32", [count, vocab], b"".join(_f32(r) for r in rows))}, meta)
            entry["sha256"] = sha256_file(full)
            entry["bytes"] = os.path.getsize(full)
        entries.append(entry)
        logits[wid], tokens[wid] = rows, toks
    manifest = {"dataset_sha256": "synthetic", "model_revision": "synthetic", "vocab_size": vocab,
                "logit_files": entries}
    with open(os.path.join(root, "dataset-manifest.json"), "w") as f:
        json.dump(manifest, f)
    with open(os.path.join(root, "calibration/panel-v1/tokenizer.receipt.json"), "w") as f:
        json.dump({"vocab_size": tok_vocab}, f)
    return {"logits": logits, "tokens": tokens}


def _engine(root: str, panel: dict, teacher: Teacher, perturb, positions=None, drop=None, meta_extra=None) -> None:
    os.makedirs(root, exist_ok=True)
    for wid, rows in panel["logits"].items():
        pos = list(range(len(rows))) if positions is None else positions[wid]
        if drop:
            pos = [p for p in pos if p not in drop]
        data = b"".join(_f32(perturb(wid, p, rows[p])) for p in pos)
        meta = {"window_id": wid, "tokens_sha256": tokens_sha256(panel["tokens"][wid]), **(meta_extra or {})}
        _write_st(os.path.join(root, f"{wid}.safetensors"),
                  {"positions": ("I32", [len(pos)], _i32(pos)), "logits": ("F32", [len(pos), len(rows[0])], data)}, meta)


def _direct_kl(t: list[float], s: list[float]) -> float:
    """An independent two-pass reference: p, log p, log q explicitly."""
    mt, ms = max(t), max(s)
    lt = mt + log(fsum(exp(x - mt) for x in t))
    ls = ms + log(fsum(exp(x - ms) for x in s))
    return fsum(exp(a - lt) * ((a - lt) - (b - ls)) for a, b in zip(t, s))


def selftest() -> int:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")

    rng = random.Random(1)
    print("kernel")
    t = array("f", [rng.gauss(0, 3) for _ in range(1000)])
    r = kl_row(t, t)
    check("KL(p || p) is exactly 0", r[0] == 0.0, repr(r[0]))
    e2 = math.e ** 2
    want = 2 * e2 / (e2 + 3) - log(e2 + 3) + log(4)  # p = softmax([2,0,0,0]), q uniform
    got = kl_row(array("f", [2, 0, 0, 0]), array("f", [0, 0, 0, 0]))[0]
    check("known pair: KL(softmax[2,0,0,0] || uniform) = 2e^2/(e^2+3) - ln(e^2+3) + ln 4", abs(got - want) < 1e-15,
          f"{got!r} vs {want!r}")
    want_rev = log(e2 + 3) - log(4) - 0.5  # the other direction, to catch a swapped direction
    got_rev = kl_row(array("f", [0, 0, 0, 0]), array("f", [2, 0, 0, 0]))[0]
    check("direction: KL(uniform || softmax[2,0,0,0]) = ln(e^2+3) - ln 4 - 1/2", abs(got_rev - want_rev) < 1e-15,
          f"{got_rev!r} vs {want_rev!r}")
    s = array("f", [x + 7.25 for x in t])
    check("a constant shift of the logits gives 0", abs(kl_row(t, s)[0]) < 1e-13, repr(kl_row(t, s)[0]))
    big = array("f", [x + 3.0e4 for x in t])
    check("large logits (3e4) do not overflow", abs(kl_row(big, big)[0]) == 0.0 and isfinite(kl_row(t, big)[0]))
    eps = 1e-3
    delta = [rng.gauss(0, 1) for _ in t]
    s2 = [a + eps * d for a, d in zip(t, delta)]
    p = [exp(x - kl_row(t, t)[2]) for x in t]
    md = fsum(pi * d for pi, d in zip(p, delta))
    quad = 0.5 * eps * eps * fsum(pi * (d - md) ** 2 for pi, d in zip(p, delta))
    got_q = kl_row(list(t), s2)[0]
    check("small perturbations: KL ~ Var_p(delta)/2", abs(got_q / quad - 1) < 1e-2, f"{got_q:.4e} vs {quad:.4e}")
    s3 = array("f", [rng.gauss(0, 3) for _ in range(1000)])
    check("agrees with an explicit two-pass computation", abs(kl_row(t, s3)[0] - _direct_kl(list(t), list(s3))) < 1e-12)
    tp = array("f", [1.0, 0.5, -1.0, 0.0, -30.0, -30.0])  # two padded columns at the end
    sp = array("f", [0.5, 0.5, -0.5, 0.0, 4.0, 4.0])  # the engine puts mass on them
    full = kl_row(tp, sp, 4)
    masked = kl_row(tp[:4], sp[:4])[0]
    check("stored-vocab policy counts padded columns", abs(full[0] - _direct_kl(list(tp), list(sp))) < 1e-15)
    check("tokenizer policy drops them on both sides", abs(masked - _direct_kl(list(tp[:4]), list(sp[:4]))) < 1e-15
          and masked < full[0])
    check("padded-mass diagnostic", abs(full[6] - 2 * exp(-30.0 - full[2])) < 1e-25)
    check("argmax is the first maximum", kl_row(array("f", [1, 3, 3]), array("f", [3, 1, 3]))[4:6] == (1, 0))
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            kl_row(array("f", [0.0, bad]), array("f", [0.0, 0.0]))
            check(f"refuses {bad} logits", False)
        except GateError:
            check(f"refuses {bad} logits", True)

    print("statistics")
    check("quantile matches numpy's linear rule", quantile([1, 2, 3, 4], 0.25) == 1.75 and quantile([5], 0.9) == 5)
    se = clustered_se([1.0, 2.0, 6.0], [1, 1, 2])  # mean 9/4; residuals -1.25, -0.25, 1.5
    check("clustered SE by hand", abs(se - sqrt(1.5 * (1.5625 + 0.0625 + 2.25)) / 4) < 1e-15, repr(se))
    stats = [(float(k), 1.0) for k in range(10)]
    bs = block_bootstrap(stats, [1] * 10, 2000, 7)
    lo, hi = bs[0]["percentile95"]
    check("bootstrap interval covers the mean", lo < 4.5 < hi and bs[0]["observed"] == 4.5, f"[{lo}, {hi}]")
    check("bootstrap is reproducible with its seed", block_bootstrap(stats, [1] * 10, 500, 3) == block_bootstrap(stats, [1] * 10, 500, 3))
    lo2, hi2 = bs[1]["bca95"]
    check("a constant statistic has a zero-width interval", lo2 == hi2 == 1.0)
    check("cells: every position scored -> 1 each", cells([3, 0, 2, 1], 4) == [1.0] * 4)
    check("cells: nearest row, ties split", cells([1, 4, 8], 10) == [3.0, 3.5, 3.5])
    check("cells: a fully scored head is exact", cells([0, 1, 2, 3, 4, 5, 16], 20)[:6] == [1.0] * 5 + [6.0])
    check("sampling variance: exact rows add nothing", window_total_var([0.3, 0.1, 0.2], [1.0, 1.0, 1.0]) == 0.0)
    v = window_total_var([0.1, 0.3], [5.0, 5.0])  # N 10, n 2, s^2 0.02: 100 * 0.8 * 0.02 / 2
    check("sampling variance by hand", abs(v - 0.8) < 1e-15, repr(v))
    steep = [1.0 / (r + 1) ** 0.8 + 0.02 for r in range(2047)]  # a KL profile falling fast at the head
    truth = fsum(steep) / 2047
    spaced = [((2 * i + 1) * 2047) // (2 * 184) for i in range(184)]

    def estimate(pos):
        c = cells(pos, 2047)
        return fsum(ci * steep[p] for ci, p in zip(c, pos)) / 2047

    err_plain, err_head = estimate(spaced) / truth - 1, estimate(sorted(set(spaced) | set(range(5)))) / truth - 1
    check("a steep head: scoring positions 0-4 exactly removes most of the subsample's bias",
          abs(err_head) < abs(err_plain) / 3, f"relative error {err_plain:+.4f} without, {err_head:+.4f} with")

    print("end to end (synthetic panel, 3 windows x 12 positions, 40 columns, 36 real)")
    with tempfile.TemporaryDirectory() as tmp:
        troot = os.path.join(tmp, "teacher")
        sub = {"final-0002": [1, 4, 7, 10]}
        panel = _panel(troot, random.Random(2), 3, 12, 40, 36, sub)
        teacher = Teacher(troot)
        positions = {wid: (sub[wid] if wid in sub else list(range(12))) for wid in panel["logits"]}

        def args(**kw):
            base = dict(teacher=troot, engine=None, windows="", exclude="", vocab="stored", jobs=1,
                        bootstrap=200, seed=BOOTSTRAP_SEED, json=None, max_mean=None, min_top1=None)
            base.update(kw)
            return argparse.Namespace(**base)

        plan_path = os.path.join(tmp, "plan.json")
        cmd_plan(argparse.Namespace(teacher=troot, windows="", exclude="", out=plan_path))
        with open(plan_path) as f:
            plan = json.load(f)
        pw = {w["window_id"]: w for w in plan["windows"]}
        check("plan: tokens, their digest and the teacher's rows",
              all(pw[wid]["tokens"] == panel["tokens"][wid] and pw[wid]["positions"] == positions[wid]
                  and pw[wid]["tokens_sha256"] == tokens_sha256(panel["tokens"][wid]) for wid in pw))

        same = os.path.join(tmp, "engine-same")
        _engine(same, panel, teacher, lambda wid, p, row: row)
        rep_path = os.path.join(tmp, "same.json")
        rc = cmd_score(args(engine=same, json=rep_path, max_mean=1e-12))
        with open(rep_path) as f:
            rep = json.load(f)
        sm = rep["summary"]
        check("engine = teacher: every KL exactly 0, top-1 1.0, gate passes",
              rc == 0 and all(k == 0.0 for w in sm["per_window"] for k in w["kld"]) and sm["top1_agreement"] == 1.0)
        check("subset rows are scored only where the teacher has them",
              sm["scored_rows"] == 12 + 12 + 4 and sm["panel_positions"] == 36)

        noise = random.Random(3)
        pert = {}

        def perturb(wid, p, row):
            out = [x + noise.gauss(0, 0.3) for x in row]
            pert[(wid, p)] = list(array("f", out))
            return out

        eng = os.path.join(tmp, "engine-noisy")
        _engine(eng, panel, teacher, perturb, positions)
        rep2_path = os.path.join(tmp, "noisy.json")
        rc2 = cmd_score(args(engine=eng, json=rep2_path, max_mean=1e-6))
        with open(rep2_path) as f:
            rep2 = json.load(f)
        # every row of the full windows stands for itself; final-0002's rows 1, 4, 7, 10 stand for 3 each
        weight = {wid: (3.0 if wid in sub else 1.0) for wid in positions}
        direct = [weight[wid] * _direct_kl(panel["logits"][wid][p], pert[(wid, p)]) for wid in sorted(positions) for p in positions[wid]]
        mean_direct = fsum(direct) / 36
        check("mean KLD equals an independent computation", abs(rep2["summary"]["mean_kld"] - mean_direct) < 1e-13,
              f"{rep2['summary']['mean_kld']!r} vs {mean_direct!r}")
        check("a gate above the measured mean fails", rc2 == 3)
        rep3_path = os.path.join(tmp, "noisy-masked.json")
        cmd_score(args(engine=eng, json=rep3_path, vocab="tokenizer", jobs=2))
        with open(rep3_path) as f:
            rep3 = json.load(f)
        direct_m = [weight[wid] * _direct_kl(panel["logits"][wid][p][:36], pert[(wid, p)][:36])
                    for wid in sorted(positions) for p in positions[wid]]
        check("tokenizer policy (and 2 worker processes) equals an independent computation",
              abs(rep3["summary"]["mean_kld"] - fsum(direct_m) / 36) < 1e-13)
        rc4 = cmd_compare(argparse.Namespace(a=rep2_path, b=rep2_path, margin=1e-9, bootstrap=200, seed=1))
        check("compare: a report against itself differs by exactly 0 and passes", rc4 == 0)
        rc5 = cmd_compare(argparse.Namespace(a=rep2_path, b=rep_path, margin=1e-6, bootstrap=200, seed=1))
        check("compare: noisy against exact fails a tight margin", rc5 == 3)
        rc6 = cmd_canary(argparse.Namespace(teacher=troot, windows="", exclude="", vocab="stored", jobs=1))
        check("canary on the synthetic teacher passes", rc6 == 0)

        for name, kwargs in (("a missing teacher row", {"drop": {4}}),
                             ("a wrong token digest", {"meta_extra": {"tokens_sha256": "0" * 64}})):
            bad = os.path.join(tmp, "engine-bad-" + name.replace(" ", "-"))
            _engine(bad, panel, teacher, lambda wid, p, row: row, positions, **kwargs)
            try:
                cmd_score(args(engine=bad))
                check(f"refuses {name}", False)
            except GateError as e:
                check(f"refuses {name}", True, str(e).split(":")[-1].strip()[:60])
        shifted = os.path.join(tmp, "engine-shifted")
        _engine(shifted, panel, teacher, lambda wid, p, row: panel["logits"][wid][min(p + 1, 11)], positions)
        rep7_path = os.path.join(tmp, "shifted.json")
        cmd_score(args(engine=shifted, json=rep7_path))
        with open(rep7_path) as f:
            rep7 = json.load(f)
        ent = fsum(w["teacher_entropy_mean"] * w["prediction_positions"] for w in rep7["summary"]["per_window"]) / 36
        check("an off-by-one engine scores at entropy scale", rep7["summary"]["mean_kld"] > SHIFT_RATIO_MIN * ent,
              f"{rep7['summary']['mean_kld']:.3f} vs entropy {ent:.3f}")

    ok = all(checks)
    print(f"selftest: {sum(checks)}/{len(checks)} checks passed" + ("" if ok else " -- FAILED"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p, engine: bool) -> None:
        p.add_argument("--teacher", required=True, help="teacher panel directory")
        if engine:
            p.add_argument("--engine", required=True, help="directory of <window>.safetensors engine outputs")
        p.add_argument("--windows", default="", help="comma-separated window ids (default: all)")
        p.add_argument("--exclude", default="", help="comma-separated window ids to leave out")
        p.add_argument("--vocab", default="stored",
                       help="columns scored: stored (all, default), tokenizer (drop padded columns), or a number")
        p.add_argument("--jobs", type=int, default=0, help="worker processes (default min(8, CPUs))")

    p = sp.add_parser("plan", help="write the engine's input")
    p.add_argument("--teacher", required=True)
    p.add_argument("--windows", default="")
    p.add_argument("--exclude", default="")
    p.add_argument("--out", required=True)
    p = sp.add_parser("score", help="score engine outputs against the teacher")
    common(p, True)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    p.add_argument("--json", help="write the full report here (per-position values included)")
    p.add_argument("--max-mean", type=float, help="gate: panel estimate + 1.96 SE of the subsample must be below this")
    p.add_argument("--min-top1", type=float, help="gate: top-1 agreement must be at least this")
    p = sp.add_parser("compare", help="paired comparison of two score reports (A = candidate, B = baseline)")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--margin", type=float, help="gate: upper 95%% bound of mean(A - B) must be below this")
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    p.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    p = sp.add_parser("canary", help="teacher against itself (exactly 0) and against its next row")
    common(p, False)
    sp.add_parser("selftest", help="synthetic checks")
    a = ap.parse_args()
    try:
        if a.cmd == "plan":
            return cmd_plan(a)
        if a.cmd == "score":
            return cmd_score(a)
        if a.cmd == "compare":
            return cmd_compare(a)
        if a.cmd == "canary":
            return cmd_canary(a)
        return selftest()
    except GateError as e:
        print(f"klgate: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
