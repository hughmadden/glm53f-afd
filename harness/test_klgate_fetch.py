#!/usr/bin/env python3
"""Self-test of harness/klgate_fetch.py against a dataset in memory (no network).

A fake Hub serves a small dataset in the real one's layout: final windows in dataset-manifest.json,
windows of four other roles in logits/full-panel/full-panel-manifest.json, token arrays and the
receipts; a listing at one revision, whole files, resolve redirects that name the linked file,
and byte ranges. The tool must fetch exactly the rows of the panel it is asked for, copy the rows
an earlier fetch holds instead of fetching them, record the panel in FETCH-MANIFEST.json, and leave
a directory that klgate.py plans and checks in the panel's order.

    python3 harness/test_klgate_fetch.py      (standard library only)
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pathlib
import random
import shutil
import struct
import sys
import tempfile
import unittest
import urllib.parse
from array import array

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import klgate  # noqa: E402
import klgate_fetch as kf  # noqa: E402

REV = "1" * 40
ENDPOINT = "https://hub.test"
V, COUNT = 8, 12  # columns, positions per window
ROWS = [0, 1, 4, 7, 10]  # --head-rows 2 --rows-per-window 4 of 12 positions
WINDOWS = ([("final", k) for k in range(3)] + [("confirmation", k) for k in range(4)]
           + [("selection", k) for k in range(4)] + [("fit", k) for k in range(2)])


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def npy(xs: list[int]) -> bytes:
    header = "{'descr': '<i4', 'fortran_order': False, 'shape': (%d,), }" % len(xs)
    header += " " * (-(10 + len(header) + 1) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + array("i", xs).tobytes()


def dataset(rng: random.Random) -> tuple[dict[str, bytes], dict[str, list[bytes]]]:
    """The files of a small dataset, and each window's rows as stored (F32 bytes)."""
    files: dict[str, bytes] = {}
    rows_of: dict[str, list[bytes]] = {}
    finals, others = [], []
    for role, k in WINDOWS:
        wid = f"{role}-{k:04d}"
        toks = [rng.randrange(V) for _ in range(COUNT + 1)]
        arr = npy(toks)
        rows = [array("f", [rng.gauss(0, 1) + (6.0 if c == toks[r + 1] and r % 4 else 0.0) for c in range(V)]).tobytes()
                for r in range(COUNT)]
        meta = {"window_id": wid, "token_ids_sha256": sha256(arr), "capture_role": "bf16_teacher"}
        hdr = json.dumps({"__metadata__": meta, "logits": {"dtype": "F32", "shape": [COUNT, V],
                                                           "data_offsets": [0, COUNT * V * 4]}}).encode()
        hdr += b" " * (-(8 + len(hdr)) % 8)
        body = struct.pack("<Q", len(hdr)) + hdr + b"".join(rows)
        path = f"logits/window-{k:04d}.safetensors" if role == "final" else f"logits/full-panel/{role}/{wid}.safetensors"
        files[path] = body
        files[f"calibration/panel-v1/arrays/{wid}.tokens.npy"] = arr
        rows_of[wid] = rows
        entry = {"window_id": wid, "role": role, "domain": f"d{k % 4}", "path": path, "bytes": len(body),
                 "sha256": sha256(body), "token_ids_sha256": sha256(arr), "prediction_positions": COUNT}
        (finals if role == "final" else others).append(entry)
    same = {"model_revision": "m", "token_panel_receipt_sha256": "t", "vocab_size": V}
    files["dataset-manifest.json"] = json.dumps({**same, "dataset_sha256": "d", "logit_files": finals}).encode()
    files[kf.FULL_PANEL_MANIFEST] = json.dumps({**same, "full_panel_manifest_sha256": "f",
                                                "logit_files": others}).encode()
    for p in kf.META_FILES:
        files.setdefault(p, b"{}\n")
    files["calibration/panel-v1/tokenizer.receipt.json"] = json.dumps({"vocab_size": V}).encode()
    return files, rows_of


class FakeHub(kf.Hub):
    """The Hub and its CDN, served from ``files``; ``log`` records every request."""

    files: dict[str, bytes] = {}
    log: list[tuple] = []

    @property
    def http(self):
        return self

    @staticmethod
    def lfs(path: str) -> bool:
        return path.endswith((".safetensors", ".npy"))

    def get(self, url: str, headers: dict | None = None, expect: int = 1 << 16):
        self.pacer.reserve(expect)
        try:
            status, hdrs, body = self.serve(urllib.parse.urlsplit(url), headers or {})
            self.pacer.account(len(body))
            return status, hdrs, body
        finally:
            self.pacer.release(expect)

    def serve(self, u, headers: dict):
        resolve = f"/datasets/{self.repo}/resolve/{self.revision}/"
        if u.hostname == "hub.test" and u.path == f"/api/datasets/{self.repo}/revision/{self.revision}":
            sib = [{"rfilename": p, "size": len(b), "blobId": kf.git_blob_sha1(b),
                    **({"lfs": {"sha256": sha256(b)}} if self.lfs(p) else {})} for p, b in self.files.items()]
            return 200, {}, json.dumps({"sha": self.revision, "siblings": sib}).encode()
        if u.hostname == "hub.test" and u.path.startswith(resolve):
            path = urllib.parse.unquote(u.path[len(resolve):])
            body = self.files[path]
            if not self.lfs(path):
                self.log.append(("whole", path))
                return 200, {}, body
            return 302, {"location": f"https://cdn.test/{sha256(body)}", "x-repo-commit": self.revision,
                         "x-linked-etag": f'"{sha256(body)}"', "x-linked-size": str(len(body))}, b""
        if u.hostname == "cdn.test":
            path = next(p for p, b in self.files.items() if sha256(b) == u.path[1:])
            body = self.files[path]
            if "Range" not in headers:
                self.log.append(("whole", path))
                return 200, {}, body
            a, b = map(int, headers["Range"][len("bytes="):].split("-"))
            self.log.append(("range", path, a, b - a + 1))
            return 206, {"content-range": f"bytes {a}-{b}/{len(body)}"}, body[a:b + 1]
        return 404, {}, b""


class FetchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.rows = dataset(random.Random(5))
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fetch(self, name: str, *argv: str, files: dict | None = None) -> str:
        out = os.path.join(self.tmp.name, name)
        FakeHub.files, FakeHub.log = files or self.files, []
        a = kf.parser().parse_args(["--out", out, "--revision", REV, "--endpoint", ENDPOINT, "--rate", "0",
                                    "--connections", "4", "--head-rows", "2", "--rows-per-window", "4", *argv])
        with contextlib.redirect_stderr(io.StringIO()):
            kf.run(a, FakeHub)
        return out

    def row_fetches(self) -> dict[str, int]:
        """Row-sized range requests per source file (a window's header reads are other sizes)."""
        out: dict[str, int] = {}
        for entry in FakeHub.log:
            if entry[0] == "range" and entry[3] == V * 4:
                out[entry[1]] = out.get(entry[1], 0) + 1
        return out

    def manifest(self, out: str) -> dict:
        with open(os.path.join(out, "FETCH-MANIFEST.json")) as f:
            return json.load(f)

    def check_rows(self, out: str, wid: str) -> None:
        with klgate.SafeTensors(os.path.join(out, "teacher-rows", f"{wid}.safetensors")) as st:
            self.assertEqual(st.ints("positions"), ROWS)
            got = [st.row("logits", i).tobytes() for i in range(len(ROWS))]
        self.assertEqual(got, [self.rows[wid][r] for r in ROWS], wid)

    def test_panel_is_a_prefix_in_role_order(self):
        out = self.fetch("p5", "--panel", "5")
        want = ["final-0000", "final-0001", "final-0002", "confirmation-0000", "confirmation-0001"]
        m = self.manifest(out)
        self.assertEqual(m["panel"]["windows"], want)
        self.assertEqual(m["panel"]["window_ids_sha256"], klgate.window_ids_sha256(want))
        self.assertEqual(m["panel"]["roles"], {"final": 3, "confirmation": 2})
        self.assertEqual(sorted(os.listdir(os.path.join(out, "teacher-rows"))), sorted(f"{w}.safetensors" for w in want))
        for wid in want:
            self.check_rows(out, wid)
        self.assertEqual(sum(self.row_fetches().values()), 5 * len(ROWS))
        self.assertIn(kf.FULL_PANEL_MANIFEST, [f["path"] for f in m["whole_files"]])
        with open(os.path.join(out, "SHA256SUMS")) as f:
            for line in f:
                digest, path = line.rstrip("\n").split("  ", 1)
                self.assertEqual(klgate.sha256_file(os.path.join(out, path)), digest, path)
        self.assertFalse([n for _, _, ns in os.walk(out) for n in ns if ".partial" in n])

    def test_final_windows_alone_need_no_full_panel_manifest(self):
        out = self.fetch("p3", "--panel", "3")
        self.assertEqual(self.manifest(out)["panel"]["windows"], ["final-0000", "final-0001", "final-0002"])
        self.assertFalse(os.path.exists(os.path.join(out, kf.FULL_PANEL_MANIFEST)))
        self.assertNotIn(("whole", kf.FULL_PANEL_MANIFEST), FakeHub.log)

    def test_a_larger_panel_copies_the_rows_it_already_has(self):
        first = self.fetch("reuse-first", "--panel", "5")
        out = self.fetch("reuse-larger", "--panel", "9", "--reuse", first)
        m = self.manifest(out)
        order = m["panel"]["windows"]
        self.assertEqual(order[:5], self.manifest(first)["panel"]["windows"])
        self.assertEqual(order[5:], ["confirmation-0002", "confirmation-0003", "selection-0000", "selection-0001"])
        fetched = self.row_fetches()
        by_id = {r["window_id"]: r for r in m["windows"]}
        for wid in order:
            src = by_id[wid]["source"]["path"]
            self.check_rows(out, wid)
            if wid in order[:5]:
                self.assertEqual(by_id[wid]["rows_reused_from_an_earlier_local_file"], len(ROWS))
                self.assertNotIn(src, fetched)
                self.assertEqual(klgate.sha256_file(os.path.join(out, "teacher-rows", f"{wid}.safetensors")),
                                 klgate.sha256_file(os.path.join(first, "teacher-rows", f"{wid}.safetensors")))
            else:
                self.assertEqual(fetched[src], len(ROWS))
        # klgate.py reads the directory as the panel, in its order
        teacher = klgate.Teacher(out)
        self.assertEqual(teacher.ids("", ""), order)
        self.assertEqual(teacher.ids("", "", "final"), order[:3])
        plan = os.path.join(self.tmp.name, "plan.json")
        with contextlib.redirect_stdout(io.StringIO()):
            klgate.cmd_plan(klgate.argparse.Namespace(teacher=out, windows="", roles="", exclude="", out=plan))
            canary = klgate.cmd_canary(klgate.argparse.Namespace(teacher=out, windows="", roles="", exclude="",
                                                                 vocab="stored", jobs=1))
        with open(plan) as f:
            p = json.load(f)
        self.assertEqual([w["window_id"] for w in p["windows"]], order)
        self.assertTrue(all(w["positions"] == ROWS for w in p["windows"]))
        self.assertEqual(p["panel"]["window_ids_sha256"], m["panel"]["window_ids_sha256"])
        self.assertEqual(canary, 0)

    def test_a_stopped_closing_step_resumes_without_fetching_rows(self):
        """The tool stopped after its last row, inside the closing step: some windows renamed, the others
        still ``.partial`` with every row listed, no manifest and no checksums. The same command finishes it."""
        done = self.fetch("stopped-src", "--panel", "5")
        out = os.path.join(self.tmp.name, "stopped")
        shutil.copytree(done, out)
        rows_dir = os.path.join(out, "teacher-rows")
        stopped = ["confirmation-0000", "confirmation-0001"]
        for wid in stopped:
            path = os.path.join(rows_dir, f"{wid}.safetensors")
            os.rename(path, path + ".partial")
            with open(path + ".partial.rows", "w") as f:
                f.write("".join(f"{k}\n" for k in range(len(ROWS))))
        os.remove(os.path.join(out, "FETCH-MANIFEST.json"))
        os.remove(os.path.join(out, "SHA256SUMS"))
        self.fetch("stopped", "--panel", "5")
        self.assertEqual(self.row_fetches(), {})
        self.assertFalse([n for _, _, ns in os.walk(out) for n in ns if ".partial" in n])
        for name in ("SHA256SUMS", *(f"teacher-rows/{w}.safetensors" for w in self.manifest(done)["panel"]["windows"])):
            with open(os.path.join(out, name), "rb") as a, open(os.path.join(done, name), "rb") as b:
                self.assertEqual(a.read(), b.read(), name)
        by_id = {r["window_id"]: r for r in self.manifest(out)["windows"]}
        self.assertTrue(all(by_id[w]["rows_sha256"] for w in stopped))
        self.assertTrue(all(by_id[w]["rows_sha256"] is None for w in by_id if w not in stopped))

    def test_named_windows_keep_their_order(self):
        out = self.fetch("named", "--windows", "selection-0001,final-0002")
        self.assertEqual(self.manifest(out)["panel"]["windows"], ["selection-0001", "final-0002"])
        self.assertTrue(os.path.exists(os.path.join(out, kf.FULL_PANEL_MANIFEST)))

    def test_refusals(self):
        for argv in (["--panel", "0"], ["--panel", str(len(WINDOWS) + 1)], ["--windows", "final-0009"],
                     ["--windows", "final-0001,final-0001"], ["--panel", "4", "--windows", "final-0000"]):
            with self.subTest(argv=argv), self.assertRaises(kf.FetchError):
                self.fetch("refused", *argv)
        other = dict(self.files)
        full = json.loads(other[kf.FULL_PANEL_MANIFEST])
        full["token_panel_receipt_sha256"] = "another panel"
        other[kf.FULL_PANEL_MANIFEST] = json.dumps(full).encode()
        with self.assertRaises(kf.FetchError):
            self.fetch("refused-manifest", "--panel", "4", files=other)


if __name__ == "__main__":
    unittest.main()
