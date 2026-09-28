#!/usr/bin/env python3
"""Self-test of harness/api_contract.py, before it is pointed at a server.

A scripted fake server on a loopback port either keeps the contract (thinking switches, clear_thinking,
one reasoning field, chunk heads, the usage chunk, UTF-8 in both encodings, the declared tool, isolated
requests) or breaks exactly one part of it. The harness must pass the good server and fail exactly the
rows that own the broken part. A fake whose text is meaningless shows what --dev does: its model checks
are printed but not judged.

    python3 harness/test_api_contract.py      (standard library only)
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE / "api_contract.py"
LT = "<"
ALL_ROWS = ["LIVE", "CREATED", "JSON", "STREAM-MARKUP", "USAGE", "UTF8-esc", "UTF8-raw", "TOOLS-json",
            "TOOLS-stream", "THINK-OFF", "CLEAR-THINKING", "ISO"]
MODEL_ROWS = {"LIVE", "UTF8-esc", "UTF8-raw", "TOOLS-json", "TOOLS-stream", "THINK-OFF", "ISO"}


def thinking_of(body, mode):
    """The request's thinking switch, in the order the glm53f API reads it; GLM's default is on."""
    if mode == "think-ignored":
        return True
    kw = body.get("chat_template_kwargs") or {}
    for v in (kw.get("enable_thinking"), kw.get("thinking"), body.get("enable_thinking")):
        if isinstance(v, bool):
            return v
    t = body.get("thinking")
    if isinstance(t, dict) and isinstance(t.get("type"), str):
        return t["type"] != "disabled"
    effort = body.get("reasoning_effort") or kw.get("reasoning_effort")
    if effort == "none" and mode != "effort-none-ignored":
        return False
    return True


def clear_of(body, mode):
    if mode == "clear-ignored":
        return False
    for v in ((body.get("chat_template_kwargs") or {}).get("clear_thinking"), (body.get("thinking") or {}).get("clear_thinking")):
        if isinstance(v, bool):
            return v
    return False


def render(messages, thinking, clear):
    """A toy prompt: what the prompt-token count is taken from."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    out = []
    for i, m in enumerate(messages):
        text = m.get("content") if isinstance(m.get("content"), str) else ""
        if m.get("role") == "assistant":
            r = m.get("reasoning_content") or m.get("reasoning") or ""
            keep = not clear or i > last_user
            out.append("[a][t]" + (r if keep else "") + "[/t]" + text)
        else:
            out.append(f"[{m.get('role')}]" + text)
    out.append("[a][t]" if thinking else "[a][t][/t]")
    return "".join(out)


class Fake:
    """A chat completions server that breaks one part of the contract (`mode`), or none (`good`)."""

    def __init__(self, mode):
        self.mode, self.dead, self.memory, self.lock = mode, False, [], threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                fake.send_json(self, 200, {"object": "list", "data": [{"id": "glm-5.3-flash", "object": "model"}]})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                try:
                    fake.answer(self, raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the harness gave up on a slow reply

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.srv.server_address[1]}"

    @staticmethod
    def send_json(h, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def reply_text(self, q, tools, thinking):
        """(reasoning, content, calls) for the last user message `q`."""
        if self.mode == "garbage":
            return ("zq vx" if thinking else ""), "lorem qq", []
        reasoning = "Let me think about it." if thinking else ""
        if tools:
            name = "read_file" if self.mode == "undeclared-tool" else tools[0]["function"]["name"]
            args = '{"file_path": src/theme/palette.js}' if self.mode == "bad-args" else json.dumps({"file_path": "src/theme/palette.js"})
            return reasoning, "I will read the file.", [(name, args)]
        if q.startswith("Repeat exactly this text and nothing else: "):
            echo = q.split("nothing else: ", 1)[1]
            return reasoning, "Dont stop - Tokyo cafe :) quoted" if self.mode == "ascii" else echo, []
        if q.startswith("Reply exactly: OK") or q.startswith("The secret word is"):
            return reasoning, "OK", []
        if q.startswith("What is the secret word"):
            with self.lock:
                said = self.mode == "carry" and any("ZEBRA-17" in m for m in self.memory)
            return reasoning, "ZEBRA-17" if said else "NONE", []
        if "2 + 2" in q:
            return reasoning, "4", []
        if q == "ping":
            return reasoning, "pong", []
        return reasoning, "The answer is 42.", []

    def answer(self, h, raw):
        mode = self.mode
        if self.dead:
            return self.send_json(h, 500, {"error": {"message": "engine lock poisoned"}})
        if mode == "surrogate400" and b"\\ud83d" in raw:
            return self.send_json(h, 400, {"error": {"message": "json: bad \\u codepoint"}})
        text = raw.decode("latin-1") if mode == "mojibake" else raw.decode("utf-8")
        body = json.loads(text)
        messages = body["messages"]
        q = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "") or ""
        with self.lock:
            self.memory.append(q)
        if mode == "slow" and q.startswith("Reply exactly: OK"):
            time.sleep(1.5)
        if mode == "panic" and q.startswith("Repeat exactly this text"):
            self.dead = True  # one delta, then the connection drops: no finish, no [DONE]; 500s after
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.end_headers()
            h.wfile.write(b'data: {"choices": [{"index": 0, "delta": {"content": "Don"}}]}\n\n')
            h.wfile.flush()
            h.close_connection = True
            return
        thinking = thinking_of(body, mode)
        # `off-low-effort`: a template with no off mode (GLM-5.3-Flash) maps "off" to its Low
        # effort: thinking on, a short reasoning, and a prompt unlike thinking on's.
        kw_effort = body.get("reasoning_effort") or (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
        low = mode == "off-low-effort" and thinking is False and kw_effort != "none"
        if low:
            thinking = True
        tools = body.get("tools") or []
        reasoning, content, calls = self.reply_text(q, tools, thinking)
        if low:
            reasoning = "Brief."
        pt = len(render(messages, thinking, clear_of(body, mode))) + (3 if low else 0)
        ct = len(reasoning) // 4 + len(content) // 4 + 1
        usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct + (1 if mode == "usage-mismatch" else 0)}
        created = int(time.time()) * (1000 if mode == "created-ms" else 1)
        finish = "tool_calls" if calls else "stop"
        if body.get("stream"):
            self.stream(h, body, created, reasoning, content, calls, finish, usage)
        else:
            msg = {"role": "assistant", "content": None if calls else content}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
                                     for i, (n, a) in enumerate(calls)]
            obj = {"id": "chatcmpl-1", "object": "chat.completion.chunk" if mode == "wrong-object" else "chat.completion",
                   "created": created, "model": "glm-5.3-flash",
                   "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}
            self.send_json(h, 200, obj)
        if mode == "dieafter" and q.startswith("What is the secret word"):
            self.dead = True

    def stream(self, h, body, created, reasoning, content, calls, finish, usage):
        mode = self.mode
        head = {"id": "chatcmpl-2", "object": "chat.completion.chunk", "created": created, "model": "glm-5.3-flash"}
        events = []

        def ev(delta, finish_reason=None):
            e = dict(head) if (not events or mode != "chunk-no-created") else {}
            e["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
            events.append(e)

        def thirds(s):
            k = max(1, len(s) // 3)
            return [p for p in (s[:k], s[k:2 * k], s[2 * k:]) if p]

        ev({"role": "assistant"})
        if reasoning and mode == "markup-leak":
            content = reasoning + LT + "/think>" + content  # the reasoning block streamed as content
        elif reasoning:
            for p in thirds(reasoning):
                ev({"reasoning" if mode == "reasoning-alias" else "reasoning_content": p})
        for p in thirds(content):
            ev({"content": p})
        for i, (n, a) in enumerate(calls):
            ev({"tool_calls": [{"index": i, "id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": ""}}]})
            ev({"tool_calls": [{"index": i, "function": {"arguments": a}}]})
        ev({}, finish)
        if (body.get("stream_options") or {}).get("include_usage") and mode != "no-usage-chunk":
            u = dict(head, choices=[], usage=usage)
            if mode == "usage-with-choices":
                u["choices"] = [{"index": 0, "delta": {}, "finish_reason": None}]
            events.append(u)
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        for e in events:
            h.wfile.write(b"data: " + json.dumps(e, ensure_ascii=False).encode("utf-8") + b"\n\n")
        h.wfile.write(b"data: [DONE]\n\n")


def run(mode, *extra):
    with Fake(mode) as fake, tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run([sys.executable, str(HARNESS), "--base", fake.url, "--out", tmp, *extra],
                           capture_output=True, text=True, timeout=120)
        rec = json.loads((pathlib.Path(tmp) / "api-contract.json").read_text(encoding="utf-8"))
        md = (pathlib.Path(tmp) / "api-contract.md").read_text(encoding="utf-8")
    return p, {r["row"]: r for r in rec["rows"]}, rec, md


class GoodServer(unittest.TestCase):
    def test_off_as_low_effort_passes(self):
        """A server whose template has no off mode maps thinking off to its Low effort (a short
        reasoning, a different prompt); reasoning_effort=none still gives no reasoning."""
        p, rows, _, _ = run("off-low-effort")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual({k for k, r in rows.items() if r["verdict"] != "PASS"}, set(), p.stdout)

    def test_every_row_passes(self):
        p, rows, rec, md = run("good")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(list(rows), ALL_ROWS)
        self.assertEqual({k for k, r in rows.items() if r["verdict"] != "PASS"}, set(), p.stdout)
        self.assertTrue(all(r["alive_after"] for r in rows.values()))
        self.assertTrue(md.startswith("# API contract: PASS"), md)
        self.assertIn("RESULT: PASS", p.stdout)
        self.assertEqual(rec["model"], "glm-5.3-flash")  # found through /v1/models

    def test_a_subset_of_rows(self):
        p, rows, _, _ = run("good", "--rows", "LIVE,THINK-OFF")
        self.assertEqual((p.returncode, list(rows)), (0, ["LIVE", "THINK-OFF"]), p.stdout + p.stderr)


class BrokenServers(unittest.TestCase):
    """Each mode breaks one part of the contract; exactly the rows that own it fail."""

    CASES = {
        "reasoning-alias": {"STREAM-MARKUP", "TOOLS-stream"},  # streamed reasoning under `reasoning`
        "markup-leak": {"STREAM-MARKUP", "TOOLS-stream"},      # the think block streamed as content
        "chunk-no-created": {"CREATED"},                       # head fields on the first chunk only
        "created-ms": {"CREATED"},
        "wrong-object": {"CREATED"},
        "usage-mismatch": {"JSON", "USAGE"},
        "no-usage-chunk": {"USAGE"},
        "usage-with-choices": {"USAGE"},
        "undeclared-tool": {"TOOLS-json", "TOOLS-stream"},
        "bad-args": {"TOOLS-json", "TOOLS-stream"},
        "surrogate400": {"UTF8-esc"},                          # escaped surrogate pairs refused
        "mojibake": {"UTF8-raw"},                              # raw UTF-8 decoded as Latin-1
        "think-ignored": {"THINK-OFF"},
        "effort-none-ignored": {"THINK-OFF"},
        "clear-ignored": {"CLEAR-THINKING"},
        "carry": {"ISO"},                                      # context carried across requests
        "dieafter": {"ISO"},                                   # the row is clean; the server then dies
        "slow": {"LIVE"},                                      # answered after --live-timeout
        "ascii": {"UTF8-esc", "UTF8-raw"},                     # never streams a multi-byte character
        # Dies mid-stream, then answers 500: the row that killed it fails, and every row after it.
        "panic": {"UTF8-esc", "UTF8-raw", "TOOLS-json", "TOOLS-stream", "THINK-OFF", "CLEAR-THINKING", "ISO"},
    }

    def test_each_broken_part_fails_exactly_its_rows(self):
        for mode, failing in self.CASES.items():
            with self.subTest(mode=mode):
                extra = ("--live-timeout", "1") if mode == "slow" else ()
                p, rows, rec, _ = run(mode, *extra)
                self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
                self.assertEqual(rec["result"], "FAIL")
                self.assertEqual(list(rows), ALL_ROWS, p.stdout)  # a failed row never stops the others
                got = {k for k, r in rows.items() if r["verdict"] == "FAIL"}
                self.assertEqual(got, failing, p.stdout)
                if mode == "panic":
                    self.assertIs(rows["UTF8-esc"]["alive_after"], False)
                if mode == "dieafter":
                    self.assertIs(rows["ISO"]["alive_after"], False)
                    self.assertIn("alive after the row", rows["ISO"]["note"])
                if mode == "undeclared-tool":
                    self.assertIn("read_file", rows["TOOLS-json"]["note"])

    def test_a_server_that_does_not_answer(self):
        with Fake("good") as fake:
            url = fake.url
        p = subprocess.run([sys.executable, str(HARNESS), "--base", url, "--model", "m", "--timeout", "5"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("not run", p.stderr)
        p = subprocess.run([sys.executable, str(HARNESS), "--base", url], capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)


class DevelopmentServer(unittest.TestCase):
    """Meaningless text: structural rows pass; rows with model checks fail, or are INCONCLUSIVE with --dev."""

    def test_model_checks_are_judged_only_without_dev(self):
        p, rows, _, _ = run("garbage")
        self.assertEqual(p.returncode, 1, p.stdout)
        # ISO cannot fail on meaningless text (nothing leaks), so it passes either way.
        self.assertEqual({k for k, r in rows.items() if r["verdict"] == "FAIL"}, MODEL_ROWS - {"ISO"}, p.stdout)
        p, rows, rec, md = run("garbage", "--dev")
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertEqual({k for k, r in rows.items() if r["verdict"] == "INCONCLUSIVE"}, MODEL_ROWS, p.stdout)
        self.assertEqual({k for k, r in rows.items() if r["verdict"] == "PASS"}, set(ALL_ROWS) - MODEL_ROWS, p.stdout)
        self.assertTrue(rec["dev"] and "--dev" in md)


class Hygiene(unittest.TestCase):
    def test_sources_are_ascii_without_literal_tags(self):
        """A character written in place of its escape (an editing tool decoding it), or a tag
        spelled out, shows here."""
        for f in (HARNESS, pathlib.Path(__file__)):
            text = f.read_text(encoding="utf-8")
            self.assertTrue(text.isascii(), f)
            for tag in ("think", "/think", "tool_call", "/tool_call", "arg_key", "arg_value"):
                self.assertNotIn(LT + tag + ">", text, f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
