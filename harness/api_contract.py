#!/usr/bin/env python3
"""API contract rows for an OpenAI-compatible chat completions server.

Black-box rows that catch serving defects token-level tests miss. Each row sends a few short
requests and checks the replies. After every row a liveness probe (a tiny request) must answer,
so a request that leaves the server dead fails the row that caused it. Rows run once, never
retried. Python 3 standard library only; nothing here names a deployment.

Every check is one of three kinds:
- structural (S): the server's contract, whatever its weights: field names and shapes, SSE
  framing, usage arithmetic, and whether a request switch reached the prompt, which
  usage.prompt_tokens shows;
- model (M): needs a real model: the words of an answer, a tool actually called;
- info (i): recorded, never judged.
With --dev, for a development server whose text is meaningless, model checks are run and printed
but not judged: a row whose structural checks pass is INCONCLUSIVE when it has model checks.

Rows:
  LIVE            a short prompt answered within --live-timeout (M: the answer is OK)
  CREATED         `created` is integer Unix seconds near this machine's clock; `id`, `object`,
                  `created` and `model` on the reply and on every streamed chunk, with the same
                  `id` and `created` throughout the stream
  JSON            a non-streamed reply: JSON content type, one choice, the assistant message,
                  finish_reason, usage arithmetic, reasoning only under the reasoning field, no
                  template markup in content
  STREAM-MARKUP   a streamed reply with thinking at the server's default: no template markup
                  (think and tool-call tags, <|...|> tokens) in any content delta or in the joined
                  content, reasoning only under the reasoning field and without the closing think
                  tag, and the same field in the non-streamed reply to the same request
                  (i: whether the two texts agree)
  USAGE           stream_options.include_usage: exactly one usage chunk, the last before [DONE],
                  with `choices: []`; none without the option; the same prompt_tokens as the
                  non-streamed reply
  UTF8-esc        a multi-script echo (curly quotes, CJK, an emoji, Cyrillic, Greek) sent as
  UTF8-raw        \\u-escaped JSON (surrogate pairs), then as raw UTF-8: accepted, streamed as
                  strict UTF-8, and read alike (the same prompt_tokens for both)
                  (M: the echo carries the CJK word and multi-byte text, no U+FFFD)
  TOOLS-json      one declared tool: calls have an id, type `function`, a name and arguments
  TOOLS-stream    that are a JSON object; finish_reason is `tool_calls` exactly when there are
                  calls; no markup in content (M: a call is made, every call names the declared
                  tool, and it reads the file asked for)
  THINK-OFF       reasoning_effort=none gives an empty reasoning field (also when streamed);
                  chat_template_kwargs.enable_thinking=false and thinking.type=disabled give no
                  reasoning or, where the chat template has no off mode (GLM-5.3-Flash: its Low
                  effort), a reasoning only under the reasoning field; each off form renders a
                  prompt that differs from thinking on (M: thinking on reasons; off answers
                  directly)
  CLEAR-THINKING  clear_thinking (in chat_template_kwargs, or thinking.clear_thinking) drops an
                  earlier turn's reasoning from the prompt; it is off by default
  ISO             request isolation: a word planted by one request is unknown to the next (M)

The rows are modelled on mimo26f-afd's L5 API cell (harness/l5_api.py at v1.2.0), whose ISO row
is request isolation; CREATED covers timestamps. harness/PROVENANCE.md records what was adapted.
harness/test_api_contract.py checks every criterion against a fake server that breaks it.

usage: api_contract.py --base URL [--model ID] [--dev] [--out DIR] [--rows ROW,...]
  --base is the server root (a trailing /v1 is accepted). Without --model, the first id of
  GET /v1/models is used. Exit status: 0 when no row fails, 1 when one does (or the server stops
  answering), 2 when no model id can be found.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

S, M, I = "structural", "model", "info"
KIND_MARK = {S: "S", M: "M", I: "i"}

# Template markup that must never reach content. Built from pieces, never written literally, so
# an agent reading this file does not take it for markup.
LT = "<"
TAGS = tuple(LT + t for t in (
    "think>", "/think>", "tool_call>", "/tool_call>", "arg_key>", "/arg_key>", "arg_value>",
    "/arg_value>", "tool_response>", "/tool_response>", "function=", "/function>", "parameter=",
    "/parameter>"))
THINK_CLOSE = LT + "/think>"
THINK_OPEN = LT + "think>"
SPECIAL = re.compile(re.escape(LT) + r"\|[A-Za-z0-9_/.:-]{1,48}\|>")  # <|user|>, <|endoftext|>
SPECIAL_FW = re.compile(re.escape(LT) + "\uff5c[^\uff5c<>]{1,48}\uff5c>")  # full-width bars
REASONING_NAMES = ("reasoning_content", "reasoning")

LIVE_PROMPT = "Reply exactly: OK"
PROBE_PROMPT = "ping"
UTF8_TEXT = ("Don\u2019t stop \u2014 \u6771\u4eac caf\u00e9 \U0001F600 \u201cquoted\u201d "
             "\u041f\u0440\u0438\u0432\u0435\u0442 \u03ba\u03cc\u03c3\u03bc\u03b5")
UTF8_WORD = "\u6771\u4eac"  # the CJK word the echo must carry
ISO_WORD = "ZEBRA-17"
ISO_PLANT = f"The secret word is {ISO_WORD}. Reply with exactly: OK"
ISO_ASK = ("What is the secret word from the earlier message? Reply with the word only. "
           "If there is no earlier message, reply exactly: NONE")
TOOL = {"type": "function", "function": {
    "name": "read", "description": "Read a file and return its contents with line numbers.",
    "parameters": {"type": "object", "properties": {
        "file_path": {"type": "string", "description": "Path of the file to read"}},
        "required": ["file_path"]}}}
TOOL_MESSAGES = [
    {"role": "system", "content": "You are a coding assistant in a software repository. "
                                  "Use the provided tool to read files."},
    {"role": "user", "content": "Read src/theme/palette.js and tell me what is on line 3."},
]
THINK_ON = ("chat_template_kwargs.enable_thinking=true", {"chat_template_kwargs": {"enable_thinking": True}})
THINK_OFF_FORMS = (
    ("chat_template_kwargs.enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("thinking.type=disabled", {"thinking": {"type": "disabled"}}),
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
)
EARLIER_REASONING = "The user greets me, so I greet them back and offer help. " * 8
ROW_NAMES = ("LIVE", "CREATED", "JSON", "STREAM-MARKUP", "USAGE", "UTF8-esc", "UTF8-raw",
             "TOOLS-json", "TOOLS-stream", "THINK-OFF", "CLEAR-THINKING", "ISO")


def now_text():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def clip(s, n=120):
    s = s if isinstance(s, str) else repr(s)
    return s if len(s) <= n else s[:n] + "..."


def markup_in(text):
    """Template markup found in `text`: tags, <|...|> tokens and their full-width-bar kin."""
    if not isinstance(text, str):
        return []
    return [t for t in TAGS if t in text] + SPECIAL.findall(text)[:3] + SPECIAL_FW.findall(text)[:3]


def reasoning_markup(text):
    """What reasoning must not hold: the closing think tag (it ends the block), a leading opening
    tag (the prompt opened the block), or a special token."""
    if not isinstance(text, str):
        return []
    found = [THINK_CLOSE] if THINK_CLOSE in text else []
    if text.lstrip().startswith(THINK_OPEN):
        found.append("leading " + THINK_OPEN)
    return found + SPECIAL.findall(text)[:3] + SPECIAL_FW.findall(text)[:3]


# ------------------------------------------------------------------------------------------ HTTP
class Reply:
    """A finished non-streamed request."""

    def __init__(self, status=None, body=None, text="", ctype="", error=None, wall=0.0):
        self.status, self.body, self.text, self.ctype, self.error, self.wall = status, body, text, ctype, error, wall


class Stream:
    """A finished streamed request: the parsed data events in order, and what went wrong."""

    def __init__(self):
        self.status, self.ctype, self.error, self.wall = None, "", None, 0.0
        self.events, self.bad, self.done, self.comments, self.sent = [], [], False, 0, b""


class Client:
    def __init__(self, base, timeout):
        base = base.rstrip("/")
        self.base = base[:-3] if base.endswith("/v1") else base
        self.timeout = timeout
        self.requests = 0

    def open(self, path, body=None, raw_utf8=False, timeout=None):
        data, headers = None, {}
        if body is not None:
            data = json.dumps(body, ensure_ascii=not raw_utf8).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        self.requests += 1
        return urllib.request.urlopen(req, timeout=timeout or self.timeout), data

    def models(self):
        resp, _ = self.open("/v1/models")
        with resp:
            return [m.get("id") for m in json.load(resp).get("data", [])]

    def json(self, body, raw_utf8=False, timeout=None):
        t0 = time.monotonic()
        try:
            resp, _ = self.open("/v1/chat/completions", body, raw_utf8, timeout)
            with resp:
                raw, status, ctype = resp.read(), resp.status, resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            raw, status, ctype = e.read(), e.code, e.headers.get("Content-Type", "")
        except Exception as e:  # refused, reset, timed out
            return Reply(error=f"{type(e).__name__}: {e}", wall=time.monotonic() - t0)
        wall = time.monotonic() - t0
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            return Reply(status, None, "", ctype, f"reply is not UTF-8: {e}", wall)
        try:
            obj = json.loads(text)
        except ValueError as e:
            return Reply(status, None, text, ctype, f"HTTP {status}, reply is not JSON: {e}: {clip(text)}", wall)
        return Reply(status, obj, text, ctype, None if status == 200 else f"HTTP {status}: {clip(text, 300)}", wall)

    def stream(self, body, include_usage=False, raw_utf8=False, timeout=None):
        body = dict(body, stream=True)
        if include_usage:
            body["stream_options"] = {"include_usage": True}
        s, t0 = Stream(), time.monotonic()
        try:
            resp, s.sent = self.open("/v1/chat/completions", body, raw_utf8, timeout)
        except urllib.error.HTTPError as e:
            s.status, s.error = e.code, f"HTTP {e.code}: {clip(e.read().decode('utf-8', 'replace'), 300)}"
            s.wall = time.monotonic() - t0
            return s
        except Exception as e:
            s.error, s.wall = f"{type(e).__name__}: {e}", time.monotonic() - t0
            return s
        try:
            with resp:
                s.status, s.ctype = resp.status, resp.headers.get("Content-Type", "")
                for raw in resp:
                    try:
                        line = raw.decode("utf-8").rstrip("\r\n")
                    except UnicodeDecodeError:
                        s.bad.append(f"a line that is not UTF-8: {raw[:80]!r}")
                        continue
                    if not line:
                        continue
                    if line.startswith(":"):
                        s.comments += 1
                        continue
                    if not line.startswith("data:"):
                        s.bad.append(f"not an SSE data line: {clip(line, 80)}")
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        s.done = True
                        break
                    try:
                        s.events.append(json.loads(payload))
                    except ValueError:
                        s.bad.append(f"an event that is not JSON: {clip(payload, 80)}")
        except Exception as e:  # the connection dropped mid-stream
            s.error = f"{type(e).__name__}: {e}"
        s.wall = time.monotonic() - t0
        return s


# ------------------------------------------------------------------------------------ gathering
def first_choice(body):
    ch = body.get("choices") if isinstance(body, dict) else None
    return ch[0] if isinstance(ch, list) and ch and isinstance(ch[0], dict) else None


def message(body):
    ch = first_choice(body)
    msg = ch.get("message") if ch else None
    return msg if isinstance(msg, dict) else None


def prompt_tokens(body):
    u = body.get("usage") if isinstance(body, dict) else None
    v = u.get("prompt_tokens") if isinstance(u, dict) else None
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def other_reasoning_names(obj, field):
    """Other reasoning names that carry text in a message or delta: reasoning under a second
    name, which a client reading `field` loses."""
    return [n for n in REASONING_NAMES if n != field and isinstance(obj.get(n), str) and obj.get(n)]


class Gathered:
    """A stream's deltas joined: content, reasoning under `field`, tool calls by index."""

    def __init__(self, stream, field):
        self.content_deltas, self.reasoning_parts, self.others, self.framing = [], [], [], []
        self.calls, self.finish, self.role, self.usage_at = {}, None, None, []
        for i, ev in enumerate(stream.events):
            if not isinstance(ev, dict):
                self.framing.append(f"event {i} is not an object")
                continue
            if ev.get("usage") is not None:
                self.usage_at.append(i)
            choices = ev.get("choices")
            if not isinstance(choices, list) or not all(isinstance(ch, dict) for ch in choices):
                self.framing.append(f"event {i}: choices is not a list of objects")
                continue
            for ch in choices:
                d = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
                if self.role is None and d.get("role"):
                    self.role = d["role"]
                if isinstance(d.get("content"), str):
                    self.content_deltas.append(d["content"])
                if isinstance(d.get(field), str):
                    self.reasoning_parts.append(d[field])
                self.others += other_reasoning_names(d, field)
                for tc in d.get("tool_calls") or []:
                    self._call(i, tc)
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
        self.content = "".join(self.content_deltas)
        self.reasoning = "".join(self.reasoning_parts)

    def _call(self, i, tc):
        k = tc.get("index")
        if not isinstance(k, int):
            self.framing.append(f"event {i}: a tool_calls delta without an integer index")
            return
        f = tc.get("function") or {}
        if k not in self.calls:
            self.calls[k] = {"id": tc.get("id"), "type": tc.get("type"), "name": f.get("name") or "",
                             "arguments": f.get("arguments") or ""}
            if not tc.get("id") or not f.get("name"):
                self.framing.append(f"event {i}: the first delta of call {k} lacks its id or name")
            return
        slot = self.calls[k]
        slot["name"] += f.get("name") or ""
        slot["arguments"] += f.get("arguments") or ""

    def call_list(self):
        return [self.calls[k] for k in sorted(self.calls)]


def call_problems(calls):
    probs, ids = [], set()
    for n, c in enumerate(calls):
        if not isinstance(c.get("id"), str) or not c["id"]:
            probs.append(f"call {n} has no id")
        elif c["id"] in ids:
            probs.append(f"call id {c['id']!r} repeats")
        else:
            ids.add(c["id"])
        if c.get("type") != "function":
            probs.append(f"call {n} has type {c.get('type')!r}")
        if not isinstance(c.get("name"), str) or not c["name"]:
            probs.append(f"call {n} has no name")
        args = c.get("arguments")
        if not isinstance(args, str):
            probs.append(f"call {n}: arguments are not a string")
            continue
        try:
            v = json.loads(args)
        except ValueError:
            probs.append(f"call {n}: arguments are not JSON: {clip(args, 80)}")
            continue
        if not isinstance(v, dict):
            probs.append(f"call {n}: arguments are not a JSON object")
    return probs


def usage_problems(u, max_tokens=None):
    if not isinstance(u, dict):
        return ["no usage object"]
    vals = {k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    bad = [k for k, v in vals.items() if not isinstance(v, int) or isinstance(v, bool) or v < 0]
    if bad:
        return [f"{', '.join(bad)} not a non-negative integer ({vals})"]
    probs = []
    if vals["total_tokens"] != vals["prompt_tokens"] + vals["completion_tokens"]:
        probs.append(f"total_tokens {vals['total_tokens']} != prompt + completion {vals}")
    if vals["prompt_tokens"] == 0:
        probs.append("prompt_tokens is 0")
    if max_tokens is not None and vals["completion_tokens"] > max_tokens:
        probs.append(f"completion_tokens {vals['completion_tokens']} > max_tokens {max_tokens}")
    return probs


def head_problems(obj, object_name, now, skew):
    """`id`, `object`, `created` (integer Unix seconds within `skew` of `now`) and `model`."""
    probs = []
    if not isinstance(obj.get("id"), str) or not obj.get("id"):
        probs.append(f"id {obj.get('id')!r}")
    if obj.get("object") != object_name:
        probs.append(f"object {obj.get('object')!r}, expected {object_name!r}")
    c = obj.get("created")
    if not isinstance(c, int) or isinstance(c, bool):
        probs.append(f"created {c!r} is not an integer")
    elif c > 10**11:
        probs.append(f"created {c} looks like milliseconds")
    elif abs(c - now) > skew:
        probs.append(f"created {c} is {c - now:+.0f} s from this clock (limit {skew} s)")
    if not isinstance(obj.get("model"), str) or not obj.get("model"):
        probs.append(f"model {obj.get('model')!r}")
    return probs


# ------------------------------------------------------------------------------------------ rows
class Row:
    def __init__(self, name):
        self.name, self.checks, self.t0, self.alive_after, self.wall = name, [], time.monotonic(), None, 0.0

    def check(self, kind, ok, what, detail=""):
        self.checks.append({"kind": kind, "ok": ok if ok is None else bool(ok), "what": what, "detail": clip(str(detail), 300)})
        return ok

    def answered(self, rep, what="request"):
        ok = rep.error is None and rep.status == 200 and isinstance(rep.body, dict)
        return self.check(S, ok, f"{what}: HTTP 200 with a JSON object", rep.error or f"{rep.wall:.2f} s")

    def streamed(self, st, what="stream"):
        probs = ([st.error] if st.error else []) + st.bad[:3]
        if st.status == 200 and not st.ctype.startswith("text/event-stream"):
            probs.append(f"content type {st.ctype!r}")
        if st.status == 200 and not st.done and not st.error:
            probs.append("no data: [DONE]")
        ok = st.status == 200 and not probs
        return self.check(S, ok, f"{what}: HTTP 200, SSE events that parse, data: [DONE]",
                          "; ".join(probs) or f"{len(st.events)} events, {st.wall:.2f} s")

    def verdict(self, dev):
        if any(c["kind"] == S and not c["ok"] for c in self.checks) or self.alive_after is False:
            return "FAIL"
        model = [c for c in self.checks if c["kind"] == M]
        if model and dev:
            return "INCONCLUSIVE"
        return "FAIL" if any(not c["ok"] for c in model) else "PASS"


class Ctx:
    def __init__(self, a, client, model):
        self.a, self.http, self.model, self.shared = a, client, model, {}

    def body(self, prompt=None, messages=None, max_tokens=64, thinking=None, **extra):
        b = {"model": self.model, "messages": messages or [{"role": "user", "content": prompt}],
             "max_tokens": max_tokens, "temperature": 0}
        if thinking is not None:
            b["chat_template_kwargs"] = {"enable_thinking": thinking}
        b.update(extra)
        return b


def row_live(ctx):
    r = Row("LIVE")
    rep = ctx.http.json(ctx.body(LIVE_PROMPT, max_tokens=16, thinking=False), timeout=ctx.a.live_timeout)
    if not r.answered(rep, f"answered within {ctx.a.live_timeout} s"):
        return r
    msg = message(rep.body)
    r.check(S, msg is not None, "a choice with a message", "present" if msg is not None else clip(rep.text, 200))
    content = (msg or {}).get("content") or ""
    r.check(M, "OK" in content.upper(), "the answer is OK", repr(clip(content, 80)))
    return r


def row_created(ctx):
    r = Row("CREATED")
    body = ctx.body("Say hello.", max_tokens=8, thinking=False)
    rep = ctx.http.json(body)
    if r.answered(rep, "non-streamed"):
        probs = head_problems(rep.body, "chat.completion", time.time(), ctx.a.skew)
        r.check(S, not probs, "reply: id, object, created (Unix seconds, near this clock), model",
                "; ".join(probs) or f"created {rep.body.get('created')}, this clock {time.time():.0f}")
    st = ctx.http.stream(body, include_usage=True)
    if not r.streamed(st):
        return r
    now, probs = time.time(), []
    first = st.events[0] if st.events and isinstance(st.events[0], dict) else {}
    for i, ev in enumerate(st.events):
        p = head_problems(ev, "chat.completion.chunk", now, ctx.a.skew) if isinstance(ev, dict) else ["not an object"]
        if isinstance(ev, dict) and ev.get("id") != first.get("id"):
            p.append(f"id {ev.get('id')!r} != the first chunk's {first.get('id')!r}")
        if isinstance(ev, dict) and ev.get("created") != first.get("created"):
            p.append(f"created {ev.get('created')!r} != the first chunk's {first.get('created')!r}")
        probs += [f"chunk {i}: {x}" for x in p]
    r.check(S, st.events and not probs, "every chunk: id, object, created, model; one id and created",
            "; ".join(probs[:4]) + (f" (+{len(probs) - 4} more)" if len(probs) > 4 else "")
            or f"{len(st.events)} chunks, id {first.get('id')!r}, created {first.get('created')}")
    return r


def row_json(ctx):
    r = Row("JSON")
    rep = ctx.http.json(ctx.body("What is the capital of France? Answer in one word.", max_tokens=ctx.a.max_tokens))
    if not r.answered(rep):
        return r
    field = ctx.a.reasoning_field
    r.check(S, rep.ctype.lower().startswith("application/json"), "content type application/json", rep.ctype)
    choices = rep.body.get("choices")
    one = isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) and choices[0].get("index") == 0
    r.check(S, one, "one choice, index 0", clip(json.dumps(choices), 200))
    msg = message(rep.body) or {}
    r.check(S, msg.get("role") == "assistant", "message role assistant", repr(msg.get("role")))
    r.check(S, isinstance(msg.get("content"), str), "content is a string (no tools were offered)", repr(type(msg.get("content")).__name__))
    fr = (first_choice(rep.body) or {}).get("finish_reason")
    r.check(S, fr in ("stop", "length"), "finish_reason stop or length", repr(fr))
    probs = usage_problems(rep.body.get("usage"), ctx.a.max_tokens)
    r.check(S, not probs, "usage: integers, total = prompt + completion, completion <= max_tokens",
            "; ".join(probs) or json.dumps(rep.body.get("usage")))
    others = other_reasoning_names(msg, field)
    rv = msg.get(field)
    r.check(S, not others and (rv is None or isinstance(rv, str)), f"reasoning only under {field}",
            f"also under {others}" if others else f"{len(rv or '')} chars under {field}")
    leak = markup_in(msg.get("content"))
    r.check(S, not leak, "no template markup in content", f"{leak}: {clip(msg.get('content'))}" if leak else "none")
    rleak = reasoning_markup(rv)
    r.check(S, not rleak, "no closing think tag or special token in reasoning", str(rleak) if rleak else "none")
    return r


def row_stream_markup(ctx):
    r = Row("STREAM-MARKUP")
    field = ctx.a.reasoning_field
    body = ctx.body("What is 17 + 25? Think it through briefly, then give the number.", max_tokens=ctx.a.max_tokens)
    st = ctx.http.stream(body)
    if not r.streamed(st):
        return r
    g = Gathered(st, field)
    r.check(S, g.role == "assistant", "the role chunk comes first", repr(g.role))
    per_delta = [(i, m) for i, d in enumerate(g.content_deltas) for m in markup_in(d)]
    joined = markup_in(g.content)
    r.check(S, not per_delta and not joined, "no template markup in any content delta or the joined content",
            f"deltas {per_delta[:4]}, joined {joined}" if per_delta or joined else f"{len(g.content_deltas)} content deltas")
    r.check(S, not g.others, f"streamed reasoning only under {field}",
            f"reasoning also under {sorted(set(g.others))}" if g.others else f"{len(g.reasoning_parts)} reasoning deltas")
    rleak = reasoning_markup(g.reasoning)
    r.check(S, not rleak, "no closing think tag or special token in reasoning", str(rleak) if rleak else "none")
    r.check(S, g.finish in ("stop", "length"), "finish_reason stop or length", repr(g.finish))
    r.check(I, bool(g.content), "the stream left the reasoning block (content after reasoning)",
            f"{len(g.reasoning)} reasoning chars, {len(g.content)} content chars")
    rep = ctx.http.json(body)
    if r.answered(rep, "the same request, non-streamed"):
        msg = message(rep.body) or {}
        others = other_reasoning_names(msg, field)
        r.check(S, not others, f"non-streamed reasoning only under {field}", f"also under {others}" if others else "ok")
        leak = markup_in(msg.get("content"))
        r.check(S, not leak, "no template markup in the non-streamed content", str(leak) if leak else "none")
        same = (msg.get("content") or "") == g.content and (msg.get(field) or "") == g.reasoning
        r.check(I, same, "streamed and non-streamed texts agree (greedy; the engine need not be bit-deterministic)",
                "equal" if same else f"stream {clip(g.reasoning, 60)!r} / {clip(g.content, 60)!r}; "
                f"whole {clip(msg.get(field) or '', 60)!r} / {clip(msg.get('content') or '', 60)!r}")
    return r


def row_usage(ctx):
    r = Row("USAGE")
    body = ctx.body("Name three colours.", max_tokens=16, thinking=False)
    st = ctx.http.stream(body, include_usage=True)
    streamed_pt = None
    if r.streamed(st, "with include_usage"):
        g = Gathered(st, ctx.a.reasoning_field)
        r.check(S, len(g.usage_at) == 1, "exactly one chunk carries usage", f"chunks {g.usage_at} of {len(st.events)}")
        if len(g.usage_at) == 1:
            i = g.usage_at[0]
            ev = st.events[i]
            r.check(S, i == len(st.events) - 1, "the usage chunk is the last before [DONE]", f"chunk {i} of {len(st.events)}")
            r.check(S, ev.get("choices") == [], "the usage chunk has choices []", clip(json.dumps(ev.get("choices")), 100))
            probs = usage_problems(ev.get("usage"), 16)
            r.check(S, not probs, "usage: integers, total = prompt + completion, completion <= max_tokens",
                    "; ".join(probs) or json.dumps(ev.get("usage")))
            streamed_pt = prompt_tokens(ev)
    st2 = ctx.http.stream(body)
    if r.streamed(st2, "without include_usage"):
        extra = [i for i, ev in enumerate(st2.events) if isinstance(ev, dict) and (ev.get("usage") is not None or ev.get("choices") == [])]
        r.check(S, not extra, "no usage chunk without the option", f"chunks {extra}" if extra else "none")
    rep = ctx.http.json(body)
    if r.answered(rep, "non-streamed") and streamed_pt is not None:
        pt = prompt_tokens(rep.body)
        r.check(S, pt == streamed_pt, "prompt_tokens equal to the non-streamed count", f"stream {streamed_pt}, whole {pt}")
    return r


def row_utf8(ctx, raw):
    r = Row("UTF8-raw" if raw else "UTF8-esc")
    body = ctx.body(f"Repeat exactly this text and nothing else: {UTF8_TEXT}", max_tokens=64, thinking=False)
    st = ctx.http.stream(body, include_usage=True, raw_utf8=raw)
    sent = "raw UTF-8" if b"\xf0\x9f\x98\x80" in st.sent else ("surrogate-pair escapes" if b"\\ud83d" in st.sent else "neither form?")
    r.check(I, True, "request encoding", sent)
    if not r.streamed(st):
        return r
    g = Gathered(st, ctx.a.reasoning_field)
    r.check(S, g.finish in ("stop", "length"), "finish_reason stop or length", repr(g.finish))
    pt = next((prompt_tokens(ev) for ev in st.events if isinstance(ev, dict) and ev.get("usage")), None)
    counts = ctx.shared.setdefault("utf8_prompt_tokens", {})
    counts[raw] = pt
    if raw:
        other = counts.get(False)
        if pt is not None and other is not None:
            r.check(S, pt == other, "the same prompt_tokens as the escaped request (both decoded alike)", f"raw {pt}, escaped {other}")
        else:
            r.check(I, None, "prompt_tokens against the escaped request", f"not available (raw {pt}, escaped {other})")
    multibyte = sum(1 for ch in g.content if ord(ch) > 127)
    r.check(M, UTF8_WORD in g.content, "the echo carries the CJK word", repr(clip(g.content, 100)))
    r.check(M, multibyte > 0, "multi-byte characters streamed", f"{multibyte}")
    r.check(M, "\ufffd" not in g.content, "no U+FFFD replacement character", "none" if "\ufffd" not in g.content else "found")
    return r


def row_tools(ctx, streaming):
    r = Row("TOOLS-stream" if streaming else "TOOLS-json")
    field = ctx.a.reasoning_field
    body = ctx.body(messages=TOOL_MESSAGES, max_tokens=ctx.a.tool_max_tokens, tools=[TOOL])
    declared = {TOOL["function"]["name"]}
    if streaming:
        st = ctx.http.stream(body)
        if not r.streamed(st):
            return r
        g = Gathered(st, field)
        calls, content, finish, others = g.call_list(), g.content, g.finish, sorted(set(g.others))
        r.check(S, not g.framing, "tool_calls deltas: an index, and an id and name on each call's first delta",
                "; ".join(g.framing[:3]) or f"{len(calls)} calls")
    else:
        rep = ctx.http.json(body)
        if not r.answered(rep):
            return r
        msg = message(rep.body) or {}
        calls = [{"id": t.get("id"), "type": t.get("type"), "name": (t.get("function") or {}).get("name"),
                  "arguments": (t.get("function") or {}).get("arguments")} for t in (msg.get("tool_calls") or [])]
        content, finish, others = msg.get("content"), (first_choice(rep.body) or {}).get("finish_reason"), other_reasoning_names(msg, field)
        r.check(S, content is None or isinstance(content, str), "content is a string or null", repr(type(content).__name__))
    probs = call_problems(calls)
    r.check(S, not probs, "calls: an id, type function, a name, arguments a JSON object", "; ".join(probs[:3]) or f"{len(calls)} calls")
    r.check(S, (finish == "tool_calls") == bool(calls), "finish_reason is tool_calls exactly when there are calls",
            f"finish {finish!r}, {len(calls)} calls")
    leak = markup_in(content)
    r.check(S, not leak, "no template markup in content", f"{leak}: {clip(content)}" if leak else "none")
    r.check(S, not others, f"reasoning only under {field}", f"also under {others}" if others else "ok")
    names = [c.get("name") for c in calls]
    r.check(M, bool(calls), "a call is made", f"names {names}; content {clip(content or '', 80)!r}")
    undeclared = sorted({n for n in names if n not in declared})
    r.check(M, not undeclared, "every call names the declared tool", f"undeclared {undeclared}" if undeclared else f"{names}")
    paths = []
    for c in calls:
        try:
            paths.append(json.loads(c.get("arguments") or "{}").get("file_path"))
        except (ValueError, AttributeError):
            pass
    r.check(M, any(isinstance(p, str) and "palette.js" in p for p in paths), "the call reads the file asked for", f"file_path {paths}")
    return r


def row_think_off(ctx):
    r = Row("THINK-OFF")
    field = ctx.a.reasoning_field
    base = ctx.body("What is 2 + 2? Reply with the number only.", max_tokens=48)
    on = ctx.http.json(dict(base, **THINK_ON[1]))
    if not r.answered(on, THINK_ON[0]):
        return r
    on_pt, on_msg = prompt_tokens(on.body), message(on.body) or {}
    r.check(M, bool(on_msg.get(field)), "thinking on: the reply reasons", repr(clip(on_msg.get(field) or "", 80)))
    counts = {}
    for label, extra in THINK_OFF_FORMS:
        rep = ctx.http.json(dict(base, **extra))
        if not r.answered(rep, label):
            continue
        msg = message(rep.body) or {}
        rv, others = msg.get(field), other_reasoning_names(msg, field)
        seen = (f"{field} absent" if rv is None else f"{field} {clip(rv, 80)!r}") + (f", also under {others}" if others else "")
        if extra.get("reasoning_effort") == "none":
            r.check(S, rv in (None, "") and not others, f"{label}: empty reasoning", seen)
        else:
            # A template with no off mode maps "off" to its lowest effort: a short reasoning is
            # allowed, under the reasoning field only.
            r.check(S, not others, f"{label}: reasoning, if any, only under {field}", seen)
            r.check(I, True, f"{label}: reasoning", "none" if rv in (None, "") else f"{len(rv)} chars (low effort)")
        pt = counts[label] = prompt_tokens(rep.body)
        differs = pt is not None and on_pt is not None and pt != on_pt
        if extra.get("reasoning_effort") == "none":
            r.check(S, differs, f"{label}: the prompt differs from thinking on", f"prompt_tokens {pt} (on: {on_pt})")
        else:
            # Low effort may render a prompt of the same length as thinking on (only the effort
            # word changes); then it must reason less than thinking on did.
            on_r = on_msg.get(field) or ""
            shorter = rv in (None, "") or len(rv) < len(on_r)
            r.check(S, differs or shorter, f"{label}: honoured (another prompt, or less reasoning than thinking on)",
                    f"prompt_tokens {pt} (on: {on_pt}); reasoning {len(rv or '')} chars (on: {len(on_r)})")
        content = msg.get("content") or ""
        r.check(M, "4" in content, f"{label}: a direct answer", repr(clip(content, 80)))
    r.check(I, len(set(counts.values())) == 1, "the off forms render one prompt", json.dumps(counts))
    st = ctx.http.stream(dict(base, **THINK_OFF_FORMS[2][1]))
    if r.streamed(st, f"streamed, {THINK_OFF_FORMS[2][0]}"):
        g = Gathered(st, field)
        r.check(S, not g.reasoning and not g.others, "streamed with thinking off: no reasoning deltas",
                f"{clip(g.reasoning, 80)!r} {sorted(set(g.others))}" if g.reasoning or g.others else "none")
    default = ctx.http.json(base)
    if r.answered(default, "no switch (the server's default)"):
        pt = prompt_tokens(default.body)
        off = set(counts.values())
        r.check(I, True, "the server's default", "on" if pt == on_pt else ("off" if pt in off else f"? ({pt})"))
    return r


def row_clear_thinking(ctx):
    r = Row("CLEAR-THINKING")
    history = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Hello! How can I help?", "reasoning_content": EARLIER_REASONING},
        {"role": "user", "content": "Say OK."},
    ]
    forms = (("no clear_thinking", {}),
             ("chat_template_kwargs.clear_thinking=false", {"chat_template_kwargs": {"clear_thinking": False}}),
             ("chat_template_kwargs.clear_thinking=true", {"chat_template_kwargs": {"clear_thinking": True}}),
             ("thinking.clear_thinking=true", {"thinking": {"type": "enabled", "clear_thinking": True}}))
    pts = {}
    for label, extra in forms:
        rep = ctx.http.json(dict(ctx.body(messages=history, max_tokens=1), **extra))
        if r.answered(rep, label):
            pts[label] = prompt_tokens(rep.body)
    if len(pts) < len(forms):
        return r
    d, f, t, tt = (pts[label] for label, _ in forms)
    r.check(S, d is not None and d == f, "off by default: the same prompt as clear_thinking=false", f"{d} vs {f}")
    r.check(S, t is not None and d is not None and t < d, "clear_thinking=true drops the earlier reasoning", f"prompt_tokens {t} < {d}")
    r.check(S, tt == t, "thinking.clear_thinking=true does the same", f"{tt} vs {t}")
    alias = [dict(m) for m in history]
    alias[1]["reasoning"] = alias[1].pop("reasoning_content")
    rep = ctx.http.json(ctx.body(messages=alias, max_tokens=1))
    if r.answered(rep, "history reasoning under `reasoning`"):
        pa = prompt_tokens(rep.body)
        r.check(I, pa == d, "history reasoning sent as `reasoning` renders as `reasoning_content` does", f"{pa} vs {d}")
    return r


def row_iso(ctx):
    r = Row("ISO")
    plant = ctx.http.json(ctx.body(ISO_PLANT, max_tokens=24, thinking=False))
    if not r.answered(plant, "the planting request"):
        return r
    ask = ctx.http.json(ctx.body(ISO_ASK, max_tokens=24, thinking=False))
    if not r.answered(ask, "the independent request"):
        return r
    reply = (message(ask.body) or {}).get("content") or ""
    leaked = ISO_WORD.split("-")[0] in reply.upper()
    r.check(M, not leaked, "the second request does not know the planted word", repr(clip(reply, 80)))
    return r


ROWS = {
    "LIVE": row_live, "CREATED": row_created, "JSON": row_json, "STREAM-MARKUP": row_stream_markup,
    "USAGE": row_usage, "UTF8-esc": lambda c: row_utf8(c, raw=False), "UTF8-raw": lambda c: row_utf8(c, raw=True),
    "TOOLS-json": lambda c: row_tools(c, streaming=False), "TOOLS-stream": lambda c: row_tools(c, streaming=True),
    "THINK-OFF": row_think_off, "CLEAR-THINKING": row_clear_thinking, "ISO": row_iso,
}


def alive(ctx):
    """Liveness after a row: a tiny request must answer 200 with a choice."""
    rep = ctx.http.json(ctx.body(PROBE_PROMPT, max_tokens=4, thinking=False), timeout=ctx.a.live_timeout)
    if rep.error is None and first_choice(rep.body) is not None:
        return True, f"{rep.wall:.2f} s"
    return False, rep.error or "no choice in the reply"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="the server root, e.g. http://127.0.0.1:8000 (a trailing /v1 is accepted)")
    ap.add_argument("--model", help="the model id to request (default: the first id of GET /v1/models)")
    ap.add_argument("--dev", action="store_true", help="development server: model checks are printed, not judged")
    ap.add_argument("--out", help="write api-contract.json and api-contract.md here")
    ap.add_argument("--rows", help=f"comma-separated rows to run (default all: {','.join(ROW_NAMES)})")
    ap.add_argument("--reasoning-field", default="reasoning_content", help="where reasoning must be (default reasoning_content)")
    ap.add_argument("--max-tokens", type=int, default=512, help="budget of the JSON and STREAM-MARKUP rows")
    ap.add_argument("--tool-max-tokens", type=int, default=2048, help="budget of the TOOLS rows (thinking at the server's default)")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per request")
    ap.add_argument("--live-timeout", type=float, default=60.0, help="seconds for the LIVE row and each liveness probe")
    ap.add_argument("--skew", type=float, default=300.0, help="largest accepted difference between created and this clock")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="backslashreplace")  # evidence quotes model text; any locale prints it
    names = [n.strip() for n in a.rows.split(",")] if a.rows else list(ROW_NAMES)
    unknown = [n for n in names if n not in ROWS]
    if unknown:
        ap.error(f"unknown rows {unknown}; rows are {', '.join(ROW_NAMES)}")

    client = Client(a.base, a.timeout)
    try:
        ids = client.models()
    except Exception as e:
        ids = None
        if not a.model:
            print(f"api-contract: GET {client.base}/v1/models failed ({type(e).__name__}: {e}); pass --model", file=sys.stderr)
            return 2
    model = a.model or (ids[0] if ids else None)
    if not model:
        print("api-contract: /v1/models lists no model; pass --model", file=sys.stderr)
        return 2
    ctx = Ctx(a, client, model)
    rec = {"harness": "api_contract", "base": client.base, "model": model, "models_listed": ids, "dev": a.dev,
           "reasoning_field": a.reasoning_field, "started": now_text(), "rows": []}
    print(f"[api-contract] {rec['started']} base {client.base} model {model} (listed: {ids})"
          + (" --dev: model checks are printed, not judged" if a.dev else ""), flush=True)
    for name in names:
        try:
            row = ROWS[name](ctx)
        except Exception as e:  # a row that raises is a failed row, retained
            row = Row(name)
            row.check(S, False, "the row ran", f"{type(e).__name__}: {e}")
        row.wall = time.monotonic() - row.t0
        live, why = alive(ctx)
        row.alive_after = live
        row.check(S if not live else I, live, "alive after the row", why)
        v = row.verdict(a.dev)
        failing = [c for c in row.checks if c["ok"] is False and c["kind"] in (S, M)]
        note = "; ".join(f"{c['what']}: {c['detail']}" for c in failing[:2]) or "ok"
        rec["rows"].append({"row": name, "verdict": v, "wall_s": round(row.wall, 2), "alive_after": live,
                            "note": note, "checks": row.checks})
        print(f"  {name:<15} {v:<12} {row.wall:7.2f} s  {clip(note, 160)}", flush=True)
        for c in row.checks:
            mark = "--" if c["ok"] is None else ("ok" if c["ok"] else "NO")
            print(f"      [{KIND_MARK[c['kind']]}] {mark}  {c['what']}: {c['detail']}", flush=True)
        if name == "LIVE" and not live and not any(c["ok"] for c in row.checks if c["kind"] == S):
            print("api-contract: the server does not answer; the other rows are not run", file=sys.stderr)
            break
    rec["finished"] = now_text()
    rec["requests"] = client.requests
    verdicts = [x["verdict"] for x in rec["rows"]]
    rec["result"] = "FAIL" if "FAIL" in verdicts or len(verdicts) < len(names) else "PASS"
    tally = ", ".join(f"{verdicts.count(v)} {v}" for v in ("PASS", "INCONCLUSIVE", "FAIL") if v in verdicts)
    if a.out:
        out = pathlib.Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "api-contract.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        lines = [f"# API contract: {rec['result']}", "",
                 f"{rec['started']} to {rec['finished']}; base `{client.base}`; model `{model}`; "
                 f"{client.requests} requests; temperature 0" + ("; --dev (model checks not judged)" if a.dev else "") + ".", "",
                 "| Row | Verdict | Wall (s) | Note |", "|---|---|---:|---|"]
        lines += [f"| {x['row']} | {x['verdict']} | {x['wall_s']} | {x['note'].replace('|', '/')} |" for x in rec["rows"]]
        (out / "api-contract.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT: {rec['result']} api contract ({tally}; {client.requests} requests)"
          + (f" -> {a.out}" if a.out else ""), flush=True)
    return 0 if rec["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
