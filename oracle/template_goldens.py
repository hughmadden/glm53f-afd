#!/usr/bin/env python3
"""Chat-template goldens for crates/glm53f-tokenizer (the GLM-5.3-Flash renderer).

Renders a fixed set of conversations with the checkpoint's chat_template.jinja through
`transformers` (AutoTokenizer.apply_chat_template, tokenize=False) and records, per case,
goldens/template/<name>.json:

  messages, tools, kwargs      the inputs, as passed to apply_chat_template (tool-call
                               arguments are objects, as the template requires)
  add_generation_prompt        the flag
  rendered                     the reference text
  ids                          the reference tokenizer's ids for `rendered`
                               (tokenizers encode, add_special_tokens=False; checked equal to
                               apply_chat_template(tokenize=True))
  rendered_with_answer         when the last message is not an assistant turn: the render of
                               messages + one assistant turn with content ANSWER and no
                               reasoning, without a generation prompt. It pins the text the
                               template writes for a reasoning-free assistant turn, which is
                               what an engine's thinking-off prompt continues (the template
                               itself has no thinking switch).

plus manifest.json (versions, file digests, case list). The source keeps to ASCII.

Example (paths are placeholders):

  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v <checkpoint dir>:/model:ro -v "$PWD/oracle":/work \
      --entrypoint python3 glm53f-oracle:1 /work/template_goldens.py \
      --model /model --out /work/goldens/template
"""

import argparse
import copy
import hashlib
import json
import os
import sys

ANSWER = "ANSWER"


def u(*cps):
    return "".join(chr(c) for c in cps)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    """ASCII JSON with the angle brackets escaped (as the MiMo parser goldens are), so no golden
    file spells markup literally. The escape is built from chr(92) to keep this source free of
    escape sequences."""
    esc = chr(92) + "u00"
    text = json.dumps(obj, ensure_ascii=True, indent=1).replace("<", esc + "3c").replace(">", esc + "3e")
    with open(path, "w", encoding="ascii") as f:
        f.write(text + "\n")


def tag(name):
    return "<" + name + ">"


THINK, THINK_END = tag("think"), tag("/think")
TOOL_CALL = tag("tool_call")


def weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 14},
                    "units": {"type": "string", "enum": ["metric", "imperial"]},
                },
                "required": ["city"],
            },
        },
    }


def search_tool():
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                    "filters": {"type": "object", "properties": {"site": {"type": "string"}}},
                    "safe": {"type": "boolean"},
                    "weights": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["query"],
            },
        },
    }


def numbers_tool():
    # Numbers, escapes and non-ASCII that tojson must reproduce exactly.
    return {
        "type": "function",
        "function": {
            "description": "Edge cases: quotes \" and backslash \\ and a tab\t, a newline\n, "
                           + u(0x4E2D, 0x6587) + " " + u(0x1F600) + " and a control " + u(1) + u(0x7F) + ".",
            "name": "edge_numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "minimum": 0.5, "maximum": 1.0, "default": 1e-07,
                          "multipleOf": 0.1, "examples": [1e16, 1e15, 123456789.123, -0.0, 2.5e-05, 0.0001,
                                                          1.7976931348623157e308, 5e-324, 3.141592653589793]},
                    "b": {"type": "integer", "minimum": -9007199254740993, "maximum": 12345678901234567890},
                    "c": {"type": ["string", "null"], "default": None},
                    "d": {"type": "boolean", "default": False, "const": True},
                    "e": {"type": "array", "items": [], "prefixItems": [{}, {"type": "string"}]},
                },
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def flat_tool():
    # A tool without the {"type", "function"} wrapper: the template uses it as the function.
    return {"name": "ping", "description": "Ping a host.", "parameters": {"type": "object", "properties": {}}}


def deferred_tool():
    return {"type": "function", "function": {"name": "hidden", "description": "Deferred.",
                                             "parameters": {"type": "object"}, "defer_loading": True}}


def call(name, args, id_=None):
    tc = {"type": "function", "function": {"name": name, "arguments": args}}
    if id_ is not None:
        tc["id"] = id_
    return tc


def cases():
    c = []

    def add(name, messages, tools=None, kwargs=None, agp=True):
        c.append({"name": name, "messages": messages, "tools": tools, "kwargs": kwargs or {},
                  "add_generation_prompt": agp})

    user = {"role": "user", "content": "What is the weather in Paris?"}
    add("single_user", [{"role": "user", "content": "Hello!"}])
    add("system_user", [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hi there."}])
    add("effort_low", [user], kwargs={"reasoning_effort": "low"})
    add("effort_high", [user], kwargs={"reasoning_effort": "high"})
    add("effort_max", [user], kwargs={"reasoning_effort": "max"})
    add("effort_medium_is_max", [user], kwargs={"reasoning_effort": "medium"})
    add("effort_uppercase_is_max", [user], kwargs={"reasoning_effort": "High"})
    add("effort_none_is_max", [user], kwargs={"reasoning_effort": "none"})
    add("enable_thinking_false_is_ignored", [user], kwargs={"enable_thinking": False})
    add("no_generation_prompt", [user], agp=False)
    add("user_whitespace_kept", [{"role": "user", "content": "  padded\n\n question  \n"}])
    add("multi_turn_plain", [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ])
    history = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "reasoning_content": "Simple arithmetic: 4.", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    add("reasoning_kept_by_default", history)
    add("clear_thinking_drops_earlier", history, kwargs={"clear_thinking": True})
    add("clear_thinking_false_explicit", history, kwargs={"clear_thinking": False})
    add("think_in_content", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "  " + THINK + "inline reasoning" + THINK_END + "\n\nthe answer  "},
        {"role": "user", "content": "q2"},
    ])
    add("think_in_content_multiple", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "pre" + THINK + "a" + THINK + "b" + THINK_END + "mid" + THINK_END + "tail"},
        {"role": "user", "content": "q2"},
    ])
    add("reasoning_content_wins_over_content_tags", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "field", "content": THINK + "tag" + THINK_END + "answer"},
        {"role": "user", "content": "q2"},
    ])
    add("reasoning_content_empty_string", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "", "content": "answer"},
        {"role": "user", "content": "q2"},
    ])
    add("reasoning_content_null", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": None, "content": "answer"},
        {"role": "user", "content": "q2"},
    ])
    add("assistant_empty_and_whitespace", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": " \n\t "},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "q4"},
    ])
    add("assistant_python_whitespace", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": u(0x1C) + u(0x3000) + "x" + u(0xA0) + u(0x2029) + u(0x1F)},
        {"role": "user", "content": "q2"},
    ])
    add("assistant_last_no_prompt", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "r", "content": "a"},
    ], agp=False)
    add("tools_basic", [user], tools=[weather_tool()])
    add("tools_multiple", [user], tools=[weather_tool(), search_tool(), flat_tool()])
    add("tools_strict_and_deferred", [user], tools=[numbers_tool(), deferred_tool(), weather_tool()])
    add("tools_with_system", [{"role": "system", "content": "Use tools."}, user], tools=[search_tool()])
    add("tools_empty_list", [user], tools=[])
    tc_args = {"city": "Paris", "days": 3, "units": "metric"}
    add("tool_call_round_trip", [
        user,
        {"role": "assistant", "reasoning_content": "Need the weather tool.", "content": "",
         "tool_calls": [call("get_weather", tc_args, "call_1")]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 21, "sky": "clear"}'},
    ], tools=[weather_tool()])
    rich_args = {"query": "rust " + TOOL_CALL + " parsing", "max_results": 10, "filters": {"site": "example.org"},
                 "safe": True, "weights": [0.5, 1.0, 2, -3.25e-5], "note": None, "empty": "",
                 "nested": {"a": [1, {"b": [True, False, None]}], "s": "x\"y\\z" + u(0x4E2D)},
                 "text": "line1\nline2\t" + u(0x1F600)}
    add("tool_call_value_types", [
        {"role": "user", "content": "search"},
        {"role": "assistant", "content": "Searching now.", "tool_calls": [call("search", rich_args, "c9")]},
        {"role": "tool", "tool_call_id": "c9", "content": "no results"},
    ], tools=[search_tool()])
    add("parallel_calls_sorted_by_call_order", [
        user,
        {"role": "assistant", "content": None, "tool_calls": [
            call("get_weather", {"city": "Paris"}, "a1"), call("get_weather", {"city": "Rome"}, "b2"),
            call("get_weather", {"city": "Oslo"}, "c3")]},
        {"role": "tool", "tool_call_id": "c3", "content": "Oslo: 5C"},
        {"role": "tool", "tool_call_id": "a1", "content": "Paris: 21C"},
        {"role": "tool", "tool_call_id": "b2", "content": "Rome: 25C"},
    ], tools=[weather_tool()])
    add("parallel_calls_missing_ids_keep_order", [
        user,
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris"}, "a1"), call("get_weather", {"city": "Rome"}, "b2")]},
        {"role": "tool", "content": "second"},
        {"role": "tool", "tool_call_id": "a1", "content": "first"},
    ], tools=[weather_tool()])
    add("parallel_calls_duplicate_result_ids_keep_order", [
        user,
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris"}, "a1"), call("get_weather", {"city": "Rome"}, "b2")]},
        {"role": "tool", "tool_call_id": "b2", "content": "x"},
        {"role": "tool", "tool_call_id": "b2", "content": "y"},
    ], tools=[weather_tool()])
    add("parallel_calls_unknown_result_id_keep_order", [
        user,
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris"}, "a1"), call("get_weather", {"city": "Rome"}, "b2")]},
        {"role": "tool", "tool_call_id": "b2", "content": "x"},
        {"role": "tool", "tool_call_id": "zz", "content": "y"},
    ], tools=[weather_tool()])
    add("parallel_calls_duplicate_call_ids_keep_order", [
        user,
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris"}, "a1"), call("get_weather", {"city": "Rome"}, "a1")]},
        {"role": "tool", "tool_call_id": "a1", "content": "x"},
    ], tools=[weather_tool()])
    add("tool_calls_without_ids", [
        user,
        {"role": "assistant", "content": "", "tool_calls": [
            call("get_weather", {"city": "Paris"}), call("get_weather", {"city": "Rome"})]},
        {"role": "tool", "content": "Paris: 21C"},
        {"role": "tool", "content": "Rome: 25C"},
    ], tools=[weather_tool()])
    add("tool_call_empty_arguments", [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "", "tool_calls": [call("ping", {}, "p1")]},
        {"role": "tool", "tool_call_id": "p1", "content": "pong"},
    ], tools=[flat_tool()])
    add("tool_response_text_parts", [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "", "tool_calls": [call("ping", {}, "p1")]},
        {"role": "tool", "tool_call_id": "p1", "content": [{"type": "text", "text": "po"},
                                                            {"type": "text", "text": "ng"}]},
    ], tools=[flat_tool()])
    add("tool_first_message", [
        {"role": "tool", "content": "orphan result"},
        {"role": "user", "content": "q"},
    ])
    add("agent_loop_interleaved", [
        {"role": "system", "content": "Agent."},
        {"role": "user", "content": "Plan a trip."},
        {"role": "assistant", "reasoning_content": "Check weather first.", "content": "",
         "tool_calls": [call("get_weather", {"city": "Paris", "days": 2}, "t1")]},
        {"role": "tool", "tool_call_id": "t1", "content": "sunny"},
        {"role": "assistant", "reasoning_content": "Now search.", "content": "Weather is fine.",
         "tool_calls": [call("search", {"query": "Paris museums"}, "t2")]},
        {"role": "tool", "tool_call_id": "t2", "content": "Louvre"},
    ], tools=[weather_tool(), search_tool()])
    add("agent_loop_interleaved_clear_thinking", [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "reasoning_content": "old reasoning", "content": "old answer"},
        {"role": "user", "content": "Plan a trip."},
        {"role": "assistant", "reasoning_content": "Check weather first.", "content": "",
         "tool_calls": [call("get_weather", {"city": "Paris", "days": 2}, "t1")]},
        {"role": "tool", "tool_call_id": "t1", "content": "sunny"},
    ], tools=[weather_tool()], kwargs={"clear_thinking": True})
    add("images_as_markers", [
        {"role": "user", "content": [
            {"type": "text", "text": "Compare "},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": " and "},
            {"type": "image", "image": "data:image/png;base64,BBBB"},
            {"type": "text", "text": "."}]},
    ])
    add("media_parts", [
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            {"type": "text", "text": "describe"},
            {"type": "unknown_part", "text": "dropped"}]},
    ])
    add("assistant_parts_content", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "text", "text": " part one "}, {"type": "text", "text": "two "}]},
        {"role": "user", "content": "q2"},
    ])
    add("developer_role_dropped", [
        {"role": "developer", "content": "Developer instructions."},
        {"role": "user", "content": "q"},
    ])
    add("system_mid_conversation", [
        {"role": "user", "content": "a"},
        {"role": "system", "content": "Now answer in French."},
        {"role": "user", "content": "b"},
    ])
    add("unicode_content", [
        {"role": "user", "content": u(0x4F60, 0x597D) + " " + u(0x1F600) + " caf" + u(0xE9) + "\r\n"},
        {"role": "assistant", "reasoning_content": u(0x60F3), "content": u(0x597D, 0x7684)},
        {"role": "user", "content": u(0x0645, 0x0631, 0x062D, 0x0628, 0x0627)},
    ])
    add("markup_in_user_content", [
        {"role": "user", "content": "literal " + TOOL_CALL + "x" + tag("/tool_call") + " and " + THINK + " tags"},
    ])
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--out", required=True, help="output directory (goldens/template)")
    ap.add_argument("--image", default="", help="oracle image id, recorded in the manifest")
    args = ap.parse_args()

    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    at = AutoTokenizer.from_pretrained(args.model)
    raw = tokenizers.Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    os.makedirs(args.out, exist_ok=True)

    def render(messages, tools, kwargs, agp):
        return at.apply_chat_template(copy.deepcopy(messages), tools=copy.deepcopy(tools), tokenize=False,
                                      add_generation_prompt=agp, **kwargs)

    names = []
    for case in cases():
        msgs, tools, kwargs, agp = case["messages"], case["tools"], case["kwargs"], case["add_generation_prompt"]
        text = render(msgs, tools, kwargs, agp)
        ids = raw.encode(text, add_special_tokens=False).ids
        via = at.apply_chat_template(copy.deepcopy(msgs), tools=copy.deepcopy(tools), tokenize=True,
                                     add_generation_prompt=agp, return_dict=False, **kwargs)
        if isinstance(via, dict) or hasattr(via, "keys"):
            via = via["input_ids"]
        if list(via) != ids:
            raise SystemExit("apply_chat_template(tokenize=True) disagrees with encode for " + case["name"])
        out = dict(case)
        out["rendered"] = text
        out["ids"] = ids
        if msgs[-1]["role"] != "assistant":
            out["rendered_with_answer"] = render(msgs + [{"role": "assistant", "content": ANSWER}], tools, kwargs,
                                                 False)
        write_json(os.path.join(args.out, case["name"] + ".json"), out)
        names.append(case["name"])

    template_path = os.path.join(args.model, "chat_template.jinja")
    write_json(os.path.join(args.out, "manifest.json"), {
            "generator": "oracle/template_goldens.py",
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "python": sys.version.split()[0],
            "image": args.image,
            "files": {
                "chat_template.jinja": sha256_file(template_path),
                "tokenizer.json": sha256_file(os.path.join(args.model, "tokenizer.json")),
                "tokenizer_config.json": sha256_file(os.path.join(args.model, "tokenizer_config.json")),
            },
            "render": "AutoTokenizer.apply_chat_template(messages, tools=tools, tokenize=False, "
                      "add_generation_prompt=flag, **kwargs)",
            "answer": ANSWER,
            "cases": names,
        })
    print("wrote %d template cases to %s" % (len(names), args.out))


if __name__ == "__main__":
    main()
