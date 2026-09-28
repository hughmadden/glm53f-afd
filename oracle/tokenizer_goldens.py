#!/usr/bin/env python3
"""Tokenizer goldens for crates/glm53f-tokenizer.

Runs the reference `tokenizers` library on the checkpoint's tokenizer.json and records:

  cases.json            the input texts (id, text)
  ids.json              per case: token ids, pre-tokenizer pieces (as substrings of the
                        input), decode(ids) and decode(ids, skip_special_tokens=True)
  decode.json           decode of seeded random id sequences (single-byte tokens, the fixed
                        cases' tokens and added tokens mixed), both skip settings
  unicode_classes.json  the character classes the pre-tokenizer regex uses, probed over every
                        Unicode scalar value through the reference regex engine itself:
                        L (letters), N (numbers), S (the regex's whitespace) as code point
                        ranges, and the letters that match each contraction letter under the
                        regex's case-insensitive mode
  manifest.json         versions and file digests

Every file is written as ASCII JSON (non-ASCII as escapes). Python runs only inside the
oracle image, never on the host. Example (paths are placeholders):

  docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v <checkpoint dir>:/model:ro -v "$PWD/oracle":/work \
      --entrypoint python3 glm53f-oracle:1 /work/tokenizer_goldens.py \
      --model /model --out /work/goldens/tokenizer

The source file keeps to ASCII: non-ASCII test text is built from code points with chr().
"""

import argparse
import hashlib
import json
import os
import random
import sys
import unicodedata


def u(*cps):
    """A string from code points (keeps this file ASCII)."""
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


def bytes_to_unicode():
    """GPT-2 byte-level map (byte -> char), as tokenizers' ByteLevel uses it."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_MAP = bytes_to_unicode()


def bytelevel(s):
    return "".join(BYTE_MAP[b] for b in s.encode("utf-8"))


# Added-token literals, built from pieces so this file carries no literal markup.
def tag(name):
    return "<" + name + ">"


THINK, THINK_END = tag("think"), tag("/think")
TOOL_CALL, TOOL_CALL_END = tag("tool_call"), tag("/tool_call")
ARG_KEY, ARG_KEY_END = tag("arg_key"), tag("/arg_key")
ARG_VALUE, ARG_VALUE_END = tag("arg_value"), tag("/arg_value")
SYSTEM, USER, ASSISTANT, OBSERVATION = tag("|system|"), tag("|user|"), tag("|assistant|"), tag("|observation|")
ENDOFTEXT = tag("|endoftext|")
GMASK, SOP = "[gMASK]", tag("sop")
IMAGE, BOI, EOI = tag("|image|"), tag("|begin_of_image|"), tag("|end_of_image|")


def fixed_cases():
    c = []

    def add(name, text):
        c.append({"id": name, "text": text})

    add("empty", "")
    add("ascii_sentence", "Hello, world! This is a test of the tokenizer.")
    add("prose", "The quick brown fox jumps over the lazy dog. It was the best of times; it was "
                 "the worst of times. Numbers like 42, 3.14 and 1,000,000 appear too.")
    add("leading_space", " leading space")
    add("multi_space", "a  b   c    d     e")
    add("trailing_spaces", "trailing   ")
    add("spaces_before_newline", "line one   \nline two")
    add("newlines", "a\n\nb\n\n\nc\n")
    add("crlf", "a\r\nb\r\n\r\nc\r")
    add("tabs", "\tindented\n\t\tdouble\t\n")
    add("ws_newline_ws", "x \n \n  y")
    add("only_whitespace", "   \n\t  ")
    add("space_then_letters_digits", " a1 b22 c333 d4444")
    add("digits", "1 12 123 1234 12345 123456 1234567890")
    add("digits_mixed", "v1.2.3 3.14159 2026-09-28 0x1F 1e-7 -0.5")
    add("contractions", "I'm you're he's they've we'll she'd can't it's")
    add("contractions_upper", "I'M YOU'RE HE'S THEY'VE WE'LL SHE'D CAN'T IT'S")
    add("contraction_prefix", "'salut 'tree 'really 'velvet 'mmm 'llama 'done")
    add("apostrophes", "rock 'n' roll '' ''' 'x ' s")
    add("long_s_contraction", "it'" + u(0x17F) + " old'" + u(0x17F) + "x")
    add("punct_runs", "Wait... what?!?! (really) [x] {y} <z> ;:,. --- ***")
    add("punct_newlines", "end.\n\nNext;\r\n!!\n\n")
    add("code_python", "def f(x):\n    return x ** 2  # square\n\n\nprint(f(3))\n")
    add("code_rust", "fn main() {\n    let v: Vec<u32> = (0..10).collect();\n    println!(\"{:?}\", v);\n}\n")
    add("json_text", '{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}')
    add("markdown", "# Title\n\n- item one\n- item two\n\n```bash\nls -la\n```\n\n| a | b |\n|---|---|\n")
    add("url_email", "See https://example.com/path?q=1&r=two#frag or mail a.b@example.org")
    add("chinese", u(0x4F60, 0x597D, 0xFF0C, 0x4E16, 0x754C, 0xFF01, 0x4ECA, 0x5929, 0x5929, 0x6C14, 0x5F88,
                     0x597D, 0x3002))
    add("japanese", u(0x3053, 0x3093, 0x306B, 0x3061, 0x306F, 0x4E16, 0x754C, 0x3001, 0x30AB, 0x30BF, 0x30AB,
                      0x30CA, 0x3002))
    add("korean", u(0xC548, 0xB155, 0xD558, 0xC138, 0xC694) + " " + u(0xC138, 0xACC4))
    add("arabic", u(0x0645, 0x0631, 0x062D, 0x0628, 0x0627) + " " + u(0x0661, 0x0662, 0x0663, 0x0664, 0x0665))
    add("hindi_marks", u(0x0928, 0x092E, 0x0938, 0x094D, 0x0924, 0x0947) + " " + u(0x0915, 0x093F))
    add("latin_accents", "caf" + u(0xE9) + " cafe" + u(0x301) + " na" + u(0xEF) + "ve Stra" + u(0xDF) + "e "
        + u(0x3A9) + "mega")
    add("emoji", u(0x1F600) + " " + u(0x1F44D, 0x1F3FD) + " " + u(0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467)
        + " " + u(0x1F1E6, 0x1F1FA) + " " + u(0x31, 0xFE0F, 0x20E3))
    add("odd_spaces", "a" + u(0xA0) + "b" + u(0x3000) + "c" + u(0x2002) + "d" + u(0x2028) + "e" + u(0x2029)
        + "f" + u(0x85) + "g" + u(0x200B) + "h" + u(0xFEFF) + "i" + u(0x1680) + "j")
    add("controls", "a" + u(0) + "b" + u(7) + "c" + u(0x1B) + "d" + u(0x7F) + "e" + u(0x1C) + "f" + u(0x1F)
        + "g" + u(0x0B) + "h" + u(0x0C) + "i")
    add("space_before_odd_space", " " + u(0xA0) + "x " + u(0x3000) + " y")
    add("numerics", u(0xB2) + " " + u(0xBD) + " " + u(0x216B) + " " + u(0x2460) + u(0x2461) + " x"
        + u(0xB2, 0xB3) + "1234")
    add("symbols", u(0x2211, 0x222B, 0x2264, 0x2265, 0x2192) + " a" + u(0x2192) + "b " + u(0x20AC) + "5")
    add("modifier_letters", u(0x2B0) + "a" + u(0x2B7) + " " + u(0x3005) + u(0x4E00))
    add("specials_misc", u(0xE000) + " " + u(0xFFFD) + " " + u(0x10FFFF) + " " + u(0xFDD0) + "x" + u(0xFDD1))
    add("kelvin_dotted_i", "'" + u(0x212A) + " 'K " + u(0x130) + "stanbul " + u(0x131))
    add("long_letters", "a" * 1000)
    add("long_pair", "ab" * 300)
    add("long_digits", "1234567890" * 20)
    add("long_punct", "=-" * 200)
    add("long_spaces_then_word", " " * 100 + "x")
    add("long_newlines", "\n" * 50 + "x")
    add("mixed_ws_run", " \t \n \t\n  \r\n \t x")

    # Added tokens (special and non-special), adjacency and near-misses.
    add("chat_prefix", GMASK + SOP + SYSTEM + "Reasoning Effort: Max" + USER + "hi" + ASSISTANT + THINK)
    add("think_block", THINK + "reasoning here" + THINK_END + "answer")
    add("tool_call_markup", TOOL_CALL + "get_weather" + ARG_KEY + "city" + ARG_KEY_END + ARG_VALUE + "Paris"
        + ARG_VALUE_END + TOOL_CALL_END)
    add("nothink", "/nothink please and x/nothinky")
    add("partial_tags", "<think <|user| </think > <tool_call <" + THINK + "> [gMASK [sMASK]")
    add("tags_with_spaces", " " + THINK + " \n" + THINK_END + "  " + USER + " \n ")
    add("stop_tokens_inline", "done" + ENDOFTEXT + "more" + OBSERVATION + USER + "x")
    add("image_markers", BOI + IMAGE + EOI + "caption " + tag("|video|") + tag("|begin_of_video|"))
    add("adjacent_specials", "a" + USER + "b" + ASSISTANT + ASSISTANT + "c")
    add("observation_block", OBSERVATION + tag("tool_response") + '{"temp": 21}' + tag("/tool_response"))
    return c


def random_cases(rng):
    pools = [
        (30, [chr(x) for x in range(0x20, 0x7F)]),
        (8, [" ", " ", "\n", "\t", "\r\n", "  "]),
        (6, [chr(x) for x in range(0x4E00, 0x4E00 + 400)]),
        (4, [chr(x) for x in range(0x0400, 0x0450)]),
        (3, [chr(x) for x in range(0x0391, 0x03C9)]),
        (3, [chr(x) for x in range(0x0300, 0x0340)]),
        (3, [chr(x) for x in range(0x1F600, 0x1F650)]),
        (3, [chr(x) for x in range(0x0660, 0x066A)] + [chr(x) for x in (0xB2, 0xBD, 0x2460, 0x216B, 0xFF11)]),
        (2, [chr(x) for x in (0xA0, 0x3000, 0x2002, 0x2028, 0x2029, 0x85, 0x200B, 0xFEFF, 0x1C, 0x1F, 0x0B,
                              0x0C)]),
        (2, ["'s", "'T", "'re", "'LL", "'d", "'ve", "'M"]),
        (2, [THINK, THINK_END, TOOL_CALL, USER, GMASK, "/nothink", IMAGE, ARG_KEY, ENDOFTEXT]),
        (1, [chr(x) for x in (0xE000, 0xFFFD, 0x10FFFF, 0xFDD0, 0x17F, 0x212A, 0x130, 0x1D7D8, 0x10400)]),
        (2, [chr(x) for x in range(0xAC00, 0xAC00 + 200)]),
        (2, [chr(x) for x in range(0x0900, 0x0970)]),
    ]
    weights = [w for w, _ in pools]
    out = []
    for i in range(40):
        n = rng.randint(1, 160)
        s = []
        for _ in range(n):
            pool = rng.choices(pools, weights=weights)[0][1]
            s.append(rng.choice(pool))
        out.append({"id": "random_%02d" % i, "text": "".join(s)})
    words = ["the", "of", "and", "tokenizer", "model", "GLM", "Flash", "server", "request", "stream",
             "reasoning", "tool", "call", "value", "Hello", "World", "don't", "it's", "x86-64", "sm_120",
             "1024", "3.5", "(", ")", ",", ".", ":", ";", "\n", "  ", "\t", "--", "==", "->", "\"quoted\""]
    for i in range(20):
        n = rng.randint(5, 120)
        out.append({"id": "words_%02d" % i, "text": " ".join(rng.choice(words) for _ in range(n))})
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    out.append({"id": "base64_blob", "text": "".join(rng.choice(alphabet) for _ in range(600))})
    return out


def pieces_of(pre, text):
    """The reference pre-tokenizer's pieces as substrings of `text` (offsets checked)."""
    out = []
    for piece, (s, e) in pre.pre_tokenize_str(text):
        sub = text[s:e]
        if bytelevel(sub) != piece:
            raise SystemExit("pre-tokenizer offsets are not char offsets for %r" % text[:40])
        out.append(sub)
    if "".join(out) != text:
        raise SystemExit("pre-tokenizer pieces do not cover the text: %r" % text[:40])
    return out


def ranges(cps):
    out = []
    for cp in cps:
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def probe_classes(pre):
    """Classify every scalar value through the reference regex (see module doc)."""
    def n(s):
        return len(pre.pre_tokenize_str(s))

    letters, numbers, spaces = [], [], []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        if n("a" + ch) == 1:
            letters.append(cp)          # \p{L}+ continues through ch
        elif n("1" + ch) == 1:
            numbers.append(cp)          # \p{N}{1,3} continues through ch
        elif n("!" + ch) != 1 or ch in "\r\n":
            spaces.append(cp)           # ` ?[^\s\p{L}\p{N}]+[\r\n]*` stops before ch (CR/LF join it)
    # Contraction folds: compile each letter of the contraction alternative on its own,
    # case-insensitive as in the full pattern, through the same regex engine, and find every
    # letter it matches ("x'" + c + "x" splits into three pieces when `'c` matches). The
    # two-letter contractions are also probed against single letters, in case one letter
    # folds to both.
    from tokenizers import Regex, pre_tokenizers
    folds = {}
    for k in ("s", "t", "m", "d", "r", "e", "v", "l", "re", "ve", "ll"):
        split = pre_tokenizers.Split(Regex("(?i:'" + k + ")"), behavior="isolated")
        folds[k] = [cp for cp in letters if len(split.pre_tokenize_str("x'" + chr(cp) + "x")) == 3]
    return letters, numbers, spaces, folds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="checkpoint directory (tokenizer.json, tokenizer_config.json)")
    ap.add_argument("--out", required=True, help="output directory (goldens/tokenizer)")
    ap.add_argument("--image", default="", help="oracle image id, recorded in the manifest")
    ap.add_argument("--skip-unicode", action="store_true", help="keep an existing unicode_classes.json")
    args = ap.parse_args()

    import tokenizers
    import transformers

    tok_path = os.path.join(args.model, "tokenizer.json")
    tok = tokenizers.Tokenizer.from_file(tok_path)
    pre = tok.pre_tokenizer
    os.makedirs(args.out, exist_ok=True)

    rng = random.Random(20260928)
    fixed = fixed_cases()
    cases = fixed + random_cases(rng)
    seen = set()
    for c in cases:
        assert c["id"] not in seen, c["id"]
        seen.add(c["id"])

    outs = []
    for c in cases:
        text = c["text"]
        ids = tok.encode(text, add_special_tokens=False).ids
        with_special = tok.encode(text, add_special_tokens=True).ids
        if with_special != ids:
            raise SystemExit("add_special_tokens changed the ids for %s" % c["id"])
        outs.append({
            "id": c["id"],
            "ids": ids,
            "pieces": pieces_of(pre, text),
            "decoded": tok.decode(ids, skip_special_tokens=False),
            "decoded_skip_special": tok.decode(ids, skip_special_tokens=True),
        })

    # Decode goldens: seeded random id sequences that split UTF-8 across tokens. The ids are the
    # single-byte tokens, the added tokens and the tokens of the fixed cases above, so the decoded
    # text is made of this file's own text (and U+FFFD), never of arbitrary vocabulary.
    vocab = tok.get_vocab(with_added_tokens=False)
    byte_ids = sorted(vocab[BYTE_MAP[b]] for b in range(256))
    added = sorted(tok.get_added_tokens_decoder().keys())
    case_ids = sorted({i for o in outs[:len(fixed)] for i in o["ids"]} - set(added))
    decode_cases = []
    for i in range(300):
        seq = []
        for _ in range(rng.randint(1, 40)):
            r = rng.random()
            if r < 0.5:
                seq.append(rng.choice(byte_ids))
            elif r < 0.88:
                seq.append(rng.choice(case_ids))
            else:
                seq.append(rng.choice(added))
        decode_cases.append({
            "id": "decode_%03d" % i,
            "ids": seq,
            "decoded": tok.decode(seq, skip_special_tokens=False),
            "decoded_skip_special": tok.decode(seq, skip_special_tokens=True),
        })

    def dump(name, obj):
        write_json(os.path.join(args.out, name), obj)

    dump("cases.json", cases)
    dump("ids.json", outs)
    dump("decode.json", decode_cases)

    if not args.skip_unicode:
        letters, numbers, spaces, folds = probe_classes(pre)
        # Cross-check against this Python's Unicode database, for the record only.
        py_l = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF
                and unicodedata.category(chr(cp)).startswith("L")]
        py_n = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF
                and unicodedata.category(chr(cp)).startswith("N")]
        py_s = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF
                and (unicodedata.category(chr(cp)) in ("Zs", "Zl", "Zp") or cp in (9, 10, 11, 12, 13, 0x85))]

        def diff(a, b):
            sa, sb = set(a), set(b)
            return {"regex_only": ranges(sorted(sa - sb)), "unicodedata_only": ranges(sorted(sb - sa))}

        dump("unicode_classes.json", {
            "note": "Classes of the pre-tokenizer regex, probed over every scalar value with the reference "
                    "tokenizers library. L: 'a'+c is one piece. N: '1'+c is one piece. S: neither, and "
                    "'!'+c is two pieces (or c is CR/LF). contraction_folds: letters c for which the "
                    "case-insensitive contraction alternative matches c in place of the ASCII letter.",
            "L": ranges(letters),
            "N": ranges(numbers),
            "S": ranges(spaces),
            "contraction_folds": {k: v for k, v in folds.items()},
            "python_unicodedata_version": unicodedata.unidata_version,
            "vs_python_unicodedata": {"L": diff(letters, py_l), "N": diff(numbers, py_n), "S": diff(spaces, py_s)},
        })

    dump("manifest.json", {
        "generator": "oracle/tokenizer_goldens.py",
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "image": args.image,
        "files": {
            "tokenizer.json": sha256_file(tok_path),
            "tokenizer_config.json": sha256_file(os.path.join(args.model, "tokenizer_config.json")),
        },
        "encode": "Tokenizer.encode(text, add_special_tokens=False) (True gives the same ids for every case)",
        "decode": "Tokenizer.decode(ids, skip_special_tokens=False|True)",
        "cases": len(cases),
        "decode_cases": len(decode_cases),
    })
    print("wrote %d cases, %d decode cases to %s" % (len(cases), len(decode_cases), args.out))


if __name__ == "__main__":
    main()
