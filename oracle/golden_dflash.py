#!/usr/bin/env python3
"""Golden fixtures for the DFlash2 drafter of GLM-5.3-Flash, taken from the z-lab/dflash reference.

The reference is `dflash/model.py` of z-lab/dflash at the pinned commit (REFERENCE_COMMIT): the
`DFlash2DraftModel` (five Qwen3-style layers with two-tap dynamic convolutions, a candidate
selector) and the drafting loop `dflash_generate`. It is not installed in the oracle image: pass the
file with `--reference`; it is loaded by path and refused unless its sha256 is REFERENCE_SHA256. It
runs unmodified. The drafter loads with the reference's own `DFlash2DraftModel.from_pretrained`.

The drafter has no embedding or LM head: it uses the target's (`model.language_model.embed_tokens`
and `lm_head` of zai-org/GLM-5.3-Flash, read by name from `--target`).

Context features ("taps") are synthetic: the target model is not run here. A tap row is five
4096-wide target hidden states concatenated (20,480 values); `synth_taps` makes them from a counter
hash that `crates/glm53f-dflash` reimplements, so the fixtures record only their digests.

Each step reproduces one iteration of `dflash_generate` (model.py lines 233-257): the block
[anchor, mask x 7] at positions start..start+7, the new context rows at the positions before it,
the draft cache cropped back to `start`, then `DFlash2DraftModel.propose` at temperature 0.

Sets
    dflash-short   300 context rows, a draft; then 4 more rows (the anchor and 3 accepted drafts)
                   and a second draft. Every layer's intermediates, the context K/V, logits,
                   top-16, the selector's projection, path and scores.
    dflash-window  2,100 context rows (beyond the 2,048 window), a draft; then 6 more rows and a
                   second draft.
    dflash-native  the short case in the checkpoint's own dtype (BF16), with its difference from
                   the FP32 run.

Numerics as in golden_layers.py: FP32 is the primary contract (every parameter upcast exactly, the
modules run in FP32 on the CPU with pinned code paths and thread count); the native set shows the
reference's own BF16 rounding.

Usage
    golden_dflash.py generate --reference FILE --drafter DIR --target DIR --out DIR [--threads 16]
    golden_dflash.py verify DIR [DIR ...]
    golden_dflash.py compare DIR_A DIR_B

It runs in the oracle image next to golden_layers.py (whose helpers it imports), for example with
this directory mounted at /src and the reference file's directory at /ref:
    docker run --rm --network none --user "$(id -u):$(id -g)" -v "$PWD/oracle:/src:ro" \
      -v <dir holding model.py>:/ref:ro -v <drafter>:/weights/drafter:ro \
      -v <GLM-5.3-Flash with embed_tokens and lm_head>:/weights/target:ro -v "$PWD/oracle/goldens:/out" \
      --entrypoint python3 glm53f-oracle:1 /src/golden_dflash.py generate --reference /ref/model.py \
      --drafter /weights/drafter --target /weights/target --out /out
"""

from __future__ import annotations

import golden_layers as GL  # noqa: I001  (first: pins the CPU numerics environment before torch loads)

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import transformers

FORMAT_VERSION = 1
REFERENCE_REPO = "z-lab/dflash"
REFERENCE_COMMIT = "07ebd93db9f472af339b644bb70221ad8428328a"
REFERENCE_PATH = "dflash/model.py"
REFERENCE_SHA256 = "f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e"
DRAFTER_REPO = "incoai/GLM-5.3-Flash-DFlash2"
DRAFTER_REVISION = "bf582e4eacc1810f76656d1811693ff6c6737d2a"

EMBED = "model.language_model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
TAP_WIDTH = 5 * 4096
TOP_K = 16
TEMPERATURE_Q = 0.7

# Anchors are real token ids: the first decode tokens of the layer goldens ("Sunlight scatters off air").
CASES = {
    "dflash-short": {"context": 300, "seed0": 1, "anchor0": 29975, "accept": 3, "seed1": 2, "anchor1": 4145},
    "dflash-window": {"context": 2100, "seed0": 3, "anchor0": 1136, "accept": 5, "seed1": 4, "anchor1": 10170},
}


# --------------------------------------------------------------------------------------------------
# Synthetic context features, shared with crates/glm53f-dflash (src/synth.rs)


def splitmix64(x: np.ndarray) -> np.ndarray:
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def synth_taps(seed: int, start: int, rows: int) -> torch.Tensor:
    """Tap rows for positions start..start+rows, BF16 [rows, 20480].

    Element (pos, col): h = splitmix64(seed * 2^48 + pos * 20480 + col) (wrapping u64);
    v = (h >> 40) / 2^23 - 1, uniform in [-1, 1) and exact in f32; rounded to BF16 (nearest even)."""
    with np.errstate(over="ignore"):
        pos = np.arange(start, start + rows, dtype=np.uint64)[:, None]
        col = np.arange(TAP_WIDTH, dtype=np.uint64)[None, :]
        x = (np.uint64(seed) << np.uint64(48)) + pos * np.uint64(TAP_WIDTH) + col
        h = splitmix64(x)
    v = (h >> np.uint64(40)).astype(np.float64) / float(1 << 23) - 1.0
    return torch.from_numpy(v.astype(np.float32)).to(torch.bfloat16)


# --------------------------------------------------------------------------------------------------
# Reference, weights and the target stub


def load_reference(path: str):
    digest = GL.sha256_file(path)
    if digest != REFERENCE_SHA256:
        raise SystemExit(f"{path}: sha256 {digest}, expected {REFERENCE_SHA256} "
                         f"({REFERENCE_REPO} @ {REFERENCE_COMMIT} : {REFERENCE_PATH})")
    spec = importlib.util.spec_from_file_location("dflash_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Target(nn.Module):
    """What the reference reads from the target model: its input embedding and `lm_head`."""

    def __init__(self, embed: torch.Tensor, head: torch.Tensor):
        super().__init__()
        self.embed_tokens = nn.Embedding(embed.shape[0], embed.shape[1], _weight=embed, _freeze=True)
        self.lm_head = nn.Linear(head.shape[1], head.shape[0], bias=False)
        self.lm_head.weight = nn.Parameter(head, requires_grad=False)

    def get_input_embeddings(self):
        return self.embed_tokens


def load_drafter(ref, drafter_dir: str, dtype: torch.dtype):
    model, info = ref.DFlash2DraftModel.from_pretrained(drafter_dir, dtype=dtype, output_loading_info=True)
    bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    if bad:
        raise SystemExit(f"drafter load: {bad}")
    model.eval()
    return model


# --------------------------------------------------------------------------------------------------
# Recording


class Recorder:
    """Per-layer intermediates of one draft forward, from hooks and wrappers on the reference's modules."""

    def __init__(self, model):
        self.out: dict[str, torch.Tensor] = {}
        self.on = False
        self.handles = []
        self.wrapped = []
        for i, layer in enumerate(model.layers):
            p = f"L{i}."
            self._hook_pre(layer, p + "in")
            self._hook(layer, p + "out")
            self._hook(layer.input_layernorm, p + "attn_norm")
            self._hook(layer.attention_conv.kernel_projection, p + "attn_dyn")
            self._hook(layer.self_attn, p + "attn_raw", lambda o: o[0])
            self._hook(layer.post_attention_layernorm, p + "mlp_norm")
            self._hook(layer.mlp_conv.kernel_projection, p + "mlp_dyn")
            self._hook(layer.mlp, p + "mlp_raw")
            self._wrap_conv(layer.attention_conv, p + "attn_conv_in", p + "attn_conv_out")
            self._wrap_conv(layer.mlp_conv, p + "mlp_conv_in", p + "mlp_conv_out")
        self._hook(model.hidden_norm, "ctx_features")
        self._hook(model.norm, "final")

    def _put(self, key, t):
        if self.on:
            self.out[key] = t.detach()[0].clone()

    def _hook(self, mod, key, fn=lambda o: o):
        self.handles.append(mod.register_forward_hook(lambda m, a, o: self._put(key, fn(o))))

    def _hook_pre(self, mod, key):
        def pre(m, args, kwargs):
            self._put(key, kwargs["hidden_states"])
        self.handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))

    def _wrap_conv(self, conv, key_prepare, key_finish):
        prepare, finish = conv.prepare, conv.finish

        def prep(hidden):
            out, kernel = prepare(hidden)
            self._put(key_prepare, out)
            return out, kernel

        def fin(hidden, dynamic):
            out = finish(hidden, dynamic)
            self._put(key_finish, out)
            return out

        conv.prepare, conv.finish = prep, fin

    def take(self) -> dict[str, torch.Tensor]:
        out, self.out = self.out, {}
        return out


def lattice(selector, draft_hidden, anchor, cand_ids, unary):
    """SGLang's `_score_edges`: score[e, p, c] = unary[e, c] + <A[pred[e, p]] * proj(h[e]), B[cand[e, c]]>,
    pred = cand[e - 1] and the anchor for slot 0 (sgl-project/sglang PR 36708 head, models/dflash.py
    lines 909-931). Written here from that definition, on the reference's modules."""
    h = selector.hidden_projection(draft_hidden)                        # [slots, r]
    keys = selector.successor_codebook(cand_ids)                        # [slots, k, r]
    pred_ids = torch.cat([anchor.expand(1, TOP_K), cand_ids[:-1]], 0)   # [slots, k]
    preds = selector.predecessor_codebook(pred_ids)                     # [slots, k, r]
    return unary[:, None, :] + torch.einsum("lpr,lcr->lpc", preds * h[:, None, :], keys)


def draft_step(ref, model, target, cache, taps, start, anchor, rec):
    """One iteration of dflash_generate (model.py 233-257): the new context rows `taps` sit at
    positions start - len(taps) .. start - 1, the block [anchor, mask x 7] at start .. start + 7."""
    block = model.block_size
    ids = torch.full((1, block), int(model.mask_token_id), dtype=torch.long)
    ids[0, 0] = anchor
    noise = ref._raw_input_embeddings(target, ids, float(ref._draft_value(model.config, "input_embedding_scale", 1.0)))
    positions = torch.arange(start - taps.shape[0], start + block)[None]
    dtype = model.fc.weight.dtype
    rec.on = True
    hidden = model(target_hidden=taps[None].to(dtype), noise_embedding=noise.to(dtype), position_ids=positions,
                   past_key_values=cache, use_cache=True)
    rec.on = False
    draft_hidden = hidden[:, 1 - block:, :]
    kv_block = [(layer.keys[0, :, -block:].transpose(0, 1).clone(), layer.values[0, :, -block:].transpose(0, 1).clone())
                for layer in cache.layers]                                   # [8, heads, 128] each, before the crop
    ref._crop_to(cache, start)
    head = ref._output_head(target)
    tokens, cands, _ = model.propose(draft_hidden, ids[:, 0], head, 0.0)

    logits = model.compute_logits(draft_hidden, head)[0]                    # [7, vocab]
    sel = model.candidate_selector
    unary, cand_ref = torch.topk(logits, sel.top_k, dim=-1, sorted=False)    # the reference's own call
    if not torch.equal(cand_ref, cands[0]):
        raise RuntimeError("propose and the recomputed top-k disagree")
    vals, cand_sorted = torch.topk(logits, sel.top_k, dim=-1, sorted=True)
    path = tokens[0]
    anchor_t = torch.tensor([anchor])
    lat = lattice(sel, draft_hidden[0], anchor_t, cand_sorted, vals)          # [7, 16, 16], sorted candidates
    # Walk the lattice: slot 0 from the anchor row, then the row of the previous choice.
    prev, walk, path_scores = 0, [], []
    for e in range(lat.shape[0]):
        row = lat[e, prev]
        prev = int(torch.argmax(row))
        walk.append(int(cand_sorted[e, prev]))
        path_scores.append(row)
    walk = torch.tensor(walk)
    path_scores = torch.stack(path_scores)
    q = torch.softmax(path_scores.float() / TEMPERATURE_Q, dim=-1)
    rec_out = rec.take()
    return {
        "hidden": hidden[0], "draft_hidden": draft_hidden[0], "logits": logits, "kv_block": kv_block,
        "topk_vals": vals, "topk_ids": cand_sorted, "ref_candidates": cand_ref, "ref_unary": unary,
        "hproj": sel.hidden_projection(draft_hidden[0]), "path": path, "walk": walk, "lattice": lat,
        "path_scores": path_scores, "q": q, "rec": rec_out, "noise": noise[0], "positions": positions[0],
    }


LAYER_KEYS = {
    "in": "block hidden state entering the layer [8, 4096]",
    "attn_norm": "input_layernorm output (attention_conv.prepare input)",
    "attn_dyn": "attention_conv.kernel_projection(attn_norm) [8, 1024] = [side 2][tap 2][group 256]; side 0 for prepare, 1 for finish",
    "attn_conv_in": "attention_conv.prepare output: the q/k/v projections' input",
    "attn_raw": "self_attn output after o_proj (attention_conv.finish input)",
    "attn_conv_out": "attention_conv.finish output, added to the residual",
    "mlp_norm": "post_attention_layernorm output (mlp_conv.prepare input)",
    "mlp_dyn": "mlp_conv.kernel_projection(mlp_norm) [8, 1024]",
    "mlp_conv_in": "mlp_conv.prepare output: the MLP input",
    "mlp_raw": "MLP output down(silu(gate(x)) * up(x)) (mlp_conv.finish input)",
    "mlp_conv_out": "mlp_conv.finish output, added to the residual",
    "out": "block hidden state leaving the layer",
}


def write_step(w, p: str, r: dict, cache, ctx_rows: int, full: bool) -> None:
    """Record step `p` ("s0", "s1", ...). `ctx_rows`: how many of the newest context rows' K/V to keep."""
    w.add(f"{p}.positions", r["positions"].to(torch.int32), "draft forward position ids: new context rows, then the block")
    w.add(f"{p}.block_embed", r["noise"].float(), "block input: target embedding rows of [anchor, mask x 7] [8, 4096]")
    if full:
        for i in range(len(cache.layers)):
            for key, desc in LAYER_KEYS.items():
                w.add(f"{p}.L{i}.{key}", r["rec"][f"L{i}.{key}"].float(), f"layer {i}: {desc}")
            k, v = r["kv_block"][i]
            w.add(f"{p}.L{i}.k_block", k.float(), f"layer {i}: block keys after k_norm and RoPE [8, kv_heads, 128]")
            w.add(f"{p}.L{i}.v_block", v.float(), f"layer {i}: block values [8, kv_heads, 128]")
    else:
        for i in range(len(cache.layers)):
            w.add(f"{p}.L{i}.out", r["rec"][f"L{i}.out"].float(), f"layer {i}: block hidden state leaving the layer")
    if "ctx_features" in r["rec"] and full:
        w.add(f"{p}.ctx_features", r["rec"]["ctx_features"].float(),
              "hidden_norm(fc(taps)) for the new context rows [rows, 4096]")
    if ctx_rows:
        ks = torch.stack([layer.keys[0, :, -ctx_rows:].transpose(0, 1) for layer in cache.layers])
        vs = torch.stack([layer.values[0, :, -ctx_rows:].transpose(0, 1) for layer in cache.layers])
        w.add(f"{p}.ctx_k", ks.float(), f"draft cache after the crop: keys of the newest {ctx_rows} context rows "
              "[layers, rows, kv_heads, 128], after k_norm and RoPE at their positions")
        w.add(f"{p}.ctx_v", vs.float(), f"draft cache after the crop: values of the newest {ctx_rows} context rows")
    w.add(f"{p}.final", r["hidden"].float(), "model output: norm(h) for the 8 block rows [8, 4096]")
    w.add(f"{p}.logits", r["logits"].float(), "compute_logits of rows 1..7 through the target lm_head [7, vocab], FP32")
    w.add(f"{p}.topk_vals", r["topk_vals"].float(), "top-16 logits per row, descending (torch.topk sorted=True)")
    w.add(f"{p}.topk_ids", r["topk_ids"].to(torch.int32), "top-16 token ids in the order of topk_vals")
    w.add(f"{p}.ref_candidates", r["ref_candidates"].to(torch.int32),
          "the reference's candidates in its own order (torch.topk sorted=False)")
    w.add(f"{p}.hproj", r["hproj"].float(), "candidate_selector.hidden_projection of rows 1..7 [7, 256]")
    w.add(f"{p}.lattice", r["lattice"].float(),
          "SGLang edge scores [slot, predecessor, candidate] over the sorted candidates (slot 0: every row is the anchor)")
    w.add(f"{p}.path", r["path"].to(torch.int32), "DFlash2DraftModel.propose at temperature 0: the 7 drafts")
    w.add(f"{p}.path_scores", r["path_scores"].float(), "the lattice row the greedy walk used at each slot [7, 16]")
    w.add(f"{p}.q_t07", r["q"].float(), f"softmax(path_scores / {TEMPERATURE_Q}) per slot: a sampled walk's q along this path")


def rel(a: torch.Tensor, ref: torch.Tensor) -> dict:
    return GL.rel_err(a, ref)


# --------------------------------------------------------------------------------------------------
# generate


def run_case(ref, model, target, name: str, case: dict, full: bool):
    cache = ref._make_cache(model.config)
    rec = Recorder(model)
    ctx0 = synth_taps(case["seed0"], 0, case["context"])
    start = case["context"]
    s0 = draft_step(ref, model, target, cache, ctx0, start, case["anchor0"], rec)
    kept = [case["anchor0"]] + [int(t) for t in s0["path"][: case["accept"]]]
    snap0 = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    ctx1 = synth_taps(case["seed1"], start, len(kept))
    s1 = draft_step(ref, model, target, cache, ctx1, start + len(kept), case["anchor1"], rec)
    for h in rec.handles:
        h.remove()
    return {"ctx0": ctx0, "ctx1": ctx1, "s0": s0, "s1": s1, "kept": kept, "cache": cache, "snap0": snap0}


class _CacheView:
    """The layers of a cache snapshot, for write_step."""

    def __init__(self, snap):
        self.layers = [type("L", (), {"keys": k, "values": v}) for k, v in snap]


def generate(args) -> int:
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if sys.byteorder != "little":
        raise SystemExit("fixtures are little-endian; run on a little-endian host")
    if transformers.__version__ != GL.TRANSFORMERS_VERSION:
        raise SystemExit(f"need transformers {GL.TRANSFORMERS_VERSION}, found {transformers.__version__}")

    ref = load_reference(args.reference)
    ck = GL.Checkpoint([args.target])
    embed = ck.get(EMBED)
    head = ck.get(LM_HEAD)
    with open(os.path.join(args.drafter, "config.json")) as f:
        draft_cfg = json.load(f)
    tf = GL.reference_info()
    qwen3 = sys.modules[ref.Qwen3MLP.__module__].__file__
    source = {
        "reference": {"repo": REFERENCE_REPO, "commit": REFERENCE_COMMIT, "path": REFERENCE_PATH,
                      "sha256": REFERENCE_SHA256, "classes": "DFlash2DraftModel, CandidateSelector, "
                      "GroupedDynamicCausalConv, Qwen3DFlashDecoderLayer, Qwen3DFlashAttention; dflash_generate's "
                      "drafting step",
                      "transformers": {"version": tf["version"], "wheel_sha256": tf["wheel_sha256"],
                                       "qwen3_modeling_sha256": GL.sha256_file(qwen3),
                                       "attention": "sdpa (the reference's own choice) with its explicit boolean mask"}},
        "drafter": {"repo": DRAFTER_REPO, "revision": DRAFTER_REVISION,
                    "config_sha256": GL.sha256_file(os.path.join(args.drafter, "config.json")),
                    "weights_sha256": GL.sha256_file(os.path.join(args.drafter, "model.safetensors"))},
        "target": {"repo": ck.meta.get("source_repo"), "revision": ck.meta.get("source_revision"),
                   "tensors": {n: {"dtype": ck.read[n][0], "shape": ck.read[n][1], "sha256": ck.read[n][2]}
                               for n in (EMBED, LM_HEAD)}},
        "runtime": GL.runtime_info(args.threads),
        "script_sha256": GL.sha256_file(os.path.abspath(__file__)),
        "helpers_sha256": {"golden_layers.py": GL.sha256_file(os.path.abspath(GL.__file__))},
        "taps": {"rule": synth_taps.__doc__.strip(), "width": TAP_WIDTH},
    }
    results = {}
    with torch.no_grad():
        model = load_drafter(ref, args.drafter, torch.float32)
        target = Target(embed.float(), head.float())
        layer0 = model.layers[0].self_attn
        facts = {"is_causal": bool(layer0.is_causal), "sliding_window": layer0.sliding_window,
                 "block_size": int(model.block_size), "mask_token_id": int(model.mask_token_id),
                 "target_layer_ids": list(model.target_layer_ids), "top_k": int(model.candidate_selector.top_k),
                 "draft_config": draft_cfg.get("dflash_config")}
        print(f"reference drafter: {facts}", flush=True)
        for name, case in CASES.items():
            t0 = time.monotonic()
            results[name] = run_case(ref, model, target, name, case, full=(name == "dflash-short"))
            print(f"  {name}: {time.monotonic() - t0:.1f} s", flush=True)
        inv_freq = model.rotary_emb.inv_freq.clone()
        del model, target
        native = load_drafter(ref, args.drafter, torch.bfloat16)
        target_bf16 = Target(embed, head)
        t0 = time.monotonic()
        nat = run_case(ref, native, target_bf16, "dflash-short", CASES["dflash-short"], full=False)
        print(f"  dflash-native: {time.monotonic() - t0:.1f} s", flush=True)

    for name, case in CASES.items():
        r = results[name]
        full = name == "dflash-short"
        notes = {
            "facts": facts,
            "case": case,
            "kept_rows": r["kept"],
            "taps_sha256": {"s0": GL.sha256_bytes(GL.tensor_bytes(r["ctx0"])),
                            "s1": GL.sha256_bytes(GL.tensor_bytes(r["ctx1"]))},
            "steps": (f"s0: context rows 0..{case['context'] - 1} (seed {case['seed0']}), anchor {case['anchor0']} at "
                      f"{case['context']}; s1: rows {case['context']}..{case['context'] + case['accept']} (seed "
                      f"{case['seed1']}: the anchor and {case['accept']} accepted drafts), anchor {case['anchor1']} at "
                      f"{case['context'] + case['accept'] + 1}"),
            "walk_equals_propose": [bool(torch.equal(r[s]["walk"], r[s]["path"])) for s in ("s0", "s1")],
            "padding_ids_in_top16": [int((r[s]["topk_ids"] >= 154856).sum()) for s in ("s0", "s1")],
            "layout": "row-major little-endian; K/V heads are the draft's 8 KV heads of 128",
        }
        w = GL.SetWriter(args.out, name, source, notes)
        w.add("rope.inv_freq", inv_freq.float(), "Qwen3RotaryEmbedding.inv_freq of the drafter (64 values, FP32)")
        mask_row = embed[int(facts["mask_token_id"])].float()
        w.add("mask_embed", mask_row, "target embedding row of mask_token_id 154856 (FP32 of the BF16 row)")
        write_step(w, "s0", r["s0"], _CacheView(r["snap0"]), case["context"] if full else 0, full)
        write_step(w, "s1", r["s1"], r["cache"], len(r["kept"]), full)
        w.close()

    base = results["dflash-short"]
    diffs = {}
    for s in ("s0", "s1"):
        a, b = nat[s], base[s]
        diffs[s] = {"final": rel(a["hidden"], b["hidden"]), "logits": rel(a["logits"], b["logits"]),
                    "top16_same_set": [bool(set(a["topk_ids"][e].tolist()) == set(b["topk_ids"][e].tolist()))
                                       for e in range(a["topk_ids"].shape[0])],
                    "path_same": [bool(x) for x in (a["path"] == b["path"]).tolist()]}
    notes = {"facts": facts, "case": CASES["dflash-short"], "diff_vs_fp32": diffs,
             "dtype": "native: every drafter tensor BF16 as stored; target embedding and lm_head BF16; the reference's "
                      "Qwen3RMSNorm and rotary upcast internally; compute_logits returns BF16 logits (recorded as FP32)"}
    w = GL.SetWriter(args.out, "dflash-native", source, notes)
    for s in ("s0", "s1"):
        r = nat[s]
        for i in range(5):
            w.add(f"{s}.L{i}.out", r["rec"][f"L{i}.out"].float(), f"layer {i}: block hidden state leaving the layer (BF16)")
        w.add(f"{s}.final", r["hidden"].float(), "model output norm(h), BF16 [8, 4096]")
        w.add(f"{s}.logits", r["logits"].float(), "logits of rows 1..7 (BF16 values) [7, vocab]")
        w.add(f"{s}.topk_ids", r["topk_ids"].to(torch.int32), "top-16 ids, descending logit")
        w.add(f"{s}.path", r["path"].to(torch.int32), "propose at temperature 0")
    w.close()
    print(json.dumps(diffs, indent=1))
    print(f"done in {time.monotonic() - started:.0f} s; peak RSS "
          f"{GL.resource.getrusage(GL.resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="run the reference and write golden sets")
    g.add_argument("--reference", required=True, help=f"{REFERENCE_PATH} of {REFERENCE_REPO} @ {REFERENCE_COMMIT}")
    g.add_argument("--drafter", required=True, help=f"{DRAFTER_REPO} checkpoint directory (config.json, model.safetensors)")
    g.add_argument("--target", required=True,
                   help="GLM-5.3-Flash directory holding embed_tokens and lm_head (safetensors + index)")
    g.add_argument("--out", required=True, help="output directory (one subdirectory per golden set)")
    g.add_argument("--threads", type=int, default=16, help="CPU threads (default 16; part of the numerics contract)")
    v = sub.add_parser("verify", help="check every .bin against its manifest digest")
    v.add_argument("dirs", nargs="+")
    c = sub.add_parser("compare", help="compare two output directories digest by digest")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    return {"generate": generate, "verify": GL.verify, "compare": GL.compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
