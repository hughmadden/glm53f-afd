#!/usr/bin/env python3
"""Self-test for golden_layers.py on a tiny random GLM-5.3-Flash-format checkpoint.

The real checkpoint is too large to load as one model here, so golden_layers.py builds the reference
modules one layer at a time and loads their weights itself. This test checks that shortcut against
the reference's own loader and model, on a checkpoint small enough for both:

1. Write a tiny random checkpoint in the official on-disk format: the same tensor names, FP8 E4M3
   weights with 128 x 128 `weight_scale_inv` blocks where the official checkpoint has them, BF16 and
   F32 tensors where it has those, per-expert tensors, and separate q/k/v short-convolution weights.
2. Run `golden_layers.py generate` on it (FP32 and native sets).
3. Load it with `Glm5NextForConditionalGeneration.from_pretrained`, twice:
   - with dtype=float32, then `.float()` (the loader keeps some renamed tensors in BF16);
   - with the default dtype (native: BF16, FP8 dequantised because there is no GPU).
4. Require bit-identical results:
   - every parameter golden_layers.py loads equals the loader's tensor (value and, native, dtype),
     experts included;
   - FP32: every recorded layer's output streams at the prompt and at each decode step, and the
     logits, equal the whole model's, run with its own cache;
   - native: each recorded layer, fed the whole native model's inputs for that layer, reproduces
     that model's outputs.

Usage (inside the oracle image, no weights needed):
    python3 /oracle/selftest_tiny.py [--workdir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace

import golden_layers as gl  # pins the CPU numerics before torch loads

import torch  # noqa: E402

HIDDEN = 256
TINY_TEXT = {
    "model_type": "glm5_next_text",
    "dtype": "bfloat16",
    "vocab_size": 512,
    "hidden_size": HIDDEN,
    "intermediate_size": 384,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_shared_experts": 1,
    "n_routed_experts": 16,
    "num_experts_per_tok": 4,
    "routed_scaling_factor": 2.5,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "kv_lora_rank": 128,
    "q_lora_rank": 128,
    "qk_rope_head_dim": 0,
    "qk_nope_head_dim": 64,
    "qk_head_dim": 64,
    "v_head_dim": 64,
    "head_dim": 0,
    "index_n_heads": 2,
    "index_head_dim": 64,
    "index_topk": 8,
    "index_kpool": 4,
    "index_kpool_always_select_tail": True,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "rms_norm_eps": 1e-5,
    "swiglu_limit": 10.0,
    "hidden_act": "silu",
    "first_k_dense_replace": 3,
    "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention", "linear_attention"],
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 2,
    "indexer_types": ["full"] * 5,
    "linear_attn_config": {
        "num_heads": 4,
        "head_dim": 64,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 1, 2, 4],
        "full_attn_layers": [3],
    },
    "max_position_embeddings": 4096,
    "pad_token_id": 0,
    "eos_token_id": [1],
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "moe_router_dtype": "float32",
    "use_cache": True,
}
TINY_VISION = {
    "depth": 1,
    "hidden_size": 64,
    "num_heads": 2,
    "intermediate_size": 128,
    "out_hidden_size": HIDDEN,
    "projection_intermediate_size": 128,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "rope_parameters": {"rope_type": "axial", "rope_theta": 10000.0},
}


def _quantize_fp8(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = w.shape
    sr, sc = math.ceil(rows / block), math.ceil(cols / block)
    scale_inv = torch.empty(sr, sc, dtype=torch.float32)
    q = torch.empty(rows, cols, dtype=torch.float8_e4m3fn)
    for i in range(sr):
        for j in range(sc):
            blk = w[i * block:(i + 1) * block, j * block:(j + 1) * block].float()
            s = blk.abs().amax().clamp(min=1e-12) / 448.0
            scale_inv[i, j] = s
            q[i * block:(i + 1) * block, j * block:(j + 1) * block] = (blk / s).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale_inv


def write_tiny_checkpoint(out_dir: str, seed: int = 1234) -> None:
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    t = TINY_TEXT
    h, lin = t["hidden_size"], t["linear_attn_config"]
    kh, kd = lin["num_heads"], lin["head_dim"]
    qkv = kh * kd
    tensors: dict[str, torch.Tensor] = {}
    keep_modules: list[str] = ["lm_head", "model.embed_tokens", "model.norm"]

    def rnd(*shape, std=0.02):
        return torch.randn(*shape, generator=g) * std

    def put(name, value, dtype=torch.bfloat16):
        tensors[name] = value.to(dtype).contiguous()

    def put_linear(prefix, out_f, in_f, fp8, std=None):
        std = std if std is not None else 1.0 / math.sqrt(in_f)
        w = rnd(out_f, in_f, std=std)
        if fp8:
            q, s = _quantize_fp8(w)
            tensors[prefix + ".weight"] = q
            tensors[prefix + ".weight_scale_inv"] = s
        else:
            put(prefix + ".weight", w)
            keep_modules.append(prefix.replace("model.language_model.", "model."))

    put("model.language_model.embed_tokens.weight", rnd(t["vocab_size"], h, std=1.0))
    put("model.language_model.norm.weight", 1.0 + rnd(h, std=0.1))
    put("lm_head.weight", rnd(t["vocab_size"], h, std=1.0 / math.sqrt(h)))
    hc = t["hc_mult"]
    mix = (2 + hc) * hc
    for i in range(t["num_hidden_layers"]):
        p = f"model.language_model.layers.{i}."
        for site in ("attn", "ffn"):
            put(p + f"hc_{site}_fn", rnd(mix, hc * h, std=0.02))
            put(p + f"hc_{site}_base", rnd(mix, std=0.5), torch.float32)
            put(p + f"hc_{site}_scale", 1.0 + rnd(3, std=0.1), torch.float32)
        put(p + "input_layernorm.weight", 1.0 + rnd(h, std=0.1))
        put(p + "post_attention_layernorm.weight", 1.0 + rnd(h, std=0.1))
        a = p + "self_attn."
        if t["layer_types"][i] == "linear_attention":
            for n in ("q_proj", "k_proj", "v_proj"):
                put_linear(a + n, qkv, h, fp8=False)
            put_linear(a + "o_proj", h, qkv, fp8=False)
            put_linear(a + "b_proj", kh, h, fp8=False)
            put_linear(a + "f_a_proj", kd, h, fp8=False)
            put_linear(a + "f_b_proj", qkv, kd, fp8=False)
            put_linear(a + "g_a_proj", kd, h, fp8=False)
            put_linear(a + "g_b_proj", qkv, kd, fp8=False)
            for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
                put(a + n + ".weight", rnd(qkv, 1, 4, std=0.5))
            put(a + "o_norm.weight", 1.0 + rnd(kd, std=0.1))
            put(a + "A_log", torch.empty(kh).uniform_(0.0, 2.0, generator=g), torch.float32)
            put(a + "dt_bias", rnd(qkv, std=1.0), torch.float32)
        else:
            nh, dn, dv = t["num_attention_heads"], t["qk_nope_head_dim"], t["v_head_dim"]
            ql, kl = t["q_lora_rank"], t["kv_lora_rank"]
            put_linear(a + "q_a_proj", ql, h, fp8=True)
            put_linear(a + "q_b_proj", nh * dn, ql, fp8=True)
            put_linear(a + "kv_a_proj_with_mqa", kl, h, fp8=True)
            put_linear(a + "o_proj", h, nh * dv, fp8=True)
            put_linear(a + "kv_b_proj", nh * (dn + dv), kl, fp8=False)
            put(a + "q_a_layernorm.weight", 1.0 + rnd(ql, std=0.1))
            put(a + "kv_a_layernorm.weight", 1.0 + rnd(kl, std=0.1))
            ih, idim = t["index_n_heads"], t["index_head_dim"]
            put_linear(a + "indexer.wq_b", ih * idim, ql, fp8=False)
            put_linear(a + "indexer.wk", idim, h, fp8=False)
            put_linear(a + "indexer.weights_proj", ih, h, fp8=False)
            put(a + "indexer.k_norm.weight", 1.0 + rnd(idim, std=0.1))
            put(a + "indexer.k_norm.bias", rnd(idim, std=0.1))
            put(a + "indexer.index_kpool_compress_ape", rnd(t["index_kpool"], idim, std=0.5))
            put(a + "indexer.index_kpool_compress_gate", rnd(idim, h, std=1.0 / math.sqrt(h)))
        m = p + "mlp."
        if t["mlp_layer_types"][i] == "dense":
            put_linear(m + "gate_proj", t["intermediate_size"], h, fp8=True)
            put_linear(m + "up_proj", t["intermediate_size"], h, fp8=True)
            put_linear(m + "down_proj", h, t["intermediate_size"], fp8=True)
        else:
            e, inter = t["n_routed_experts"], t["moe_intermediate_size"]
            put_linear(m + "gate", e, h, fp8=False)
            put(m + "gate.e_score_correction_bias", rnd(e, std=0.05), torch.float32)
            for n, (o, i_) in {"gate_proj": (inter, h), "up_proj": (inter, h), "down_proj": (h, inter)}.items():
                put_linear(m + "shared_experts." + n, o, i_, fp8=True)
            for x in range(e):
                for n, (o, i_) in {"gate_proj": (inter, h), "up_proj": (inter, h), "down_proj": (h, inter)}.items():
                    put_linear(m + f"experts.{x}." + n, o, i_, fp8=True)
    os.makedirs(out_dir, exist_ok=True)
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"source_repo": "tiny-random", "source_revision": f"seed-{seed}"},
                   "weight_map": {k: "model.safetensors" for k in sorted(tensors)}}, f, indent=1)
    config = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "transformers_version": "5.17.0",
        "tie_word_embeddings": False,
        "text_config": TINY_TEXT,
        "vision_config": TINY_VISION,
        "quantization_config": {
            "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": sorted(set(keep_modules)) + [
                "attn_mha", "attn_mqa", "dt_bias", "hyper_connection", "mapping_proj", "router",
                "weights_proj", "visual", "model.visual"],
        },
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)


def read_set(root: str, name: str) -> dict[str, torch.Tensor]:
    """Load a golden set back from disk (manifest + raw little-endian files)."""
    d = os.path.join(root, name)
    with open(os.path.join(d, "manifest.json")) as f:
        man = json.load(f)
    dt = {"f32": torch.float32, "bf16": torch.bfloat16, "i32": torch.int32, "u8": torch.uint8}
    out = {}
    for k, e in man["tensors"].items():
        with open(os.path.join(d, e["file"]), "rb") as f:
            raw = bytearray(f.read())
        t = torch.frombuffer(raw, dtype=torch.int16 if e["dtype"] == "bf16" else dt[e["dtype"]])
        out[k] = (t.view(torch.bfloat16) if e["dtype"] == "bf16" else t).reshape(e["shape"])
    return out


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


class Failures:
    def __init__(self):
        self.items: list[str] = []
        self.checks = 0

    def check(self, ok: bool, what: str) -> None:
        self.checks += 1
        if not ok:
            self.items.append(what)
            print("FAIL", what)


def capture_layers(model, layer_ids):
    """Hooks recording each decoder layer's input and output streams per forward call."""
    rec = {i: [] for i in layer_ids}
    handles = []
    layers = model.model.language_model.layers
    for i in layer_ids:
        handles.append(layers[i].register_forward_pre_hook(
            lambda m, a, i=i: rec[i].append({"in": a[0].detach().clone()})))
        handles.append(layers[i].register_forward_hook(
            lambda m, a, o, i=i: rec[i][-1].__setitem__("out", o[0].detach().clone())))
    return rec, handles


def run_model(model, prompt, decode):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.config.get_text_config())
    logits = []
    with torch.no_grad():
        # logits_to_keep=1, as generate() does: one LM-head row per forward
        out = model(input_ids=torch.tensor([prompt]), past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits.append(out.logits[0])
        for t in decode:
            out = model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            logits.append(out.logits[0])
    return torch.cat(logits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=None, help="scratch directory (default: a new temporary one)")
    args = ap.parse_args()
    torch.set_num_threads(16)
    work = args.workdir or tempfile.mkdtemp(prefix="glm53f-tiny-")
    ckpt, out = os.path.join(work, "checkpoint"), os.path.join(work, "goldens")
    write_tiny_checkpoint(ckpt)
    g = torch.Generator().manual_seed(7)
    prompt = torch.randint(2, TINY_TEXT["vocab_size"], (13,), generator=g).tolist()
    decode = torch.randint(2, TINY_TEXT["vocab_size"], (4,), generator=g).tolist()
    print(f"tiny checkpoint in {ckpt}; prompt {len(prompt)} tokens, {len(decode)} decode steps")

    rc = gl.generate(SimpleNamespace(weights=[ckpt], out=out, threads=16, repo=None, revision=None, last_layer=4,
                                     prompt_ids=",".join(map(str, prompt)), decode_ids=",".join(map(str, decode))))
    if rc:
        return rc
    fails = Failures()
    fails.check(gl.verify(SimpleNamespace(dirs=[out])) == 0, "verify of the written sets")

    from transformers import Glm5NextForConditionalGeneration

    ck = gl.Checkpoint([ckpt])
    tcfg = gl.text_config(ck)
    rec_layers = list(gl.RECORD_LAYERS)
    kw = dict(attn_implementation="eager", experts_implementation="eager")

    # ---- FP32 -----------------------------------------------------------------------------------
    ref = Glm5NextForConditionalGeneration.from_pretrained(ckpt, dtype=torch.float32, **kw).float().eval()
    ref_layers = ref.model.language_model.layers
    for i in rec_layers:
        mine = gl.build_layer(tcfg, i, ck, "fp32")
        theirs = dict(ref_layers[i].named_parameters()) | dict(ref_layers[i].named_buffers())
        for n, t in list(mine.named_parameters()) + list(mine.named_buffers()):
            fails.check(same(t, theirs[n]), f"fp32 layer {i} parameter {n}")
        if isinstance(mine.mlp, gl.M.Glm5NextTextMoE):
            for e in range(TINY_TEXT["n_routed_experts"]):
                fails.check(same(mine.mlp.experts.gate_up_proj[e], theirs["mlp.experts.gate_up_proj"][e]),
                            f"fp32 layer {i} expert {e} gate_up")
                fails.check(same(mine.mlp.experts.down_proj[e], theirs["mlp.experts.down_proj"][e]),
                            f"fp32 layer {i} expert {e} down")
    rec, handles = capture_layers(ref, rec_layers)
    ref_logits = run_model(ref, prompt, decode)
    for h in handles:
        h.remove()
    for i in rec_layers:
        golden = read_set(out, f"layer{i:02d}-prefill") | read_set(out, f"layer{i:02d}-decode")
        fails.check(same(golden["prefill.in_streams"], rec[i][0]["in"][0]), f"fp32 layer {i} prefill input")
        fails.check(same(golden["prefill.out_streams"], rec[i][0]["out"][0]), f"fp32 layer {i} prefill output")
        steps = torch.cat([r["out"][0] for r in rec[i][1:]])
        fails.check(same(golden["decode.out_streams"], steps), f"fp32 layer {i} decode outputs")
    head = read_set(out, "head")
    fails.check(same(head["head.logits"], ref_logits), "fp32 logits (tiny model = the recorded chain)")
    del ref

    # ---- native ---------------------------------------------------------------------------------
    nat = Glm5NextForConditionalGeneration.from_pretrained(ckpt, **kw).eval()
    nat_layers = nat.model.language_model.layers
    for i in rec_layers:
        mine = gl.build_layer(tcfg, i, ck, "native")
        theirs = dict(nat_layers[i].named_parameters()) | dict(nat_layers[i].named_buffers())
        for n, t in list(mine.named_parameters()) + list(mine.named_buffers()):
            fails.check(same(t, theirs[n]), f"native layer {i} parameter {n} ({t.dtype} vs {theirs[n].dtype})")
        if isinstance(mine.mlp, gl.M.Glm5NextTextMoE):
            for e in range(TINY_TEXT["n_routed_experts"]):
                fails.check(same(mine.mlp.experts.gate_up_proj[e], theirs["mlp.experts.gate_up_proj"][e]),
                            f"native layer {i} expert {e} gate_up")
    rec, handles = capture_layers(nat, rec_layers)
    run_model(nat, prompt, decode)
    for h in handles:
        h.remove()
    phases = gl.phases_for(prompt, decode)
    for i in rec_layers:
        layer = gl.build_layer(tcfg, i, ck, "native")
        r = gl.Recorder()
        r.attach(layer, i)
        with torch.no_grad(), r.kda_capture():
            gl.run_layer(tcfg, layer, i, {p: rec[i][k]["in"][0] for k, (p, _) in enumerate(phases)}, r,
                            torch.bfloat16)
        r.detach()
        for k, (p, _) in enumerate(phases):
            fails.check(same(r.data[i][p]["out_streams"], rec[i][k]["out"][0]), f"native layer {i} {p} output")

    print(f"selftest: {fails.checks} checks, {len(fails.items)} failures")
    return 1 if fails.items else 0


if __name__ == "__main__":
    sys.exit(main())
