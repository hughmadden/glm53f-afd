#!/usr/bin/env python3
"""Per-layer golden fixtures for GLM-5.3-Flash, taken from the `transformers` glm5_next reference.

The reference modules (`Glm5NextTextDecoderLayer` and its parts) come unmodified from the pinned
`transformers` package. This script only builds them one layer at a time, loads their weights from
the official checkpoint, runs a short prompt and single-token decode steps on the CPU, and records
what the modules compute, using forward hooks and call wrappers.

Weights
    Tensors are read by name from one or more directories that each hold safetensors files and a
    `model.safetensors.index.json` weight map: the official checkpoint, or subsets of it with the
    original tensor names. FP8 E4M3 weights are dequantised with transformers' own
    `Fp8Dequantize._dequantize_one`, the function `from_pretrained` uses when it dequantises this
    checkpoint: fp32(w) * weight_scale_inv per 128 x 128 block, rounded once to the target dtype.
    Checkpoint names are mapped to module names as transformers' `conversion_mapping` does for
    glm5_next. Routed experts are dequantised one expert at a time, when the reference's expert
    loop first touches them.

Numerics
    fp32    Every parameter is cast to FP32 after dequantisation, and the modules run in FP32. This
            is the primary contract.
    native  The dtypes `from_pretrained` gives this checkpoint on a CPU: BF16 everywhere, except
            tensors stored as F32 in the checkpoint (A_log, dt_bias, the mHC base and scale, the
            router's correction bias). Each recorded layer runs on its FP32 golden input cast to
            BF16, so this shows the reference's own rounding per layer.

Usage
    golden_layers.py generate --weights DIR [--weights DIR ...] --out DIR
    golden_layers.py verify DIR [DIR ...]
    golden_layers.py compare DIR_A DIR_B
"""

from __future__ import annotations

import os

# Pin the CPU code paths before torch (and MKL/oneDNN) load: AVX2 kernels for ATen, MKL and oneDNN,
# and MKL's strict conditional-numerical-reproducibility mode, so the results do not depend on whether
# the CPU has AVX-512. The thread count is part of the contract too (see --threads).
NUMERICS_ENV = {"ATEN_CPU_CAPABILITY": "avx2", "MKL_CBWR": "AVX2,STRICT", "ONEDNN_MAX_CPU_ISA": "AVX2"}
os.environ.update(NUMERICS_ENV)

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402

import transformers  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.integrations.finegrained_fp8 import Fp8Dequantize  # noqa: E402
from transformers.models.glm5_next import modeling_glm5_next as M  # noqa: E402
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig  # noqa: E402

FORMAT_VERSION = 1
TRANSFORMERS_VERSION = "5.17.0"
TRANSFORMERS_WHEEL_SHA256 = "78ec1ce21579b38dfb83950a0658cd119f87212a2fcfdff478096ce9d6c03801"

# The prompt: one user turn, rendered with the checkpoint's chat template and tokenizer.
PROMPT_MESSAGES = [{
    "role": "user",
    "content": "Briefly, in two sentences: why does the sky look blue during the day but red at sunset?",
}]
# Fixed decode tokens, fed one at a time after the prompt. They are not sampled: the layers recorded
# here are not a whole model, so they cannot choose a next token.
DECODE_TEXT = "Sunlight scatters off air molecules, and blue light scatters more"
DECODE_STEPS = 8

RECORD_LAYERS = (0, 3, 4)       # KDA + dense MLP, DSA + MoE, KDA + MoE; layers 0..4 run as a chain
VARIANT_TOPK = 16               # extra DSA run with index_topk 16 (4 pools + tail): exercises pool dropping

# Checkpoint name <-> module name, as transformers' conversion_mapping["glm5_next"] (5.17.0) renames them.
RENAMES = (
    ("self_attn.forget_gate.f_a_proj.", "self_attn.f_a_proj."),
    ("self_attn.forget_gate.f_b_proj.", "self_attn.f_b_proj."),
    ("self_attn.forget_gate.dt_bias", "self_attn.dt_bias"),
    ("self_attn.forget_gate.A_log", "self_attn.A_log"),
    ("attn_hc.fn", "hc_attn_fn"),
    ("attn_hc.base", "hc_attn_base"),
    ("attn_hc.scale", "hc_attn_scale"),
    ("ffn_hc.fn", "hc_ffn_fn"),
    ("ffn_hc.base", "hc_ffn_base"),
    ("ffn_hc.scale", "hc_ffn_scale"),
)
TEXT_PREFIX = "model.language_model."
# Small KDA tensors stored in each KDA layer's sets, as the checkpoint stores them, so a layer check can
# run from the fixtures (the large projections are read from the checkpoint).
KDA_SMALL_WEIGHTS = ("self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight",
                     "self_attn.A_log", "self_attn.dt_bias", "self_attn.o_norm.weight")
# The native KDA tensors recorded per layer: the kernel's inputs (projections), the conv and gate
# outputs, and the core output.
NATIVE_KDA_KEYS = ("kda.qkv_preconv", "kda.f_proj", "kda.b_logits", "kda.gate", "kda.q", "kda.k", "kda.v",
                   "kda.g", "kda.beta", "kda.core_out")
CONV_ROUNDING_NOTE = ("the reference's short conv is the unfused PyTorch fallback: natively (BF16 weights and "
                      "inputs) F.conv1d rounds its output to BF16, then SiLU rounds again. A fused conv + SiLU "
                      "that rounds once differs from it by about 1 BF16 ulp on a fifth of the outputs "
                      "(native notes.diff_vs_fp32: LNN.conv_fused_vs_reference). The FP32 goldens run the same "
                      "code in FP32")
_DEQUANT = Fp8Dequantize(hf_quantizer=None)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    elif t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = t.view(torch.uint8)
    elif t.dtype == torch.bool:
        t = t.to(torch.uint8)
    return t.numpy().tobytes()


# --------------------------------------------------------------------------------------------------
# Checkpoint access


class Checkpoint:
    """Tensors by original name across one or more safetensors directories (first directory wins)."""

    def __init__(self, dirs: list[str]):
        self.where: dict[str, str] = {}
        self.meta: dict[str, str] = {}
        for d in dirs:
            with open(os.path.join(d, "model.safetensors.index.json")) as f:
                index = json.load(f)
            for name, fn in index["weight_map"].items():
                self.where.setdefault(name, os.path.join(d, fn))
            for k in ("source_repo", "source_revision"):
                if k in index.get("metadata", {}):
                    self.meta.setdefault(k, index["metadata"][k])
        self.config_dir = next((d for d in dirs if os.path.exists(os.path.join(d, "config.json"))), None)
        self._files: dict[str, object] = {}
        self.read: dict[str, tuple[str, list[int], str]] = {}  # name -> (dtype, shape, sha256 of raw bytes)

    def get(self, name: str) -> torch.Tensor:
        path = self.where[name]
        if path not in self._files:
            self._files[path] = safe_open(path, framework="pt")
        t = self._files[path].get_tensor(name)
        if name not in self.read:
            self.read[name] = (str(t.dtype).replace("torch.", ""), list(t.shape), sha256_bytes(tensor_bytes(t)))
        return t

    def digest(self, names) -> str:
        lines = [f"{n}\t{self.read[n][0]}\t{self.read[n][1]}\t{self.read[n][2]}" for n in sorted(names)]
        return sha256_bytes("\n".join(lines).encode())


def target_dtype(mode: str, stored: torch.dtype) -> torch.dtype:
    """fp32: everything FP32. native: what from_pretrained gives this checkpoint (see module docstring)."""
    if mode == "fp32":
        return torch.float32
    return torch.float32 if stored == torch.float32 else torch.bfloat16


def load_weight(ck: Checkpoint, name: str, mode: str) -> torch.Tensor:
    t = ck.get(name)
    if t.dtype == torch.float8_e4m3fn:
        scale = ck.get(name + "_scale_inv")  # "...weight" -> "...weight_scale_inv"
        return _DEQUANT._dequantize_one(t, scale, output_dtype=target_dtype(mode, torch.bfloat16))
    return t.to(target_dtype(mode, t.dtype))


def checkpoint_name(layer_idx: int, param: str) -> str:
    for module_side, ckpt_side in RENAMES:
        if param.startswith(module_side):
            param = ckpt_side + param[len(module_side):]
            break
    return f"{TEXT_PREFIX}layers.{layer_idx}.{param}"


class ExpertBank:
    """The reference's stacked expert parameter ([E, out, in]), materialised one expert at a time.

    `Glm5NextTextExperts.forward` only indexes `gate_up_proj[e]` and `down_proj[e]` for the experts
    the router picked. Each access dequantises that expert from the checkpoint, laid out as
    transformers' conversion does: gate_up = cat(gate_proj, up_proj) along the output dimension.
    """

    def __init__(self, ck: Checkpoint, prefix: str, parts: tuple[str, ...], mode: str):
        self.ck, self.prefix, self.parts, self.mode = ck, prefix, parts, mode
        self.touched: set[int] = set()

    def __getitem__(self, idx) -> torch.Tensor:
        e = int(idx)
        self.touched.add(e)
        mats = [load_weight(self.ck, f"{self.prefix}{e}.{p}.weight", self.mode) for p in self.parts]
        return mats[0] if len(mats) == 1 else torch.cat(mats, dim=0)


def _set_tensor(root: nn.Module, dotted: str, value: torch.Tensor) -> None:
    mod_path, _, leaf = dotted.rpartition(".")
    mod = root.get_submodule(mod_path) if mod_path else root
    old = mod._parameters.get(leaf) if leaf in mod._parameters else mod._buffers.get(leaf)
    if tuple(old.shape) != tuple(value.shape):
        raise ValueError(f"{dotted}: checkpoint shape {tuple(value.shape)} != module shape {tuple(old.shape)}")
    if leaf in mod._parameters:
        mod._parameters[leaf] = nn.Parameter(value.contiguous(), requires_grad=False)
    else:
        mod._buffers[leaf] = value.contiguous()


def build_layer(tcfg, layer_idx: int, ck: Checkpoint, mode: str) -> nn.Module:
    """A reference decoder layer with checkpoint weights (every tensor loaded, nothing left on meta)."""
    with torch.device("meta"):
        layer = M.Glm5NextTextDecoderLayer(tcfg, layer_idx)
    names = [n for n, _ in layer.named_parameters()] + [n for n, _ in layer.named_buffers()]
    for name in names:
        if name.startswith("mlp.experts."):
            continue
        if name == "self_attn.conv1d.weight":
            parts = [ck.get(checkpoint_name(layer_idx, f"self_attn.{p}_conv1d.weight")) for p in "qkv"]
            value = torch.cat(parts, dim=0)  # transformers: Concatenate(dim=0) of q, k, v
            value = value.to(target_dtype(mode, value.dtype))
        else:
            value = load_weight(ck, checkpoint_name(layer_idx, name), mode)
        _set_tensor(layer, name, value)
    if isinstance(layer.mlp, M.Glm5NextTextMoE):
        experts = layer.mlp.experts
        prefix = checkpoint_name(layer_idx, "mlp.experts.")
        for attr, parts in (("gate_up_proj", ("gate_proj", "up_proj")), ("down_proj", ("down_proj",))):
            del experts._parameters[attr]
            setattr(experts, attr, ExpertBank(ck, prefix, parts, mode))
    left = [n for n, t in list(layer.named_parameters()) + list(layer.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"layer {layer_idx}: tensors not loaded: {left}")
    return layer.eval()


def text_config(ck: Checkpoint):
    cfg = Glm5NextConfig.from_pretrained(ck.config_dir)
    tcfg = cfg.text_config
    tcfg._attn_implementation = "eager"      # the modeling file's own attention code
    tcfg._experts_implementation = "eager"   # the modeling file's own expert loop
    return tcfg


# --------------------------------------------------------------------------------------------------
# Recording


class _OpCapture(TorchFunctionMode):
    """Records selected torch calls made inside one module's forward (tensors the module never returns)."""

    def __init__(self, want):
        super().__init__()
        self.want, self.calls = want, []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        tag = self.want(func, args)
        if tag is not None:
            self.calls.append((tag, args, out))
        return out


def _squeeze_batch(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 0 or t.shape[0] != 1:
        raise ValueError(f"expected batch size 1, got shape {tuple(t.shape)}")
    return t[0]


class Recorder:
    """Per-layer, per-phase record of reference intermediates (batch dimension removed)."""

    def __init__(self):
        self.data: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
        self.active: set[int] = set()  # layers being recorded; calls from other layers are ignored
        self.layer: int | None = None
        self.phase: str | None = None
        self._handles: list = []
        self._restore: list = []

    def put(self, key: str, value: torch.Tensor, layer: int | None = None) -> None:
        layer = self.layer if layer is None else layer
        if layer not in self.active:
            return
        slot = self.data.setdefault(layer, {}).setdefault(self.phase, {})
        if key in slot:
            raise RuntimeError(f"layer {layer} phase {self.phase}: {key} recorded twice")
        slot[key] = value.detach().clone()

    def _hook_out(self, mod, key, fn=lambda o: o, batched=True):
        sq = _squeeze_batch if batched else (lambda t: t)
        self._handles.append(mod.register_forward_hook(lambda m, a, o: self.put(key, sq(fn(o)))))

    def _hook_in(self, mod, key):
        self._handles.append(mod.register_forward_pre_hook(lambda m, a: self.put(key, _squeeze_batch(a[0]))))

    def _wrap_forward(self, mod, want, on_calls):
        orig = mod.forward

        def forward(*args, **kwargs):
            cap = _OpCapture(want)
            with cap:
                out = orig(*args, **kwargs)
            on_calls(cap.calls)
            return out

        mod.forward = forward
        self._restore.append(lambda: delattr(mod, "forward"))

    def attach(self, layer: nn.Module, layer_idx: int) -> None:
        self.active.add(layer_idx)

        def enter(m, args):
            self.layer = layer_idx
            self.put("in_streams", _squeeze_batch(args[0]))

        self._handles.append(layer.register_forward_pre_hook(enter))
        self._hook_out(layer, "out_streams", lambda o: o[0])
        for site in ("attn_hc", "ffn_hc"):
            hc = getattr(layer, site)
            if site == "ffn_hc":
                self._hook_in(hc, "mid_streams")
            self._hook_out(hc, f"{site}.post", lambda o: o[0])
            self._hook_out(hc, f"{site}.comb", lambda o: o[1])
            self._hook_out(hc, f"{site}.collapsed", lambda o: o[2])

            def on_calls(calls, site=site):
                pres = [a[0] for tag, a, _ in calls if tag == "pre"]
                if len(pres) != 1:
                    raise RuntimeError(f"{site}: expected one pre.unsqueeze call, saw {len(pres)}")
                self.put(f"{site}.pre", _squeeze_batch(pres[0]))

            self._wrap_forward(hc, lambda f, a: "pre" if f is torch.Tensor.unsqueeze else None, on_calls)
        self._hook_out(layer.input_layernorm, "attn_norm")
        self._hook_out(layer.post_attention_layernorm, "ffn_norm")
        attn = layer.self_attn
        if isinstance(attn, M.Glm5NextTextLinearAttention):
            self._hook_out(attn, "attn_out")
            for n in ("q_proj", "k_proj", "v_proj"):
                self._hook_out(getattr(attn, n), f"kda.{n}")
            self._hook_out(attn.forget_gate.f_b_proj, "kda.f_proj", lambda o: o.view(*o.shape[:-1], -1, attn.head_dim))
            self._hook_out(attn.b_proj, "kda.b_logits")
            self._hook_out(attn.g_b_proj, "kda.gate", lambda o: o.view(*o.shape[:-1], -1, attn.head_dim))
            self._hook_out(attn.o_norm, "kda.norm_out")
        else:
            self._hook_out(attn, "attn_out", lambda o: o[0])
            self._hook_out(attn, "mla.probs", lambda o: o[1])
            self._hook_out(attn.q_a_layernorm, "mla.q_resid")
            self._hook_out(attn.q_b_proj, "mla.q", lambda o: o.view(*o.shape[:-1], attn.num_heads, -1))
            self._hook_out(attn.kv_a_proj_with_mqa, "mla.kv_a")
            self._hook_out(attn.kv_a_layernorm, "mla.latent")
            self._handles.append(attn.o_proj.register_forward_pre_hook(
                lambda m, a: self.put("mla.out", _squeeze_batch(a[0]).view(a[0].shape[1], attn.num_heads, -1))))
            self.attach_indexer(attn.indexer)
        mlp = layer.mlp
        self._hook_out(mlp, "mlp_out")
        if isinstance(mlp, M.Glm5NextTextMoE):
            self._handles.append(mlp.gate.register_forward_hook(self._router_hook))
            self._hook_out(mlp.experts, "moe.routed_out", batched=False)  # the MoE flattens tokens first
            self._hook_out(mlp.shared_experts, "moe.shared_out")

    def attach_indexer(self, idx: nn.Module, prefix: str = "idx", topk_only: bool = False) -> None:
        self._hook_out(idx, f"{prefix}.topk")
        if topk_only:
            return
        self._hook_out(idx.wq_b, f"{prefix}.q", lambda o: o.view(*o.shape[:-1], idx.n_heads, idx.head_dim))
        self._hook_out(idx.k_norm, f"{prefix}.k")
        self._hook_out(idx.weights_proj, f"{prefix}.weights")
        gate = idx.index_kpool_compress_gate

        def want(func, args):
            if func is torch.Tensor.topk:
                return "topk"
            if func is F.linear and len(args) > 1 and args[1] is gate:
                return "gate"
            return None

        def on_calls(calls):
            scores = [a[0] for tag, a, _ in calls if tag == "topk"]
            gates = [o for tag, _, o in calls if tag == "gate"]
            if len(scores) != 1 or len(gates) != 1:
                raise RuntimeError(f"indexer: expected one topk and one gate projection, saw {len(scores)}, {len(gates)}")
            self.put(f"{prefix}.scores", _squeeze_batch(scores[0]))
            self.put(f"{prefix}.gate_scores", _squeeze_batch(gates[0]))

        self._wrap_forward(idx, want, on_calls)
        orig_pool = idx.get_pooled_states

        def get_pooled_states(*args, **kwargs):
            keys, indices, valid = orig_pool(*args, **kwargs)
            self.put(f"{prefix}.pool_keys", _squeeze_batch(keys))
            self.put(f"{prefix}.pool_indices", _squeeze_batch(indices).to(torch.int32))
            return keys, indices, valid

        idx.get_pooled_states = get_pooled_states
        self._restore.append(lambda: delattr(idx, "get_pooled_states"))

    def _router_hook(self, mod, args, out):
        logits, weights, ids = out
        self.put("moe.router_logits", logits)  # the router flattens to [tokens, experts]
        self.put("moe.topk_ids", ids.to(torch.int32))
        self.put("moe.topk_weights", weights)
        order = ids.argsort(dim=-1)
        self.put("moe.topk_ids_sorted", ids.gather(-1, order).to(torch.int32))
        self.put("moe.topk_weights_sorted", weights.gather(-1, order))

    @contextmanager
    def kda_capture(self):
        """Wrap the module-level KDA functions the reference layer calls, to see their inputs and outputs."""
        originals = {n: getattr(M, n) for n in ("chunk_kimi_delta_attention", "recurrent_kimi_delta_attention")}

        def wrap(name, fn):
            def run(query, key, value, *args, **kwargs):
                out, state = fn(query, key, value, *args, **kwargs)
                self.put("kda.path", torch.tensor([0 if name.startswith("chunk") else 1], dtype=torch.int32))
                self.put("kda.q", _squeeze_batch(query))
                self.put("kda.k", _squeeze_batch(key))
                self.put("kda.v", _squeeze_batch(value))
                self.put("kda.g", _squeeze_batch(kwargs["g"]))
                self.put("kda.beta", _squeeze_batch(kwargs["beta"]))
                self.put("kda.core_out", _squeeze_batch(out))
                self.put("kda.state", _squeeze_batch(state))
                return out, state
            return run

        for n, fn in originals.items():
            setattr(M, n, wrap(n, fn))
        try:
            yield
        finally:
            for n, fn in originals.items():
                setattr(M, n, fn)

    def detach(self):
        for h in self._handles:
            h.remove()
        for undo in self._restore:
            undo()
        self._handles, self._restore = [], []


# --------------------------------------------------------------------------------------------------
# Running


def phases_for(prompt_ids: list[int], decode_ids: list[int]) -> list[tuple[str, list[int]]]:
    return [("prefill", list(prompt_ids))] + [(f"s{i}", [t]) for i, t in enumerate(decode_ids)]


def run_layer(tcfg, layer: nn.Module, layer_idx: int, inputs: dict[str, torch.Tensor], rec: Recorder | None,
              dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """One decoder layer over every phase in order (prompt, then one token per step), with its own
    reference cache. Returns each phase's output streams.

    The chain runs layer-major: a layer sees all phases before the next layer loads. Because the decode
    tokens are fixed, every layer's per-phase inputs and cache are exactly what a phase-major run gives
    it, so the results are bit-identical (selftest_tiny.py compares against the whole-model forward).
    """
    cache = DynamicCache(config=tcfg)
    outs = {}
    seen = 0
    for phase, streams in inputs.items():
        if rec is not None:
            rec.phase, rec.layer = phase, layer_idx
        h = streams[None].to(dtype)
        n = h.shape[1]
        mask = torch.ones(1, n, dtype=torch.bool)  # as Glm5NextTextModel.forward without padding
        pos = (torch.arange(n) + seen)[None]       # NoPE: unused by the layers
        out, _ = layer(h, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True,
                       position_embeddings=None, prev_topk_indices=None)
        if rec is not None and isinstance(layer.self_attn, M.Glm5NextTextLinearAttention):
            rec.put("kda.conv_state", _squeeze_batch(cache.layers[layer_idx].conv_states[0]), layer=layer_idx)
        outs[phase] = out[0]
        seen += n
    return outs


def run_attention_variant(tcfg, attn: nn.Module, layer_idx: int, topk: int, norm_inputs, rec: Recorder):
    """The DSA attention of `layer_idx` with a smaller index_topk, sharing the layer's weights."""
    cfg = copy.deepcopy(tcfg)
    cfg.index_topk = topk
    with torch.device("meta"):
        var = M.Glm5NextTextAttention(cfg, layer_idx)
    for name, t in list(attn.named_parameters()) + list(attn.named_buffers()):
        _set_tensor(var, name, t)
    var.eval()
    rec.attach_indexer(var.indexer, prefix=f"k{topk}.idx", topk_only=True)
    rec._hook_out(var, f"k{topk}.attn_out", lambda o: o[0])
    rec._handles.append(var.o_proj.register_forward_pre_hook(
        lambda m, a: rec.put(f"k{topk}.mla.out", _squeeze_batch(a[0]).view(a[0].shape[1], var.num_heads, -1))))
    cache = DynamicCache(config=cfg)
    for phase, x in norm_inputs.items():
        rec.phase, rec.layer = phase, layer_idx
        n = x.shape[0]
        var(x[None], attention_mask=torch.ones(1, n, dtype=torch.bool), past_key_values=cache,
            prev_topk_indices=None)


class Head:
    """Final HyperHead mean, final RMSNorm and LM head, with checkpoint weights."""

    def __init__(self, tcfg, ck: Checkpoint, mode: str):
        self.hc_head = M.Glm5NextTextHyperHead()
        with torch.device("meta"):
            self.norm = M.Glm5NextTextRMSNorm(tcfg.hidden_size, eps=tcfg.rms_norm_eps)
        _set_tensor(self.norm, "weight", load_weight(ck, TEXT_PREFIX + "norm.weight", mode))
        self.lm_head = load_weight(ck, "lm_head.weight", mode)

    def __call__(self, streams: torch.Tensor):
        """streams [rows, hc, hidden]. One row per call, as generation computes logits (logits_to_keep=1):
        the CPU GEMM's rounding depends on the number of rows, so batching rows would change the bits."""
        outs = []
        for r in range(streams.shape[0]):
            collapsed = self.hc_head(streams[None, r:r + 1])
            normed = self.norm(collapsed)
            outs.append((collapsed[0], normed[0], F.linear(normed, self.lm_head)[0]))
        return tuple(torch.cat(x) for x in zip(*outs))


def make_embed(ck: Checkpoint, mode: str):
    table = ck.get(TEXT_PREFIX + "embed_tokens.weight")

    def embed(ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, table).to(target_dtype(mode, table.dtype))  # rows of the stored table

    return embed


# --------------------------------------------------------------------------------------------------
# Output


DTYPE_NAMES = {torch.float32: "f32", torch.bfloat16: "bf16", torch.int32: "i32", torch.uint8: "u8"}


class SetWriter:
    def __init__(self, root: str, name: str, source: dict, notes: dict):
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.name, self.source, self.notes = name, source, notes
        self.tensors: dict[str, dict] = {}
        self.bytes = 0

    def add(self, key: str, t: torch.Tensor, desc: str) -> None:
        if key in self.tensors:
            raise KeyError(f"{self.name}: duplicate tensor {key}")
        t = t.detach().contiguous().cpu()
        if t.dtype == torch.int64:
            if t.abs().max() >= 2**31:
                raise ValueError(f"{key}: int64 values do not fit int32")
            t = t.to(torch.int32)
        if t.dtype not in DTYPE_NAMES:
            raise TypeError(f"{key}: unsupported dtype {t.dtype}")
        raw = tensor_bytes(t)
        fname = key + ".bin"
        with open(os.path.join(self.dir, fname), "wb") as f:
            f.write(raw)
        self.tensors[key] = {"file": fname, "dtype": DTYPE_NAMES[t.dtype], "shape": list(t.shape),
                             "sha256": sha256_bytes(raw), "desc": desc}
        self.bytes += len(raw)

    def close(self) -> None:
        manifest = {"format": FORMAT_VERSION, "set": self.name, "source": self.source, "notes": self.notes,
                    "tensors": self.tensors}
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
            f.write("\n")
        with open(os.path.join(self.dir, "tensors.tsv"), "w") as f:
            f.write("# name\tfile\tdtype\tshape\tsha256\n")
            for k, v in self.tensors.items():
                f.write(f"{k}\t{v['file']}\t{v['dtype']}\t{','.join(map(str, v['shape']))}\t{v['sha256']}\n")
        mb = self.bytes / 1e6
        print(f"  {self.name}: {len(self.tensors)} tensors, {mb:.1f} MB{'  (over 30 MB)' if mb > 30 else ''}")


# What each recorded key means (prefix "prefill." / "decode." is added when writing).
DESCRIPTIONS = {
    "in_streams": "decoder layer input: the 4 mHC residual streams [tokens, hc, hidden]",
    "out_streams": "decoder layer output: the 4 mHC residual streams",
    "mid_streams": "streams after the attention site's mHC update (input of ffn_hc)",
    "attn_hc.pre": "attention-site mHC pre weights (sigmoid + eps) that collapse the streams",
    "attn_hc.post": "attention-site mHC post weights (2*sigmoid) that expand the block output",
    "attn_hc.comb": "attention-site mHC comb matrix after Sinkhorn; new_stream[i] = sum_j comb[j,i]*stream[j] + post[i]*block_out",
    "attn_hc.collapsed": "attention-site collapsed input sum_i pre[i]*stream[i] (before input_layernorm)",
    "ffn_hc.pre": "FFN-site mHC pre weights",
    "ffn_hc.post": "FFN-site mHC post weights",
    "ffn_hc.comb": "FFN-site mHC comb matrix after Sinkhorn",
    "ffn_hc.collapsed": "FFN-site collapsed input (before post_attention_layernorm)",
    "attn_norm": "input_layernorm output (attention block input)",
    "ffn_norm": "post_attention_layernorm output (MLP/MoE input; the routed rows sent to the experts)",
    "attn_out": "attention block output after o_proj",
    "mlp_out": "MLP or MoE block output (routed + shared for MoE)",
    "kda.qkv_preconv": "KDA q|k|v projections before the short convolution, concatenated [tokens, 3*heads*dim]",
    "kda.q": "KDA query after short conv + SiLU, before the L2 norm [tokens, heads, dim]",
    "kda.k": "KDA key after short conv + SiLU, before the L2 norm",
    "kda.v": "KDA value after short conv + SiLU",
    "kda.g": "KDA forget gate (log decay per channel): -5*sigmoid(exp(A_log)*(f_b(f_a(x)) + dt_bias))",
    "kda.beta": "KDA beta = sigmoid(b_proj(x)) [tokens, heads]",
    "kda.core_out": "KDA recurrence output before the gated RMSNorm [tokens, heads, dim]",
    "kda.f_proj": "forget-gate projection f_b_proj(f_a_proj(x)), before dt_bias [tokens, heads, dim]",
    "kda.b_logits": "b_proj(x), before the sigmoid that gives beta [tokens, heads]",
    "kda.gate": "output gate g_b_proj(g_a_proj(x)) for the gated RMSNorm, before sigmoid [tokens, heads, dim]",
    "kda.norm_out": "gated RMSNorm output o_norm(core_out, gate): the o_proj input [tokens, heads, dim]",
    "kda.state": "KDA recurrent state after the phase, FP32 [heads, k_dim, v_dim] (S[k, v]; out = q^T S)",
    "kda.conv_state": "short-conv cache after the phase: last 4 pre-conv inputs [channels q|k|v, 4], oldest first; column 0 never reaches an output",
    "kda.path": "reference KDA path: 0 = chunked (64-token chunks), 1 = recurrent single step",
    "mla.q_resid": "q_a_layernorm(q_a_proj(x)): MLA query latent, also the indexer's query input",
    "mla.q": "MLA query q_b_proj(q_resid) [tokens, heads, 256] (NoPE)",
    "mla.kv_a": "kv_a_proj_with_mqa(x), before kv_a_layernorm [tokens, 512]",
    "mla.latent": "kv_a_layernorm output: the 512-dim MLA latent that the KV cache holds per token",
    "mla.probs": "sparse attention probabilities (eager softmax) [heads, queries, keys]",
    "mla.out": "sparse MLA output per head, before o_proj [tokens, heads, 256]",
    "idx.q": "indexer query wq_b(q_resid) [tokens, index_heads, 128]",
    "idx.k": "indexer key k_norm(wk(x)) (LayerNorm) [tokens, 128]",
    "idx.gate_scores": "k-pool compression gate x @ index_kpool_compress_gate^T [tokens, 128]",
    "idx.weights": "indexer head weights weights_proj(x), before the n_heads^-0.5 scale [tokens, index_heads]",
    "idx.pool_keys": "pooled keys of the complete pools (softmax(gate_scores + ape) over each pool's 4 tokens) [pools, 128]",
    "idx.pool_indices": "token indices of each complete pool [pools, 4]",
    "idx.scores": "index scores per query and complete pool, invisible pools set to float32 min [queries, pools]",
    "idx.topk": "selected raw token indices: selected pools' tokens in score order, then the visible tail (<=3), then -1 padding",
    "moe.router_logits": "router logits x @ gate^T in FP32 [tokens, experts]",
    "moe.topk_ids": "router top-8 expert ids as the reference returns them (torch.topk sorted=False)",
    "moe.topk_weights": "router weights for topk_ids: sigmoid scores normalised over the 8, times 2.5",
    "moe.topk_ids_sorted": "router top-8 expert ids sorted ascending",
    "moe.topk_weights_sorted": "router weights in the order of topk_ids_sorted",
    "moe.routed_out": "sum over the 8 routed experts of weight * expert(x) (reference accumulation order: by expert id)",
    "moe.shared_out": "shared expert output",
}


def describe(key: str) -> str:
    base = re.sub(r"^(prefill|decode)\.(s\d+\.)?", "", key)
    variant = re.match(r"k(\d+)\.(.*)", base)
    if variant:
        return f"index_topk={variant.group(1)} variant: " + DESCRIPTIONS.get(variant.group(2), variant.group(2))
    return DESCRIPTIONS.get(base, base)


# Per-phase tensors without a leading token axis.
NOT_TOKEN_MAJOR = {"kda.state", "kda.conv_state", "idx.pool_keys", "idx.pool_indices", "mla.probs"}


def merge_qkv_projections(per_phase: dict[str, dict[str, torch.Tensor]]) -> None:
    """q_proj | k_proj | v_proj outputs as the one tensor the short conv consumes (its channel order)."""
    for d in per_phase.values():
        if "kda.q_proj" in d:
            d["kda.qkv_preconv"] = torch.cat([d.pop("kda.q_proj"), d.pop("kda.k_proj"), d.pop("kda.v_proj")], -1)


def write_prefill(w: SetWriter, per_phase: dict[str, dict[str, torch.Tensor]]) -> None:
    for k, v in per_phase["prefill"].items():
        w.add(f"prefill.{k}", v, describe(k))


def write_decode(w: SetWriter, per_phase: dict[str, dict[str, torch.Tensor]], state_heads: list[int],
                 skip_decode=()) -> None:
    steps = sorted((p for p in per_phase if p != "prefill"), key=lambda s: int(s[1:]))
    keys = list(per_phase[steps[0]])
    for k in keys:
        if k in skip_decode:
            continue
        vals = [per_phase[s][k] for s in steps]
        if k == "kda.state":
            stacked = torch.stack(vals)
            w.add("decode.kda.state_heads", stacked[:, state_heads],
                  f"KDA state after each decode step, heads {state_heads} only [steps, {len(state_heads)}, k_dim, v_dim]")
            w.add("decode.kda.state_final", vals[-1], "KDA state after the last decode step, all heads")
            continue
        if k == "kda.conv_state":
            w.add("decode.kda.conv_state_final", vals[-1], "short-conv cache after the last decode step")
            continue
        base = re.sub(r"^k\d+\.", "", k)
        if base not in NOT_TOKEN_MAJOR:
            if any(v.shape[0] != 1 for v in vals):
                raise RuntimeError(f"decode {k}: expected one token per step")
            vals = [v[0] for v in vals]  # one token per step: drop the token axis
        if all(v.shape == vals[0].shape for v in vals):
            w.add(f"decode.{k}", torch.stack(vals), describe(k) + " [steps, ...]")
        else:
            for s, v in zip(steps, vals):
                w.add(f"decode.{s}.{k}", v, describe(k))


def tokenize_prompt(ck: Checkpoint):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ck.config_dir)
    text = tok.apply_chat_template(PROMPT_MESSAGES, add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    ids_direct = tok.apply_chat_template(PROMPT_MESSAGES, add_generation_prompt=True, tokenize=True,
                                         return_dict=False)
    if isinstance(ids_direct, dict) or hasattr(ids_direct, "keys"):
        ids_direct = ids_direct["input_ids"]
    if list(ids_direct) != list(ids):
        raise RuntimeError("chat template: tokenize=True and tokenize-after-render disagree")
    decode = tok(DECODE_TEXT, add_special_tokens=False)["input_ids"][:DECODE_STEPS]
    files = {n: sha256_file(os.path.join(ck.config_dir, n)) for n in ("tokenizer.json", "chat_template.jinja")}
    return text, list(ids), list(decode), files


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_info(threads: int) -> dict:
    import numpy

    mkl = re.search(r"oneAPI Math Kernel Library Version ([^\n]+)", torch.__config__.show())
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "mkl": mkl.group(1).strip() if mkl else None,
        "device": "cpu",
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "threads": threads,
        "env": dict(NUMERICS_ENV),
        "deterministic_algorithms": True,
    }


def reference_info() -> dict:
    return {
        "package": "transformers",
        "version": transformers.__version__,
        "wheel_sha256": TRANSFORMERS_WHEEL_SHA256,
        "modeling_file": "transformers/models/glm5_next/modeling_glm5_next.py",
        "modeling_file_sha256": sha256_file(M.__file__),
        "attn_implementation": "eager",
        "experts_implementation": "eager",
        "kernels": "pure-PyTorch reference paths (no kernels hub, fla or causal-conv1d installed)",
    }


def bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in BF16 ulps (bit patterns mapped to a monotonic integer order)."""
    def order(t):
        bits = t.contiguous().view(torch.int16).int()
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)
    return (order(a) - order(b)).abs()


def fused_conv_vs_reference(qkv_preconv: torch.Tensor, conv_weight: torch.Tensor, qkv_ref: torch.Tensor) -> dict:
    """A fused short conv + SiLU that rounds once (FP32 inside) against the reference's native output.

    qkv_preconv [rows, C] and qkv_ref [rows, C] are consecutive tokens (prompt then decode steps), so a
    causal conv over the rows sees what the reference's conv cache gives each decode step."""
    x = qkv_preconv.float().t()[None]
    out = F.conv1d(x, conv_weight.float(), padding=conv_weight.shape[-1] - 1, groups=conv_weight.shape[0])
    fused = F.silu(out[..., :qkv_preconv.shape[0]])[0].t().to(qkv_ref.dtype)
    ulps = bf16_ulps(fused, qkv_ref)
    return {"outputs_differing": float(f"{(ulps > 0).float().mean().item():.4g}"),
            "outputs_1_ulp": float(f"{(ulps == 1).float().mean().item():.4g}"),
            "max_ulps": int(ulps.max().item())}


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> dict:
    d = (a.float() - ref.float())
    rms_ref = ref.float().pow(2).mean().sqrt().item()
    return {"max_abs": float(f"{d.abs().max().item():.6g}"),
            "rel_rms": float(f"{(d.pow(2).mean().sqrt().item() / rms_ref if rms_ref else 0.0):.6g}")}


# --------------------------------------------------------------------------------------------------
# generate


def generate(args) -> int:
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if sys.byteorder != "little":
        raise SystemExit("fixtures are little-endian; run on a little-endian host")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(f"need transformers {TRANSFORMERS_VERSION}, found {transformers.__version__}")
    for mod in ("kernels", "fla", "causal_conv1d"):
        if __import__("importlib.util").util.find_spec(mod):
            raise SystemExit(f"{mod} is installed; the oracle must use the pure-PyTorch reference paths")

    ck = Checkpoint(args.weights)
    if ck.config_dir is None:
        raise SystemExit("no config.json in the --weights directories")
    tcfg = text_config(ck)
    if args.last_layer < max(RECORD_LAYERS) or args.last_layer >= tcfg.num_hidden_layers:
        raise SystemExit(f"--last-layer must be in [{max(RECORD_LAYERS)}, {tcfg.num_hidden_layers - 1}]")
    chain = tuple(range(args.last_layer + 1))
    if args.prompt_ids:
        prompt_ids = [int(x) for x in args.prompt_ids.split(",")]
        decode_ids = [int(x) for x in args.decode_ids.split(",")] if args.decode_ids else []
        rendered, tok_files = None, {}
    else:
        rendered, prompt_ids, decode_ids, tok_files = tokenize_prompt(ck)
    heads = sorted({0, tcfg.linear_num_heads // 3, 2 * tcfg.linear_num_heads // 3, tcfg.linear_num_heads - 1})
    phases = phases_for(prompt_ids, decode_ids)
    T, D = len(prompt_ids), len(decode_ids)
    print(f"prompt {T} tokens, {D} decode steps, chain {chain}, record {RECORD_LAYERS}", flush=True)

    source = {
        "model": {"repo": args.repo or ck.meta.get("source_repo"),
                  "revision": args.revision or ck.meta.get("source_revision")},
        "reference": reference_info(),
        "runtime": runtime_info(args.threads),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "prompt": {"messages": PROMPT_MESSAGES if rendered is not None else None,
                   "rendered": rendered, "token_ids": prompt_ids, "files_sha256": tok_files,
                   "decode_text": DECODE_TEXT if rendered is not None else None,
                   "decode_token_ids": decode_ids},
        "chain_layers": list(chain),
        "record_layers": list(RECORD_LAYERS),
    }

    # FP32 chain, layer-major: layer i runs the prompt and every decode step, then layer i + 1 loads.
    rec = Recorder()
    embed = make_embed(ck, "fp32")
    streams = {p: embed(torch.tensor(ids)).unsqueeze(1).expand(-1, tcfg.hc_mult, -1).contiguous()
               for p, ids in phases}  # as Glm5NextTextModel.forward: every stream starts as the embedding
    touched, kinds = {}, {}
    with torch.no_grad(), rec.kda_capture():
        for i in chain:
            print(f"layer {i} (fp32)", flush=True)
            layer = build_layer(tcfg, i, ck, "fp32")
            if i in RECORD_LAYERS:
                rec.attach(layer, i)
            streams = run_layer(tcfg, layer, i, streams, rec)
            if i in RECORD_LAYERS and not isinstance(layer.self_attn, M.Glm5NextTextLinearAttention):
                norm_inputs = {p: rec.data[i][p]["attn_norm"] for p, _ in phases}
                run_attention_variant(tcfg, layer.self_attn, i, VARIANT_TOPK, norm_inputs, rec)
            rec.detach()
            if isinstance(layer.mlp, M.Glm5NextTextMoE):
                touched[i] = sorted(layer.mlp.experts.gate_up_proj.touched)
            kinds[i] = ("kda" if isinstance(layer.self_attn, M.Glm5NextTextLinearAttention) else "dsa",
                        "moe" if isinstance(layer.mlp, M.Glm5NextTextMoE) else "dense")
            del layer

    # Head on the last layer's output, for the rows a decoder turns into logits: the last prompt token,
    # then each decode step.
    head = Head(tcfg, ck, "fp32")
    last = torch.cat([streams["prefill"][-1:]] + [streams[p] for p, _ in phases[1:]])
    with torch.no_grad():
        head_collapsed, head_norm, logits = head(last)
        emb_rows = embed(torch.tensor(prompt_ids + decode_ids))

    fp32_policy = ("fp32: FP8 weights dequantised with transformers' Fp8Dequantize to FP32, BF16/F32 tensors "
                   "upcast; every parameter and activation FP32 on CPU")
    written = []
    for i in RECORD_LAYERS:
        kind, mlp = kinds[i]
        notes = {
            "layer": i, "attention": kind, "mlp": mlp, "dtype_policy": fp32_policy,
            "prompt_tokens": T, "decode_steps": D,
            "prefill": "the whole prompt in one forward (KDA uses the reference's chunked path)",
            "decode": "one token per forward with the reference cache, continuing from the prompt (KDA "
                      "recurrent path); decode.<name> tensors are stacked over steps, decode.sN.<name> are "
                      "per step. The state before step 0 is in the matching -prefill set.",
        }
        if kind == "kda":
            notes["state_heads"] = heads
            notes["weights"] = ("weights.*: the layer's small KDA tensors as the checkpoint stores them (exact); "
                                "the FP32 goldens use them upcast. The projections (q/k/v/o, f_a/f_b, g_a/g_b, "
                                "b_proj) are read from the checkpoint")
            notes["conv_rounding"] = CONV_ROUNDING_NOTE
        else:
            notes["index_layout"] = ("idx.topk rows hold min(512, pools) selected pools x 4 tokens in "
                                     "descending score order (tokens of pools not yet complete or visible "
                                     "are -1), then the tail slots (index_kpool - 1 = 3, -1 where unused), "
                                     "then -1 padding to 2051")
            notes["variant"] = (f"k{VARIANT_TOPK}.*: the same attention module and weights with "
                                f"index_topk={VARIANT_TOPK} (4 pools + tail), fed prefill/decode attn_norm; "
                                "not the model's configuration")
        if mlp == "moe":
            notes["experts_used"] = touched[i]
        per_phase = rec.data[i]
        merge_qkv_projections(per_phase)
        src = source | {"dtype_policy": "fp32",
                        "weights_sha256": ck.digest(n for n in ck.read if f"layers.{i}." in n)}
        for phase in ("prefill", "decode"):
            w = SetWriter(args.out, f"layer{i:02d}-{phase}", src, notes | {"phase": phase})
            if phase == "prefill":
                write_prefill(w, per_phase)
            else:
                write_decode(w, per_phase, heads, skip_decode=("mla.probs",))
            if kind == "kda":
                for n in KDA_SMALL_WEIGHTS:
                    t = ck.get(f"{TEXT_PREFIX}layers.{i}.{n}")
                    desc = f"checkpoint tensor {TEXT_PREFIX}layers.{i}.{n} as stored ({str(t.dtype)[6:]})"
                    if "conv1d" in n:
                        desc += "; the module's conv1d weight is cat(q, k, v conv weights) along dim 0"
                    w.add(f"weights.{n}", t, desc)
            w.close()
            written.append(w.name)

    w = SetWriter(args.out, "head", source | {"dtype_policy": "fp32", "weights_sha256": ck.digest(
        [TEXT_PREFIX + "embed_tokens.weight", TEXT_PREFIX + "norm.weight", "lm_head.weight"])},
        {"dtype_policy": fp32_policy,
         "rows": f"head.* rows: the last prompt token, then each of the {D} decode steps",
         "caution": (f"the input is layer {chain[-1]}'s output, not layer {tcfg.num_hidden_layers - 1}'s: these "
                     "logits test the final collapse, norm and LM head, not the model's predictions"
                     if chain[-1] != tcfg.num_hidden_layers - 1 else
                     "the input is the last decoder layer's output: these are the model's logits")})
    w.add("embed.rows", emb_rows, "embedding rows for the prompt then the decode tokens [tokens, hidden]")
    w.add("head.in_streams", last, f"layer {chain[-1]} output streams for the head rows [rows, hc, hidden]")
    w.add("head.collapsed", head_collapsed, "HyperHead: unweighted mean over the 4 streams")
    w.add("head.norm", head_norm, "final RMSNorm output")
    w.add("head.logits", logits, "LM head logits, FP32 [rows, vocab], one row per forward")
    w.close()
    written.append(w.name)

    # Native dtypes, per recorded layer, on the FP32 goldens' inputs.
    print("native (bf16) pass", flush=True)
    nrec = Recorder()
    nw = SetWriter(args.out, "native", source | {"dtype_policy": "native", "weights_sha256": None}, {})
    stats = {}
    with torch.no_grad(), nrec.kda_capture():
        for i in RECORD_LAYERS:
            nl = build_layer(tcfg, i, ck, "native")
            nrec.attach(nl, i)
            inputs = {p: rec.data[i][p]["in_streams"] for p, _ in phases}
            run_layer(tcfg, nl, i, inputs, nrec, torch.bfloat16)
            nrec.detach()
            per = nrec.data[i]
            merge_qkv_projections(per)
            steps = [p for p, _ in phases]

            def cat(k, per=per, steps=steps):
                return torch.cat([per[p][k] for p in steps])

            keep = ["out_streams", "attn_out", "mlp_out"]
            if isinstance(nl.self_attn, M.Glm5NextTextLinearAttention):
                keep += list(NATIVE_KDA_KEYS)
                nw.add(f"L{i:02d}.prefill.kda.state_heads", per["prefill"]["kda.state"][heads],
                       f"KDA state after the prompt, heads {heads}, FP32 (native run)")
                nw.add(f"L{i:02d}.decode.kda.state_heads", per[steps[-1]]["kda.state"][heads],
                       f"KDA state after the last decode step, heads {heads}, FP32 (native run)")
            else:
                keep += ["idx.topk", "mla.latent"]
            if isinstance(nl.mlp, M.Glm5NextTextMoE):
                keep += ["moe.topk_ids_sorted", "moe.topk_weights_sorted", "moe.routed_out"]
            for k in keep:
                nw.add(f"L{i:02d}.{k}", cat(k), "native run: " + describe(k) + " (prompt rows, then decode rows)")
            fp = rec.data[i]
            if isinstance(nl.self_attn, M.Glm5NextTextLinearAttention):
                qkv = torch.cat([cat(f"kda.{x}").flatten(1) for x in "qkv"], -1)
                stats[f"L{i:02d}.conv_fused_vs_reference"] = fused_conv_vs_reference(
                    cat("kda.qkv_preconv"), nl.self_attn.conv1d.weight, qkv)
            stats[f"L{i:02d}.out_streams"] = rel_err(cat("out_streams"), torch.cat([fp[p]["out_streams"] for p in steps]))
            stats[f"L{i:02d}.mlp_out"] = rel_err(cat("mlp_out"), torch.cat([fp[p]["mlp_out"] for p in steps]))
            if "moe.topk_ids_sorted" in keep:
                a = cat("moe.topk_ids_sorted")
                b = torch.cat([fp[p]["moe.topk_ids_sorted"] for p in steps])
                stats[f"L{i:02d}.router_rows_identical"] = f"{int((a == b).all(-1).sum())}/{a.shape[0]}"
            del nl
        nhead = Head(tcfg, ck, "native")
        _, _, nlogits = nhead(last.to(torch.bfloat16))
    nw.add("head.logits", nlogits, "native run: LM head on the FP32 head.in_streams cast to BF16 [rows, vocab]")
    stats["head.logits"] = rel_err(nlogits, logits)
    stats["head.argmax_identical"] = f"{int((nlogits.argmax(-1) == logits.argmax(-1)).sum())}/{logits.shape[0]}"
    nw.notes = {
        "dtype_policy": ("native: the dtypes from_pretrained gives this checkpoint on a CPU (FP8 dequantised "
                         "to BF16 by Fp8Dequantize; BF16 parameters and activations, including the short "
                         "conv; F32 checkpoint tensors kept F32: A_log, dt_bias, mHC base and scale, router "
                         "correction bias). Internal FP32 upcasts are the reference's own."),
        "inputs": "each layer is fed its FP32 golden in_streams cast to BF16, with a fresh cache",
        "conv_rounding": CONV_ROUNDING_NOTE,
        "layout": "LNN.<name>: prompt rows then decode rows; state_heads use heads " + str(heads),
        "diff_vs_fp32": stats,
    }
    nw.source["weights_sha256"] = ck.digest(ck.read)
    nw.close()
    written.append(nw.name)

    with open(os.path.join(args.out, "source-tensors.tsv"), "w") as f:
        f.write("# checkpoint tensors read to make these goldens: name, stored dtype, shape, sha256 of raw bytes\n")
        for n in sorted(ck.read):
            dt, shape, h = ck.read[n]
            f.write(f"{n}\t{dt}\t{','.join(map(str, shape))}\t{h}\n")
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"wrote {', '.join(written)} in {time.monotonic() - started:.0f} s, peak RSS {peak_gib:.1f} GiB")
    return 0


# --------------------------------------------------------------------------------------------------
# verify / compare


DTYPE_SIZES = {"f32": 4, "bf16": 2, "i32": 4, "u8": 1}


def tensor_sets(root: str) -> dict[str, dict]:
    """The golden sets under `root` that this script writes (manifests with a "tensors" map).
    Other generators keep their own formats in sibling directories; those are skipped."""
    sets = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "manifest.json")
        if os.path.exists(path):
            with open(path) as f:
                man = json.load(f)
            if isinstance(man.get("tensors"), dict):
                sets[name] = man
    return sets


def verify(args) -> int:
    bad = 0
    for d in args.dirs:
        for set_dir, man in tensor_sets(d).items():
            for name, e in man["tensors"].items():
                with open(os.path.join(d, set_dir, e["file"]), "rb") as f:
                    raw = f.read()
                n = DTYPE_SIZES[e["dtype"]]
                for dim in e["shape"]:
                    n *= dim
                if sha256_bytes(raw) != e["sha256"] or len(raw) != n:
                    print(f"MISMATCH {set_dir}/{name}")
                    bad += 1
            print(f"{set_dir}: {len(man['tensors'])} tensors checked")
    print("verify: " + ("OK" if not bad else f"{bad} mismatches"))
    return 1 if bad else 0


def compare(args) -> int:
    a, b = args.a, args.b
    sets_a, sets_b = tensor_sets(a), tensor_sets(b)
    diff = len(set(sets_a) ^ set(sets_b))
    for s in sorted(set(sets_a) ^ set(sets_b)):
        print(f"ONLY IN ONE {s}")
    for s in sorted(set(sets_a) & set(sets_b)):
        ma, mb = sets_a[s], sets_b[s]
        for name, e in ma["tensors"].items():
            if mb["tensors"].get(name, {}).get("sha256") != e["sha256"]:
                print(f"DIFFERENT {s}/{name}")
                diff += 1
        extra = set(mb["tensors"]) - set(ma["tensors"])
        diff += len(extra)
        same_manifest = sha256_file(os.path.join(a, s, "manifest.json")) == sha256_file(os.path.join(b, s, "manifest.json"))
        print(f"{s}: {len(ma['tensors'])} tensors, manifest {'identical' if same_manifest else 'differs'}")
        diff += 0 if same_manifest else 1
    print("compare: " + ("identical" if not diff else f"{diff} differences"))
    return 1 if diff else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="run the reference and write golden sets")
    g.add_argument("--weights", action="append", required=True,
                   help="directory with safetensors + model.safetensors.index.json (repeatable; first match wins)")
    g.add_argument("--out", required=True, help="output directory (one subdirectory per golden set)")
    g.add_argument("--last-layer", type=int, default=max(RECORD_LAYERS),
                   help="run the chain through this layer (default 4; the last layer, 44, gives the model's logits)")
    g.add_argument("--threads", type=int, default=16,
                   help="CPU threads (default 16); part of the numerics contract, another count changes bits")
    g.add_argument("--repo", default=None, help="model repository to record (default: from the subset index)")
    g.add_argument("--revision", default=None, help="model revision to record (default: from the subset index)")
    g.add_argument("--prompt-ids", default=None, help="comma-separated prompt ids instead of the chat prompt (tests)")
    g.add_argument("--decode-ids", default=None, help="comma-separated decode ids (with --prompt-ids)")
    v = sub.add_parser("verify", help="check every .bin against its manifest digest")
    v.add_argument("dirs", nargs="+")
    c = sub.add_parser("compare", help="compare two output directories digest by digest")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    return {"generate": generate, "verify": verify, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
