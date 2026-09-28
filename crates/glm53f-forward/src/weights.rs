//! The coordinator's weights on the device, loaded from the official FP8 checkpoint (or its
//! coordinator subset: the same tensor names, fetched without the routed experts).
//!
//! What is resident, per decoder layer run:
//!
//! | Part | Device form |
//! |---|---|
//! | mHC (both sites) | `fn` BF16 `[24][16,384]`, `base` and `scale` f32 |
//! | `input_layernorm`, `post_attention_layernorm` | BF16 `[4096]` |
//! | KDA q, k, v, b projections | one BF16 `[24,640][4096]` (q, k, v, beta rows stacked), or FP8 with [`WeightOptions::kda_fp8`] |
//! | KDA f_a, g_a | one BF16 `[256][4096]` |
//! | KDA f_b, g_b | one BF16 `[2][8192][128]` (two GEMV groups) |
//! | KDA o_proj | BF16 `[4096][8192]`, or FP8 with [`WeightOptions::kda_fp8`] |
//! | KDA conv, `A_log`, `dt_bias`, `o_norm` | BF16 `[24,576][4]` (q, k, v concatenated), f32, f32, BF16 |
//! | DSA q_a, kv_a, q_b, o | FP8 E4M3 with f32 `weight_scale_inv` per 128 x 128 block |
//! | DSA `kv_b_proj` | BF16 `[32,768][512]` |
//! | DSA norms | `q_a_layernorm` BF16; `kv_a_layernorm` f32 (the latent writer's input) |
//! | indexer `wq_b` | BF16 `[4096][1536]` |
//! | indexer wk, compress gate, weights_proj | one BF16 `[288][4096]` |
//! | indexer `k_norm`, `ape` | f32 |
//! | dense MLP (layers 0-2), shared expert | FP8 `gate_proj` and `up_proj` stacked `[2 I][4096]`, `down_proj` `[4096][I]` |
//! | router | BF16 `[288][4096]`, f32 correction bias |
//! | head | final norm BF16; LM head BF16 `[154,880][4096]` |
//!
//! The embedding is not here: it stays in page-locked host memory ([`crate::embed`]).
//! Routed experts are the expert backend's ([`crate::experts`]).
//!
//! **FP8 KDA projections (decision D2, [`WeightOptions::kda_fp8`], off by default).** The official
//! checkpoint ships the 34 KDA layers' projections in BF16 (9.37 GB, 61% of the coordinator's
//! weights). With the option, the fused q|k|v|b projection and `o_proj` are quantized at load
//! time to FP8 E4M3 with 128 x 128 block scales, the checkpoint's own scheme for its other FP8
//! weights (`glm53f-layers`' `glm53f_fp8_quantize_weight`: per block `scale = amax / 448`,
//! `q = e4m3(w / scale)`). They then run the FP8 GEMMs the DSA projections run. q|k|v|b has
//! 24,640 rows = 192 blocks of 128 and a 64-row block for beta, which has scales of its own (the
//! FP8 kernels take a partial last block of rows). The gate projections (`f_a`, `g_a`, `f_b`,
//! `g_b`: 6 MB a layer, 2% of the KDA bytes, and the forget gate's decays compound over the
//! sequence), the conv, the norms, `A_log` and `dt_bias` stay as shipped. Resident KDA bytes:
//! 275.5 MB a layer in BF16, 141.0 MB with the option (9.37 GB and 4.80 GB over 34 layers: 4.26
//! GiB less).

use std::path::Path;
use std::sync::Arc;

use glm53f_model::catalog::{Catalog, CheckpointFormat, Coverage, LAYERS};
use glm53f_model::config::{AttnKind, ModelConfig};
use glm53f_model::dtype::DType;
use glm53f_model::safetensors::Checkpoint;

use crate::device::{launched, DeviceBuffer, Stream};
use crate::error::{invalid, Result};
use crate::gemm::{Bf16Mat, Fp8Mat};
use crate::shape::*;

/// Load-time choices for the coordinator's weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightOptions {
    /// Decision D2: the KDA layers' q|k|v|b and `o_proj` quantized at load to FP8 E4M3 with
    /// 128 x 128 block scales (module documentation). Off: BF16 as the checkpoint ships them.
    pub kda_fp8: bool,
}

/// An FP8 block-128 weight on the device.
pub struct Fp8W {
    pub w: DeviceBuffer,
    pub scales: DeviceBuffer,
    pub n: usize,
    pub k: usize,
}

impl Fp8W {
    pub fn mat(&self) -> Fp8Mat {
        Fp8Mat {
            w: self.w.ptr(0),
            scales: self.scales.ptr(0),
            n: self.n,
            k: self.k,
        }
    }
}

/// A BF16 weight `[groups][n][k]` on the device.
pub struct Bf16W {
    pub buf: DeviceBuffer,
    pub n: usize,
    pub k: usize,
    pub groups: usize,
}

impl Bf16W {
    pub fn mat(&self) -> Bf16Mat {
        Bf16Mat {
            ptr: self.buf.ptr(0),
            n: self.n,
            k: self.k,
            ld: self.k,
            groups: self.groups,
            gstride: self.n * self.k,
        }
    }
}

/// A projection the checkpoint ships in BF16: as shipped, or quantized to FP8 block-128 at load
/// ([`WeightOptions::kda_fp8`]).
pub enum ProjW {
    Bf16(Bf16W),
    Fp8(Fp8W),
}

impl ProjW {
    /// Device bytes of the weight (and its scales).
    pub fn bytes(&self) -> usize {
        match self {
            ProjW::Bf16(w) => w.mat().bytes(),
            ProjW::Fp8(w) => w.mat().bytes(),
        }
    }
}

/// One mHC site: `fn` BF16 `[24][4 * hidden]`, `base` f32 `[24]`, `scale` f32 `[3]`.
pub struct HcW {
    pub fn_: DeviceBuffer,
    pub base: DeviceBuffer,
    pub scale: DeviceBuffer,
}

pub struct KdaW {
    /// q | k | v | beta-logit projections, `[24,640][4096]`.
    pub qkvb: ProjW,
    /// f_a | g_a, `[256][4096]`.
    pub fga: Bf16W,
    /// f_b and g_b as two groups, `[2][8192][128]`.
    pub fgb: Bf16W,
    /// `[4096][8192]`.
    pub o: ProjW,
    /// BF16 `[24,576][4]`.
    pub conv_w: DeviceBuffer,
    /// f32 `[64]`.
    pub a_log: DeviceBuffer,
    /// f32 `[8192]`.
    pub dt_bias: DeviceBuffer,
    /// BF16 `[128]`.
    pub o_norm: DeviceBuffer,
}

pub struct DsaW {
    pub q_a: Fp8W,
    pub kv_a: Fp8W,
    /// BF16 `[1536]`.
    pub q_a_norm: DeviceBuffer,
    pub q_b: Fp8W,
    /// f32 `[512]` (the latent writer applies it).
    pub kv_a_norm: DeviceBuffer,
    /// BF16 `[64 * 512][512]`.
    pub kv_b: DeviceBuffer,
    pub o: Fp8W,
    /// Indexer `wq_b`, BF16 `[4096][1536]`.
    pub idx_wq_b: Bf16W,
    /// wk | compress gate | weights_proj, BF16 `[288][4096]`.
    pub idx_proj: Bf16W,
    /// f32 `[128]` each.
    pub k_norm_w: DeviceBuffer,
    pub k_norm_b: DeviceBuffer,
    /// f32 `[4][128]`.
    pub ape: DeviceBuffer,
}

/// A SwiGLU MLP: gate and up stacked, then down.
pub struct MlpW {
    pub gate_up: Fp8W,
    pub down: Fp8W,
    pub inter: usize,
}

pub enum AttnW {
    Kda(KdaW),
    Dsa(DsaW),
}

pub enum FfnW {
    Dense(MlpW),
    Moe {
        /// BF16 `[288][4096]`.
        router: DeviceBuffer,
        /// f32 `[288]`.
        bias: DeviceBuffer,
        shared: MlpW,
    },
}

pub struct LayerW {
    pub attn_hc: HcW,
    pub ffn_hc: HcW,
    pub input_norm: DeviceBuffer,
    pub post_attn_norm: DeviceBuffer,
    pub attn: AttnW,
    pub ffn: FfnW,
}

pub struct HeadW {
    /// BF16 `[4096]`.
    pub norm: DeviceBuffer,
    pub lm_head: Bf16W,
}

/// The coordinator's resident weights for decoder layers `0 .. shape.layers` and the head.
pub struct DeviceModel {
    pub shape: ModelShape,
    /// Per decoder layer (shared between layers only by [`DeviceModel::load_repeating`]).
    pub layers: Vec<Arc<LayerW>>,
    pub head: HeadW,
    /// Device bytes held.
    pub bytes: usize,
    /// The load-time choices it was loaded with.
    pub opts: WeightOptions,
}

/// Reads checked tensors from a checkpoint and uploads them.
pub struct Loader<'a> {
    pub ckpt: &'a Checkpoint,
    pub bytes: usize,
    pub opts: WeightOptions,
    /// The stream the load-time quantization runs on (created on first use).
    stream: Option<Stream>,
}

impl<'a> Loader<'a> {
    pub fn new(ckpt: &'a Checkpoint) -> Loader<'a> {
        Self::with_options(ckpt, WeightOptions::default())
    }

    pub fn with_options(ckpt: &'a Checkpoint, opts: WeightOptions) -> Loader<'a> {
        Loader {
            ckpt,
            bytes: 0,
            opts,
            stream: None,
        }
    }

    /// A BF16 weight quantized on the device to FP8 E4M3 with 128 x 128 block scales
    /// (`glm53f_fp8_quantize_weight`); the BF16 copy is freed.
    pub fn quantize(&mut self, w: Bf16W) -> Result<Fp8W> {
        if w.groups != 1 || !w.k.is_multiple_of(128) || !w.n.is_multiple_of(8) {
            return Err(invalid!(
                "FP8 quantization needs one group, k % 128 == 0 and n % 8 == 0 ({} x {} x {})",
                w.groups,
                w.n,
                w.k
            ));
        }
        if self.stream.is_none() {
            self.stream = Some(Stream::new()?);
        }
        let st = self.stream.as_ref().unwrap();
        let q = Fp8W {
            w: DeviceBuffer::alloc(w.n * w.k)?,
            scales: DeviceBuffer::alloc(w.n.div_ceil(128) * (w.k / 128) * 4)?,
            n: w.n,
            k: w.k,
        };
        // SAFETY: `w` holds [n][k] BF16, `q` [n][k] codes and [ceil(n/128)][k/128] scales.
        launched(
            unsafe {
                glm53f_layers::ffi::glm53f_fp8_quantize_weight(
                    w.buf.ptr(0),
                    w.n as i32,
                    w.k as i32,
                    q.w.ptr(0),
                    q.scales.ptr(0),
                    st.raw().cast(),
                )
            },
            "glm53f_fp8_quantize_weight",
        )?;
        st.synchronize()?;
        self.bytes = self.bytes - w.buf.bytes() + q.w.bytes() + q.scales.bytes();
        Ok(q)
    }

    /// A projection that ships in BF16: as shipped, or quantized when `fp8`.
    fn proj(&mut self, w: Bf16W, fp8: bool) -> Result<ProjW> {
        Ok(if fp8 {
            ProjW::Fp8(self.quantize(w)?)
        } else {
            ProjW::Bf16(w)
        })
    }

    /// A tensor's bytes, checked against the dtype and shape the engine expects.
    pub fn read(&self, name: &str, dtype: DType, shape: &[usize]) -> Result<Vec<u8>> {
        let (_, e) = self
            .ckpt
            .get(name)
            .ok_or_else(|| invalid!("the checkpoint has no tensor {name}"))?;
        let want: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        if e.dtype != dtype || e.shape != want {
            return Err(invalid!(
                "{name}: {} {:?} in the checkpoint, expected {dtype} {want:?}",
                e.dtype,
                e.shape
            ));
        }
        Ok(self.ckpt.read_tensor(name)?)
    }

    /// Tensors concatenated in order into one device buffer.
    pub fn upload(&mut self, parts: &[(&str, DType, &[usize])]) -> Result<DeviceBuffer> {
        let total: usize = parts
            .iter()
            .map(|(_, d, s)| d.size() as usize * s.iter().product::<usize>())
            .sum();
        let buf = DeviceBuffer::alloc(total)?;
        let mut at = 0;
        for (name, dtype, shape) in parts {
            let bytes = self.read(name, *dtype, shape)?;
            buf.upload_at(at, &bytes)?;
            at += bytes.len();
        }
        self.bytes += total;
        Ok(buf)
    }

    /// A BF16 tensor, or several stacked along their rows, as one BF16 weight.
    pub fn bf16(
        &mut self,
        names: &[&str],
        rows: &[usize],
        k: usize,
        groups: usize,
    ) -> Result<Bf16W> {
        let shapes: Vec<[usize; 2]> = rows.iter().map(|&r| [r, k]).collect();
        let parts: Vec<(&str, DType, &[usize])> = names
            .iter()
            .zip(&shapes)
            .map(|(n, s)| (*n, DType::BF16, &s[..]))
            .collect();
        let n: usize = rows.iter().sum::<usize>() / groups;
        Ok(Bf16W {
            buf: self.upload(&parts)?,
            n,
            k,
            groups,
        })
    }

    /// FP8 weights stacked along their output rows (each a multiple of 128), with their
    /// scales stacked the same way.
    pub fn fp8(&mut self, modules: &[&str], rows: &[usize], k: usize) -> Result<Fp8W> {
        for &r in rows {
            if !r.is_multiple_of(128) || !k.is_multiple_of(128) {
                return Err(invalid!("FP8 stacking needs whole 128-blocks ({r} x {k})"));
            }
        }
        let wn: Vec<String> = modules.iter().map(|m| format!("{m}.weight")).collect();
        let sn: Vec<String> = modules
            .iter()
            .map(|m| format!("{m}.weight_scale_inv"))
            .collect();
        let ws: Vec<[usize; 2]> = rows.iter().map(|&r| [r, k]).collect();
        let ss: Vec<[usize; 2]> = rows.iter().map(|&r| [r / 128, k / 128]).collect();
        let wp: Vec<(&str, DType, &[usize])> = wn
            .iter()
            .zip(&ws)
            .map(|(n, s)| (n.as_str(), DType::F8E4M3, &s[..]))
            .collect();
        let sp: Vec<(&str, DType, &[usize])> = sn
            .iter()
            .zip(&ss)
            .map(|(n, s)| (n.as_str(), DType::F32, &s[..]))
            .collect();
        Ok(Fp8W {
            w: self.upload(&wp)?,
            scales: self.upload(&sp)?,
            n: rows.iter().sum(),
            k,
        })
    }

    /// A BF16 or F32 vector as f32 on the device.
    pub fn f32(&mut self, name: &str, dtype: DType, len: usize) -> Result<DeviceBuffer> {
        let bytes = self.read(name, dtype, &[len])?;
        let v: Vec<f32> = match dtype {
            DType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            DType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect(),
            other => return Err(invalid!("{name}: {other} is not a float vector")),
        };
        self.bytes += v.len() * 4;
        DeviceBuffer::from_slice(&v)
    }

    /// A tensor of any shape as f32 (from BF16 or F32).
    pub fn f32_shaped(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[usize],
    ) -> Result<DeviceBuffer> {
        let bytes = self.read(name, dtype, shape)?;
        let v: Vec<f32> = match dtype {
            DType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            DType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect(),
            other => return Err(invalid!("{name}: {other} is not a float tensor")),
        };
        self.bytes += v.len() * 4;
        DeviceBuffer::from_slice(&v)
    }

    fn hc(&mut self, p: &str, site: &str) -> Result<HcW> {
        Ok(HcW {
            fn_: self.upload(&[(&format!("{p}hc_{site}_fn"), DType::BF16, &[24, HC * HIDDEN])])?,
            base: self.upload(&[(&format!("{p}hc_{site}_base"), DType::F32, &[24])])?,
            scale: self.upload(&[(&format!("{p}hc_{site}_scale"), DType::F32, &[3])])?,
        })
    }

    fn kda(&mut self, p: &str) -> Result<KdaW> {
        let a = format!("{p}self_attn.");
        let n = |s: &str| format!("{a}{s}");
        let conv_names: Vec<String> = ["q", "k", "v"]
            .iter()
            .map(|c| n(&format!("{c}_conv1d.weight")))
            .collect();
        let conv_parts: Vec<(&str, DType, &[usize])> = conv_names
            .iter()
            .map(|c| (c.as_str(), DType::BF16, &[KDA_WIDTH, 1, 4][..]))
            .collect();
        let fp8 = self.opts.kda_fp8;
        let qkvb = self.bf16(
            &[
                &n("q_proj.weight"),
                &n("k_proj.weight"),
                &n("v_proj.weight"),
                &n("b_proj.weight"),
            ],
            &[KDA_WIDTH, KDA_WIDTH, KDA_WIDTH, KDA_HEADS],
            HIDDEN,
            1,
        )?;
        let qkvb = self.proj(qkvb, fp8)?;
        let o = self.bf16(&[&n("o_proj.weight")], &[HIDDEN], KDA_WIDTH, 1)?;
        let o = self.proj(o, fp8)?;
        Ok(KdaW {
            qkvb,
            fga: self.bf16(
                &[&n("f_a_proj.weight"), &n("g_a_proj.weight")],
                &[KDA_DIM, KDA_DIM],
                HIDDEN,
                1,
            )?,
            fgb: self.bf16(
                &[&n("f_b_proj.weight"), &n("g_b_proj.weight")],
                &[KDA_WIDTH, KDA_WIDTH],
                KDA_DIM,
                2,
            )?,
            o,
            conv_w: self.upload(&conv_parts)?,
            a_log: self.upload(&[(&n("A_log"), DType::F32, &[KDA_HEADS])])?,
            dt_bias: self.upload(&[(&n("dt_bias"), DType::F32, &[KDA_WIDTH])])?,
            o_norm: self.upload(&[(&n("o_norm.weight"), DType::BF16, &[KDA_DIM])])?,
        })
    }

    fn dsa(&mut self, p: &str) -> Result<DsaW> {
        let a = format!("{p}self_attn.");
        let n = |s: &str| format!("{a}{s}");
        Ok(DsaW {
            q_a: self.fp8(&[&n("q_a_proj")], &[Q_LORA], HIDDEN)?,
            kv_a: self.fp8(&[&n("kv_a_proj_with_mqa")], &[KV_LORA], HIDDEN)?,
            q_a_norm: self.upload(&[(&n("q_a_layernorm.weight"), DType::BF16, &[Q_LORA])])?,
            q_b: self.fp8(&[&n("q_b_proj")], &[MLA_HEADS * QK_HEAD], Q_LORA)?,
            kv_a_norm: self.f32(&n("kv_a_layernorm.weight"), DType::BF16, KV_LORA)?,
            kv_b: self.upload(&[(
                &n("kv_b_proj.weight"),
                DType::BF16,
                &[MLA_HEADS * (QK_HEAD + V_HEAD), KV_LORA],
            )])?,
            o: self.fp8(&[&n("o_proj")], &[HIDDEN], MLA_HEADS * V_HEAD)?,
            idx_wq_b: self.bf16(
                &[&n("indexer.wq_b.weight")],
                &[INDEX_HEADS * INDEX_DIM],
                Q_LORA,
                1,
            )?,
            idx_proj: self.bf16(
                &[
                    &n("indexer.wk.weight"),
                    &n("indexer.index_kpool_compress_gate"),
                    &n("indexer.weights_proj.weight"),
                ],
                &[INDEX_DIM, INDEX_DIM, INDEX_HEADS],
                HIDDEN,
                1,
            )?,
            k_norm_w: self.f32(&n("indexer.k_norm.weight"), DType::BF16, INDEX_DIM)?,
            k_norm_b: self.f32(&n("indexer.k_norm.bias"), DType::BF16, INDEX_DIM)?,
            ape: self.f32_shaped(
                &n("indexer.index_kpool_compress_ape"),
                DType::BF16,
                &[4, INDEX_DIM],
            )?,
        })
    }

    fn mlp(&mut self, prefix: &str, inter: usize) -> Result<MlpW> {
        Ok(MlpW {
            gate_up: self.fp8(
                &[&format!("{prefix}gate_proj"), &format!("{prefix}up_proj")],
                &[inter, inter],
                HIDDEN,
            )?,
            down: self.fp8(&[&format!("{prefix}down_proj")], &[HIDDEN], inter)?,
            inter,
        })
    }

    fn layer(&mut self, shape: &ModelShape, l: usize) -> Result<LayerW> {
        let p = format!("{LAYERS}{l}.");
        let attn = match shape.attn[l] {
            AttnKind::Kda => AttnW::Kda(self.kda(&p)?),
            AttnKind::Dsa => AttnW::Dsa(self.dsa(&p)?),
        };
        let ffn = if shape.is_moe(l) {
            FfnW::Moe {
                router: self.upload(&[(
                    &format!("{p}mlp.gate.weight"),
                    DType::BF16,
                    &[EXPERTS, HIDDEN],
                )])?,
                bias: self.upload(&[(
                    &format!("{p}mlp.gate.e_score_correction_bias"),
                    DType::F32,
                    &[EXPERTS],
                )])?,
                shared: self.mlp(&format!("{p}mlp.shared_experts."), SHARED_INTER)?,
            }
        } else {
            FfnW::Dense(self.mlp(&format!("{p}mlp."), DENSE_INTER)?)
        };
        Ok(LayerW {
            attn_hc: self.hc(&p, "attn")?,
            ffn_hc: self.hc(&p, "ffn")?,
            input_norm: self.upload(&[(
                &format!("{p}input_layernorm.weight"),
                DType::BF16,
                &[HIDDEN],
            )])?,
            post_attn_norm: self.upload(&[(
                &format!("{p}post_attention_layernorm.weight"),
                DType::BF16,
                &[HIDDEN],
            )])?,
            attn,
            ffn,
        })
    }
}

/// Open a checkpoint directory and check it against its `config.json` (a subset is fine: the
/// coordinator's tensors fetched on their own).
pub fn open_checkpoint(dir: &Path) -> Result<(ModelConfig, Checkpoint)> {
    let cfg = ModelConfig::load(&dir.join("config.json"))?;
    let ckpt = Checkpoint::open(dir)?;
    Catalog::from_shards(
        &cfg,
        &ckpt.shards,
        Some(CheckpointFormat::OfficialFp8),
        Coverage::Subset,
    )?;
    Ok((cfg, ckpt))
}

impl DeviceModel {
    /// Load decoder layers `0 .. shape.layers` and the head from the checkpoint.
    pub fn load(ckpt: &Checkpoint, shape: &ModelShape) -> Result<DeviceModel> {
        Self::load_repeating(ckpt, shape, shape.layers)
    }

    /// Tests and development only: decoder layers `0 .. loaded` from the checkpoint, and every
    /// later layer of `shape` on the weights of the last loaded layer of the same kinds
    /// (attention and MLP), so a forward over all 45 layers runs in the memory of `loaded`
    /// layers (layers 0-4 cover every kind). The KV and positional state stay per layer; the
    /// output is meaningless.
    pub fn load_repeating(
        ckpt: &Checkpoint,
        shape: &ModelShape,
        loaded: usize,
    ) -> Result<DeviceModel> {
        Self::load_with(ckpt, shape, loaded, WeightOptions::default())
    }

    /// [`DeviceModel::load_repeating`] with load-time choices ([`WeightOptions`]).
    pub fn load_with(
        ckpt: &Checkpoint,
        shape: &ModelShape,
        loaded: usize,
        opts: WeightOptions,
    ) -> Result<DeviceModel> {
        let mut ld = Loader::with_options(ckpt, opts);
        let mut layers: Vec<Arc<LayerW>> = Vec::with_capacity(shape.layers);
        for l in 0..shape.layers {
            if l < loaded {
                layers.push(Arc::new(ld.layer(shape, l)?));
                continue;
            }
            let same = (0..loaded.min(l))
                .rev()
                .find(|&k| shape.attn[k] == shape.attn[l] && shape.mlp[k] == shape.mlp[l])
                .ok_or_else(|| {
                    invalid!("layer {l}: no layer of its kinds among the {loaded} loaded")
                })?;
            layers.push(layers[same].clone());
        }
        let head = HeadW {
            norm: ld.upload(&[(glm53f_model::catalog::FINAL_NORM, DType::BF16, &[HIDDEN])])?,
            lm_head: ld.bf16(&[glm53f_model::catalog::LM_HEAD], &[VOCAB], HIDDEN, 1)?,
        };
        Ok(DeviceModel {
            shape: shape.clone(),
            layers,
            head,
            bytes: ld.bytes,
            opts,
        })
    }
}
