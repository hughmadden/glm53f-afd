//! `glm53f-score`: the engine side of the KL gate (`docs/KL-GATE.md`, section 4).
//!
//! ```text
//! glm53f-score --checkpoint DIR --ranks A,B,C,D --plan PLAN.json --out DIR [--pass-rows R] [options]
//! glm53f-score --checkpoint DIR --experts local --plan PLAN.json --out DIR [--dev-load-layers N] [options]
//! ```
//!
//! One process on the coordinator GPU. It
//!
//! 1. reads the plan `harness/klgate.py plan` writes (schema `glm53f-kl-plan.v1`): per window its
//!    token ids, their digest and the rows to write; checks the vocabulary (154,880 columns), every
//!    id (below 154,856, the tokenizer's) and every window's digest (sha256 of the ids as
//!    little-endian u32);
//! 2. loads the coordinator's weights as `glm53f-serve` does (the embedding in page-locked host
//!    RAM, the rest on the GPU) and connects the routed experts: the four expert ranks
//!    (`--ranks`, over RDMA with `GLM53F_RDMA=1` in an `rdma` build), or the official FP8 experts
//!    on this GPU (`--experts local`);
//! 3. builds the forward as `glm53f-serve --prefill-rows R --prefill-lanes N` would (passes of
//!    `--pass-rows` rows, two lanes by default, up to four), one slot, no drafter;
//! 4. per window: a fresh slot (empty KV, zeroed KDA states; no prefix cache, host tier or
//!    sharing between windows), the raw ids (no BOS added, no template), `GlmForward::score` in
//!    passes of `--pass-rows` rows (8 or fewer: the decode path's row-independent kernels; more:
//!    the prefill path, a lane per 64 rows up to `--prefill-lanes`), no sampling, no drafting;
//!    the plan's rows'
//!    logits, every one of the 154,880 LM-head columns, streamed to
//!    `<out>/<window>.safetensors` as they come (at most 8 rows in memory);
//! 5. writes `<out>/run.json`: the build, the configuration, the plan, and per window the tokens,
//!    rows, passes and times.
//!
//! # Output
//!
//! `<out>/<window>.safetensors` ([`out::RowWriter`]): the header (8-byte little-endian length,
//! JSON padded with spaces to 8 bytes), then `positions` I32 `[k]` (the plan's rows, ascending)
//! and `logits` F32 `[k, 154880]` (row i: the logits of `positions[i]`, as the head computes them:
//! no softmax, temperature or mask). Metadata: `window_id`, `tokens_sha256` (of the ids fed),
//! `plan_sha256` (of the plan file) and `engine` (one line: the build and every numerics
//! choice). A window is written to `<window>.safetensors.partial` and renamed when complete.
//!
//! # Options
//!
//! | Option | Environment | Default | What |
//! |---|---|---|---|
//! | `--checkpoint DIR` | `GLM53F_CHECKPOINT_DIR` | required | The official FP8 checkpoint or its coordinator subset, with `config.json` |
//! | `--plan FILE` | | required | The plan (`klgate.py plan`) |
//! | `--out DIR` | | required | Where the windows' files and `run.json` go (created) |
//! | `--pass-rows R` | | 4096 | Rows of one pass, every lane's together (1 to 4,096 per lane); the gate runs 8 or fewer and 4096 |
//! | `--windows ID,...` | | every window | Score these windows of the plan only |
//! | `--experts remote\|local\|zero` | | `remote` | Where the routed experts run; `zero`: routed outputs of zeros (development) |
//! | `--ranks A,B,C,D` | `GLM53F_SPARK_ADDRS` | required for `remote` | The four ranks, `host:port` in rank order, on the RDMA fabric |
//! | `--experts-dir DIR` | `GLM53F_EXPERTS_DIR` | the checkpoint | `local`: a checkpoint holding the routed experts |
//! | `--local-experts-gib G` | | 4 | `local`: device memory for the experts, loaded on demand |
//! | `--prefill-lanes N` | | 2 | Lanes of a prefill pass, 1 to 4 (as `glm53f-serve`) |
//! | `--kda-chunked-prefill` | | off | KDA of passes over 8 rows through the chunked kernel (a numerics change under test) |
//! | `--fp8-act bf16\|dynamic` | | `bf16` | FP8 projections of up to 8 rows: BF16 activations (W8A16) or the checkpoint's dynamic E4M3 (W8A8) |
//! | `--no-promote-k32` | | off | The FP8 tensor-core GEMM accumulates whole 128-blocks in the tensor core |
//! | `--kda-fp8` | `GLM53F_KDA_FP8=1` | off | Numerics under test (D2): the KDA q\|k\|v\|b and o projections quantized at load to FP8 block-128 (as `glm53f-serve`) |
//! | `--kda-state-bf16` | `GLM53F_KDA_STATE_BF16=1` | off | Numerics under test (D8): the KDA recurrent states in BF16 |
//! | `--prefill-w8a16` | `GLM53F_PREFILL_W8A16=1` | off | Numerics under test: FP8 projections over 8 rows with BF16 activations (W8A16) |
//! | `--kda-prefill-w8a8` | `GLM53F_KDA_PREFILL_W8A8=1` | off | With `--kda-fp8 --prefill-w8a16`: the FP8 KDA projections keep E4M3 activations over 8 rows |
//! | `--dev-layers 0-N` | | off | Development: decoder layers 0 to N only, then the head (as `glm53f-serve`) |
//! | `--dev-load-layers N` | | off | Development: load decoder layers 0 to N - 1 and run all 45 on repeats of them |
//!
//! The expert wire reads its own variables, as in `glm53f-serve`: `GLM53F_RDMA=1`,
//! `GLM53F_WIRE_NOCRC=1` (which the ranks must set too) and the others `docs/RUNNING.md` lists.
//!
//! **Development.** `--dev-layers`, `--dev-load-layers` and `--experts zero` make a development
//! run: its logits are meaningless, it says so at start, and its engine line (in every file) begins
//! with `DEVELOPMENT`. In one, `--experts local` gives zeros for the MoE layers whose experts the
//! experts directory lacks (the tests' subset, `GLM53F_EXPERTS_DIR`, holds layers 3 and 4);
//! otherwise it refuses to start without every MoE layer's experts.

pub mod out;
pub mod plan;

#[cfg(feature = "cuda")]
pub mod engine;

use std::path::PathBuf;

use glm53f_serve::{parse_dev_layers, Numerics, MAX_LANE_ROWS, MAX_PREFILL_LANES, RANKS};

/// Usage, for `--help` and errors.
pub const USAGE: &str = "usage:
  glm53f-score --checkpoint <dir> --ranks <a,b,c,d> --plan <plan.json> --out <dir> [options]
  glm53f-score --checkpoint <dir> --experts local [--experts-dir <dir>] --plan <plan.json> --out <dir> [options]
options:
  --pass-rows <r>        rows of one pass, every lane's together (default 4096; 8 or fewer: the
                         decode path)
  --windows <id,...>     score these windows of the plan only
  --experts remote|local|zero  --local-experts-gib <g>  --prefill-lanes 1-4
  --kda-chunked-prefill  --fp8-act bf16|dynamic  --no-promote-k32
numerics under test (off by default, as glm53f-serve):
  --kda-fp8              KDA projections quantized to FP8 block-128 at load (D2)
  --kda-state-bf16       KDA recurrent states stored in BF16 (D8)
  --prefill-w8a16        FP8 projections over 8 rows with BF16 activations
  --kda-prefill-w8a8     with the two above: the FP8 KDA projections keep E4M3 activations
development (the logits are meaningless):
  --dev-layers 0-N       decoder layers 0..=N only
  --dev-load-layers N    load decoder layers 0..N-1, run all 45 on repeats of them
  --experts zero         routed outputs of zeros (with --experts local in a development run, the
                         MoE layers the experts directory lacks give zeros)";

/// Where the routed experts run.
#[derive(Clone, Debug, PartialEq)]
pub enum Experts {
    /// The four ranks, `host:port` in rank order.
    Remote(Vec<String>),
    /// The official FP8 experts on this GPU, loaded from `dir` into `gib` GiB on demand.
    Local { dir: PathBuf, gib: f64 },
    /// Routed outputs of zeros (development).
    Zero,
}

/// FP8 projections of up to 8 rows: their activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8Act {
    /// BF16 (W8A16, every product exact in f32).
    Bf16,
    /// The checkpoint's dynamic E4M3 per 128-group (W8A8).
    Dynamic,
}

/// The scorer's options.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub checkpoint: PathBuf,
    pub plan: PathBuf,
    pub out: PathBuf,
    pub pass_rows: usize,
    /// Window ids to score (None: every window of the plan).
    pub windows: Option<Vec<String>>,
    pub experts: Experts,
    pub lanes: usize,
    pub kda_chunked_prefill: bool,
    pub fp8_act: Fp8Act,
    pub promote_k32: bool,
    /// Numerics under test (`glm53f-serve`'s flags).
    pub numerics: Numerics,
    /// Development: the decoder layers run (a prefix), or the layers loaded (all run on repeats
    /// of them).
    pub dev_layers: Option<usize>,
    pub dev_load_layers: Option<usize>,
}

fn parse_ranks(s: &str) -> Result<Vec<String>, String> {
    let addrs: Vec<String> = s
        .split(',')
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect();
    if addrs.len() != RANKS {
        return Err(format!(
            "--ranks: {} addresses, expected {RANKS} (host:port in rank order)",
            addrs.len()
        ));
    }
    for a in &addrs {
        let ok = a
            .rsplit_once(':')
            .is_some_and(|(h, p)| !h.is_empty() && p.parse::<u16>().is_ok());
        if !ok {
            return Err(format!("--ranks: {a:?} is not host:port"));
        }
    }
    Ok(addrs)
}

fn number<T: std::str::FromStr>(flag: &str, v: &str) -> Result<T, String> {
    v.parse()
        .map_err(|_| format!("{flag}: {v:?} is not a valid number"))
}

impl Options {
    /// Parse the command line (without the program name); `env` reads the environment.
    pub fn parse(args: &[String], env: &dyn Fn(&str) -> Option<String>) -> Result<Options, String> {
        let mut checkpoint = env("GLM53F_CHECKPOINT_DIR").map(PathBuf::from);
        let mut ranks = env("GLM53F_SPARK_ADDRS");
        let mut experts_dir = env("GLM53F_EXPERTS_DIR").map(PathBuf::from);
        let (mut plan, mut out, mut windows) = (None, None, None);
        let mut experts = "remote".to_string();
        let mut local_gib = 4.0f64;
        let (mut pass_rows, mut lanes) = (4096usize, 2usize);
        let (mut chunked, mut fp8_act, mut promote_k32) = (false, Fp8Act::Bf16, true);
        let (mut dev_layers, mut dev_load_layers) = (None, None);
        let mut numerics = Numerics::from_env(env);
        let mut it = args.iter();
        while let Some(k) = it.next() {
            if numerics.flag(k) {
                continue;
            }
            let mut val = || it.next().cloned().ok_or(format!("{k} needs a value"));
            match k.as_str() {
                "--checkpoint" => checkpoint = Some(PathBuf::from(val()?)),
                "--plan" => plan = Some(PathBuf::from(val()?)),
                "--out" => out = Some(PathBuf::from(val()?)),
                "--pass-rows" => pass_rows = number(k, &val()?)?,
                "--windows" => {
                    let ids: Vec<String> = val()?
                        .split(',')
                        .map(|w| w.trim().to_string())
                        .filter(|w| !w.is_empty())
                        .collect();
                    if ids.is_empty() {
                        return Err("--windows: no window ids".into());
                    }
                    windows = Some(ids);
                }
                "--experts" => experts = val()?,
                "--ranks" => ranks = Some(val()?),
                "--experts-dir" => experts_dir = Some(PathBuf::from(val()?)),
                "--local-experts-gib" => local_gib = number(k, &val()?)?,
                "--prefill-lanes" => lanes = number(k, &val()?)?,
                "--kda-chunked-prefill" => chunked = true,
                "--fp8-act" => {
                    fp8_act = match val()?.as_str() {
                        "bf16" => Fp8Act::Bf16,
                        "dynamic" => Fp8Act::Dynamic,
                        other => {
                            return Err(format!("--fp8-act {other}: expected bf16 or dynamic"))
                        }
                    }
                }
                "--no-promote-k32" => promote_k32 = false,
                "--dev-layers" => dev_layers = Some(parse_dev_layers(&val()?)?),
                "--dev-load-layers" => dev_load_layers = Some(number(k, &val()?)?),
                other => return Err(format!("unknown argument {other}")),
            }
        }
        let checkpoint = checkpoint
            .ok_or("--checkpoint (or GLM53F_CHECKPOINT_DIR): the coordinator's weights")?;
        let plan = plan.ok_or("--plan: the plan klgate.py plan writes")?;
        let out = out.ok_or("--out: the directory for the windows' logits")?;
        let experts = match experts.as_str() {
            "remote" => Experts::Remote(parse_ranks(&ranks.ok_or(
                "--ranks (or GLM53F_SPARK_ADDRS): the four expert ranks' fabric addresses, \
                 host:port in rank order",
            )?)?),
            "local" => Experts::Local {
                dir: experts_dir.unwrap_or_else(|| checkpoint.clone()),
                gib: local_gib,
            },
            "zero" => Experts::Zero,
            other => return Err(format!("--experts {other}: expected remote, local or zero")),
        };
        if !(1..=MAX_PREFILL_LANES).contains(&lanes) {
            return Err(format!("--prefill-lanes {lanes}: 1 to {MAX_PREFILL_LANES}"));
        }
        if !(1..=lanes * MAX_LANE_ROWS).contains(&pass_rows) {
            return Err(format!(
                "--pass-rows {pass_rows}: 1 to {} with {lanes} lane(s) (at most {MAX_LANE_ROWS} \
                 per lane)",
                lanes * MAX_LANE_ROWS
            ));
        }
        if !(local_gib.is_finite() && local_gib > 0.0) {
            return Err("--local-experts-gib takes a size in GiB".into());
        }
        if dev_layers.is_some() && dev_load_layers.is_some() {
            return Err("--dev-layers and --dev-load-layers exclude each other".into());
        }
        if dev_layers == Some(0) || dev_load_layers == Some(0) {
            return Err("--dev-layers and --dev-load-layers take positive values".into());
        }
        Ok(Options {
            checkpoint,
            plan,
            out,
            pass_rows,
            windows,
            experts,
            lanes,
            // The flag reaches either parser: glm53f-serve's numerics take it first.
            kda_chunked_prefill: chunked || numerics.kda_chunked_prefill,
            fp8_act,
            promote_k32,
            numerics,
            dev_layers,
            dev_load_layers,
        })
    }

    /// A development run: its logits are meaningless.
    pub fn development(&self) -> bool {
        self.dev_layers.is_some() || self.dev_load_layers.is_some() || self.experts == Experts::Zero
    }
}

/// The build's revision (`build.rs`): the commit, `-dirty` when the engine's sources differ.
pub const REVISION: &str = env!("GLM53F_SCORE_REVISION");
/// The target the kernels were compiled for.
pub const CUDA_ARCH: &str = env!("GLM53F_SCORE_CUDA_ARCH");

/// What the engine line describes beyond the options: the model run and the GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct Build {
    /// Decoder layers run, of the model's; decoder layers loaded (fewer: the rest repeat them).
    pub layers: usize,
    pub model_layers: usize,
    pub loaded: usize,
    /// The MoE layers run whose routed experts are zeros (development).
    pub zero_moe_layers: usize,
    /// The MoE layers run.
    pub moe_layers: usize,
    /// The GPU: compute capability and multiprocessors.
    pub gpu: String,
}

/// The engine line every output carries: the build, and every choice that moves the numerics.
pub fn engine_line(o: &Options, b: &Build) -> String {
    let mut parts = Vec::new();
    if o.development() {
        parts.push("DEVELOPMENT, the logits are meaningless".to_string());
    }
    parts.push(format!(
        "glm53f-score {} revision {REVISION}, kernels {CUDA_ARCH} on {}",
        env!("CARGO_PKG_VERSION"),
        b.gpu
    ));
    parts.push(if b.loaded < b.layers {
        format!(
            "decoder layers 0-{} of {} on the weights of layers 0-{} repeated, then the head",
            b.layers - 1,
            b.model_layers,
            b.loaded - 1
        )
    } else {
        format!(
            "decoder layers 0-{} of {}, then the head",
            b.layers - 1,
            b.model_layers
        )
    });
    let (small, act) = match o.fp8_act {
        Fp8Act::Bf16 => ("W8A16", "BF16"),
        Fp8Act::Dynamic => ("W8A8", "dynamic E4M3"),
    };
    let n = &o.numerics;
    let beyond = if n.prefill_w8a16 && n.kda_fp8 && n.kda_prefill_w8a8 {
        format!(
            "W8A16 beyond (BF16 tiles of the FP8 weights, cuBLAS), but W8A8 for the KDA \
             projections with {} accumulation",
            if o.promote_k32 {
                "k32-promoted"
            } else {
                "whole-block tensor-core"
            }
        )
    } else if n.prefill_w8a16 {
        "W8A16 beyond (BF16 tiles of the FP8 weights, cuBLAS)".to_string()
    } else {
        format!(
            "W8A8 beyond with {} accumulation",
            if o.promote_k32 {
                "k32-promoted"
            } else {
                "whole-block tensor-core"
            }
        )
    };
    parts.push(format!(
        "non-expert weights: the official FP8 checkpoint (FP8 block-128 projections {small} up to \
         8 rows ({act} activations), {beyond}; {} KDA projections; BF16 indexer projections: \
         GEMV up to 8 rows, cuBLAS beyond)",
        if n.kda_fp8 {
            "FP8 block-128 (quantized at load)"
        } else {
            "BF16"
        }
    ));
    let local = b.moe_layers - b.zero_moe_layers;
    parts.push(match &o.experts {
        Experts::Remote(_) => format!(
            "routed experts: the {RANKS} expert ranks' EXL3 4-bit shares, FP8 E4M3 wire rows \
             (UE8M0 per 32), BF16 partial planes summed in rank order"
        ),
        Experts::Local { .. } if b.zero_moe_layers == 0 => {
            format!("routed experts: the official FP8 experts on this GPU ({act} activations)")
        }
        Experts::Local { .. } => format!(
            "routed experts: the official FP8 experts on this GPU for {local} MoE layers, zeros \
             for the other {}",
            b.zero_moe_layers
        ),
        Experts::Zero => "routed experts: zeros".to_string(),
    });
    parts.push(format!(
        "KV: MLA latent FP8 E4M3 (528-byte records), pooled index keys FP8, KDA states {}",
        if n.kda_state_bf16 { "BF16" } else { "F32" }
    ));
    parts.push(if o.kda_chunked_prefill {
        "KDA over 8 rows: the chunked prefill kernel".to_string()
    } else {
        "KDA: the chain".to_string()
    });
    parts.push(format!(
        "passes of {} rows ({}; {} lane(s))",
        o.pass_rows,
        if o.pass_rows <= 8 {
            "the decode path"
        } else {
            "the prefill path"
        },
        o.lanes
    ));
    parts.push(
        "LM head: BF16 GEMV in groups of up to 8 rows, F32 logits, all 154880 columns".to_string(),
    );
    parts.join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    const RANK_LIST: &str = "rank0:8600,rank1:8600,rank2:8600,rank3:8600";

    #[test]
    fn defaults_and_the_environment() {
        let env = |k: &str| match k {
            "GLM53F_CHECKPOINT_DIR" => Some("/w".to_string()),
            "GLM53F_SPARK_ADDRS" => Some(RANK_LIST.to_string()),
            "GLM53F_EXPERTS_DIR" => Some("/e".to_string()),
            _ => None,
        };
        let o = Options::parse(&args("--plan p.json --out o"), &env).unwrap();
        assert_eq!(o.checkpoint, PathBuf::from("/w"));
        assert_eq!(o.experts, Experts::Remote(parse_ranks(RANK_LIST).unwrap()));
        assert_eq!((o.pass_rows, o.lanes, o.windows.clone()), (4096, 2, None));
        assert_eq!(
            (o.kda_chunked_prefill, o.fp8_act, o.promote_k32),
            (false, Fp8Act::Bf16, true)
        );
        assert_eq!(o.numerics, Numerics::default());
        assert!(!o.development());
        let o = Options::parse(
            &args(
                "--checkpoint /c --plan p --out o --experts local --pass-rows 8 --windows a,b \
                 --prefill-lanes 1 --kda-chunked-prefill --fp8-act dynamic --no-promote-k32",
            ),
            &env,
        )
        .unwrap();
        assert_eq!(o.checkpoint, PathBuf::from("/c"));
        assert_eq!(
            o.experts,
            Experts::Local {
                dir: PathBuf::from("/e"),
                gib: 4.0
            }
        );
        assert_eq!((o.pass_rows, o.lanes), (8, 1));
        let four = Options::parse(&args("--plan p --out o --pass-rows 16384 --prefill-lanes 4"), &env)
            .unwrap();
        assert_eq!((four.pass_rows, four.lanes), (16384, 4));
        assert_eq!(o.windows, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(
            (o.kda_chunked_prefill, o.fp8_act, o.promote_k32),
            (true, Fp8Act::Dynamic, false)
        );
        // The numerics under test, as glm53f-serve takes them; the engine line names them.
        let o = Options::parse(
            &args("--plan p --out o --kda-fp8 --kda-state-bf16 --prefill-w8a16"),
            &env,
        )
        .unwrap();
        assert!(o.numerics.kda_fp8 && o.numerics.kda_state_bf16 && o.numerics.prefill_w8a16);
        let b = Build {
            layers: 45,
            model_layers: 45,
            loaded: 45,
            zero_moe_layers: 0,
            moe_layers: 42,
            gpu: "sm_120 (170 SMs)".into(),
        };
        let line = engine_line(&o, &b);
        assert!(
            line.contains("FP8 block-128 (quantized at load) KDA projections"),
            "{line}"
        );
        assert!(line.contains("W8A16 beyond"), "{line}");
        assert!(line.contains("KDA states BF16"), "{line}");
        let plain = engine_line(
            &Options::parse(&args("--plan p --out o"), &env).unwrap(),
            &b,
        );
        assert!(plain.contains("BF16 KDA projections") && plain.contains("KDA states F32"));
        assert!(
            plain.contains("W8A8 beyond with k32-promoted accumulation"),
            "{plain}"
        );
        // Local experts default to the checkpoint.
        let o = Options::parse(
            &args("--checkpoint /c --plan p --out o --experts local"),
            &no_env,
        )
        .unwrap();
        assert_eq!(
            o.experts,
            Experts::Local {
                dir: PathBuf::from("/c"),
                gib: 4.0
            }
        );
    }

    #[test]
    fn development_runs_say_so() {
        let base = "--checkpoint /c --plan p --out o";
        for (extra, dev) in [
            ("--experts local", false),
            ("--experts zero", true),
            ("--experts local --dev-layers 0-4", true),
            ("--experts local --dev-load-layers 5", true),
        ] {
            let o = Options::parse(&args(&format!("{base} {extra}")), &no_env).unwrap();
            assert_eq!(o.development(), dev, "{extra}");
            let b = Build {
                layers: 45,
                model_layers: 45,
                loaded: 5,
                zero_moe_layers: 40,
                moe_layers: 42,
                gpu: "sm_89 (128 SMs)".into(),
            };
            let line = engine_line(&o, &b);
            assert_eq!(line.starts_with("DEVELOPMENT"), dev, "{line}");
            assert!(line.contains("passes of 4096 rows (the prefill path; 2 lane(s))"));
            assert!(line.contains("on the weights of layers 0-4 repeated"));
            assert!(!line.contains('\n'));
        }
        let o = Options::parse(&args(&format!("{base} --dev-layers 0-4")), &|k| {
            (k == "GLM53F_SPARK_ADDRS").then(|| RANK_LIST.to_string())
        })
        .unwrap();
        assert_eq!(o.dev_layers, Some(5));
    }

    #[test]
    fn bad_options_are_refused() {
        let bad = [
            "",                                        // nothing
            "--plan p --out o",                        // no checkpoint
            "--checkpoint /c --out o --experts zero",  // no plan
            "--checkpoint /c --plan p --experts zero", // no out
            "--checkpoint /c --plan p --out o",        // remote without ranks
            "--checkpoint /c --plan p --out o --ranks a:1,b:2,c:3",
            "--checkpoint /c --plan p --out o --ranks a:1,b:2,c:3,d",
            "--checkpoint /c --plan p --out o --experts gpu",
            "--checkpoint /c --plan p --out o --experts zero --pass-rows 0",
            "--checkpoint /c --plan p --out o --experts zero --pass-rows 8193",
            "--checkpoint /c --plan p --out o --experts zero --pass-rows 4097 --prefill-lanes 1",
            "--checkpoint /c --plan p --out o --experts zero --prefill-lanes 5",
            "--checkpoint /c --plan p --out o --experts zero --pass-rows 12289 --prefill-lanes 3",
            "--checkpoint /c --plan p --out o --experts zero --fp8-act e5m2",
            "--checkpoint /c --plan p --out o --experts zero --windows ,",
            "--checkpoint /c --plan p --out o --experts local --local-experts-gib 0",
            "--checkpoint /c --plan p --out o --experts zero --dev-layers 3-4",
            "--checkpoint /c --plan p --out o --experts zero --dev-load-layers 0",
            "--checkpoint /c --plan p --out o --experts zero --dev-layers 0-4 --dev-load-layers 5",
            "--checkpoint /c --plan p --out o --experts zero --drafter d",
            "--checkpoint /c --plan p --out o --experts zero --pass-rows",
        ];
        for b in bad {
            assert!(
                Options::parse(&args(b), &no_env).is_err(),
                "{b:?} was accepted"
            );
        }
    }
}
