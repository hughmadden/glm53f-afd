//! `glm53f-serve`: the coordinator daemon of glm53f-afd.
//!
//! ```text
//! glm53f-serve --checkpoint DIR --ranks A,B,C,D [--listen ADDR] [options]
//! glm53f-serve --checkpoint DIR --experts local [--dev-layers 0-N] [options]
//! ```
//!
//! One process on the coordinator GPU. At start it
//!
//! 1. loads the coordinator's weights: every non-expert tensor of the official FP8 checkpoint
//!    (the whole checkpoint, or its coordinator subset with the same tensor names). The
//!    embedding stays in page-locked host RAM; everything else goes to the GPU;
//! 2. connects the routed experts: the four expert ranks (`--experts remote`, the default), or
//!    the official FP8 experts on this GPU (`--experts local`, development on one GPU);
//! 3. with `--drafter`, loads the DFlash2 drafter onto the GPU next to the weights (each slot's
//!    fixed state then holds its 40.16 MiB context ring);
//! 4. allocates every buffer the forward's passes use (both prefill lanes' scratch for
//!    `--prefill-rows`, the verify scratch, the attention workspaces for any context), the
//!    expert exchange's buffers and, with the drafter, its tap buffer and working memory, then
//!    sizes the KV from the memory left: each slot's fixed state and a page pool shared by the
//!    slots and by the snapshot marks (below); the drafter is attached to the forward:
//!    speculative decoding, up to 7 drafts a step;
//! 5. starts the engine and the scheduler (`glm53f-coordinator`) over the forward
//!    (`glm53f-forward`'s `ServedForward`) and serves the OpenAI-compatible API
//!    (`glm53f-api`, GLM's completion dialect).
//!
//! A request then runs HTTP -> queue -> scheduler -> forward (attention, dense and shared MLPs,
//! router on this GPU; routed experts on the ranks) -> sampler -> tokens streamed back.
//!
//! # Options
//!
//! | Option | Environment | Default | What |
//! |---|---|---|---|
//! | `--checkpoint DIR` | `GLM53F_CHECKPOINT_DIR` | required | The official FP8 checkpoint or its coordinator subset, with `config.json` |
//! | `--tokenizer FILE` | `GLM53F_TOKENIZER` | `DIR/tokenizer.json` | The tokenizer |
//! | `--chat-template FILE` | | `DIR/chat_template.jinja` | The official chat template (any other is refused) |
//! | `--experts remote\|local` | | `remote` | Where the routed experts run |
//! | `--ranks A,B,C,D` | `GLM53F_SPARK_ADDRS` | required for `remote` | The four ranks, `host:port` in rank order, on the RDMA fabric |
//! | `--experts-dir DIR` | `GLM53F_EXPERTS_DIR` | the checkpoint | `local`: a checkpoint holding the routed experts of the layers run |
//! | `--local-experts-gib G` | | 4 | `local`: device memory for the experts, loaded on demand |
//! | `--listen ADDR` | `GLM53F_API_ADDR` | `127.0.0.1:8100` | The API's address |
//! | `--slots N` | `GLM53F_MAX_SLOTS` | 16 | Requests with device state at once (1-64) |
//! | `--max-context T` | | the model's (1,048,576) | Tokens one request can hold |
//! | `--kv-gib G` | | the free memory less the reserve | The KV page pool |
//! | `--reserve-gib G` | | 1 | Device memory left free after everything is allocated (kernel modules loaded on first use, the sampler, allocator slack) |
//! | `--prefill-rows R` | `GLM53F_PREFILL_ROWS` | 4096 | Rows of one prefill pass, both lanes together (at most 4,096 per lane: the wire's request cap) |
//! | `--prefill-lanes N` | `GLM53F_PREFILL_LANES` | 2 | Lanes of a prefill pass: 2 overlaps one lane's attention with the other's experts on the ranks, 1 runs the pass serially |
//! | `--drafter DIR` | `GLM53F_DFLASH_DIR` | off | The DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2`: `config.json`, `model.safetensors`); needs decoder layers 0-43 |
//! | `--dev-layers 0-N` | | off | Development mode (below) |
//!
//! The shell reads more of its own: `GLM53F_QUEUE_DEPTH`, `GLM53F_QUEUE_WAIT_MS`,
//! `GLM53F_HOST_CACHE_GB` (the host RAM tier, 0 for none), `GLM53F_PREFILL_SEGMENT_MS`,
//! `GLM53F_PREFIX_CACHE_ENTRIES`; with a drafter `GLM53F_SPEC` (0: decode one token a step),
//! `GLM53F_SPEC_POLICY` (`fixed`, `conf`, else the chain cut at `GLM53F_SPEC_TAU`) and
//! `GLM53F_DFLASH_SAMPLED_WALK` (0: sampled requests draft greedily too); the wire client
//! `GLM53F_RDMA=1` (an `rdma` build) with
//! `GLM53F_WIRE_NOCRC=1` (which the ranks must set too), `GLM53F_WIRE_MIN_GBPS`,
//! `GLM53F_WIRE_INFLIGHT` (1 holds RDMA to one exchange in flight), `GLM53F_TIMELINE`,
//! `GLM53F_PROFILE` (with the forward's lane trace).
//!
//! **The fabric.** Expert traffic runs only on the RDMA fabric: the wire client refuses a rank
//! reached through an address without a RoCE v2 device at the floor rate. `GLM53F_WIRE_ALLOW_LAN=1`
//! lifts the check for tests (loopback needs no lifting).
//!
//! # Prefill rows and lanes
//!
//! The default, 4,096 rows in two lanes of 2,048, comes from the first run on the fabric (4,096-row
//! passes, one lane, traced per MoE layer): the coordinator's work between exchanges took 34.6 ms,
//! the rank's compute 16.8 ms and the transfer about 8.6 ms, all serial. Two lanes overlap each
//! lane's coordinator work with the other lane's exchange, so a layer costs about twice the larger
//! of the two per lane. Taking the coordinator's work as linear in rows (0.25 ms at one row) and
//! the rank's rate as measured (155K, 208K and 243K rows/s at 1,024, 2,048 and 4,096 rows):
//!
//! | Pass (lanes of) | Coordinator per lane | Exchange per lane | Per layer | Per row |
//! |---|---:|---:|---:|---:|
//! | 2,048 (1,024) | 8.8 ms | 8.8 ms | 17.7 ms | 8.6 us |
//! | 4,096 (2,048) | 17.4 ms | 14.2 ms | 34.9 ms | 8.5 us |
//! | 8,192 (4,096) | 34.6 ms | 25.5 ms | 69.2 ms | 8.4 us |
//!
//! Today every size is coordinator-bound at about the same rate (about 2.7K tok/s over 42 MoE
//! layers and the rest), so the pass is sized for its other costs: 4,096 rows keeps the lanes'
//! scratch at about 3.0 GiB (8,192 would double it, out of the KV pool) and a pass at about
//! 1.5 s (the longest a running request waits for its next token). Lanes of 1,024 rows are
//! balanced today but fall behind once the coordinator gets faster: the ranks then set the pace,
//! at 6.9 us per row with lanes of 2,048 against 8.6 with lanes of 1,024 (6.2 with 4,096). The
//! flags let a run on the real hardware compare them; `glm53f-forward`'s lane trace
//! (`GLM53F_PROFILE=1`) shows where each layer's time goes.
//!
//! # Device memory
//!
//! Everything but the KV pool is allocated first, and the pool takes what is left less
//! `--reserve-gib`; a pass allocates nothing. Snapshot marks (the KDA states and conv windows of
//! a prompt or turn end, 141 MiB each) take pages of the pool (376 each), so admission, which
//! counts free pages, counts them too: under pressure it evicts retained snapshots to the host
//! tier, and a mark the pool has no room for is refused (that snapshot is skipped) instead of
//! running the device out of memory. The start-up log lists what was allocated for what.
//!
//! # Development mode
//!
//! `--dev-layers 0-N` runs decoder layers 0 to N only and applies the head to what comes out:
//! the whole serving loop works on one GPU with a slice of the weights (the checkpoint needs only
//! those layers and the head; the ranks, only the MoE layers among them, served with
//! `--allow-partial`). **Its output is meaningless text by design.** The daemon says so at
//! start and again on its serving line. Only a prefix of the layers can run: the forward's
//! weights and KV are laid out for layers `0 .. N + 1`. A drafter needs N >= 43 (its last tap
//! is the output of layer 42, read at the entry of layer 43); with fewer layers the daemon says
//! so and runs without it.

use std::path::PathBuf;

/// Usage, for `--help` and errors.
pub const USAGE: &str = "usage:
  glm53f-serve --checkpoint <dir> --ranks <a,b,c,d> [--listen <addr:port>] [options]
  glm53f-serve --checkpoint <dir> --experts local [--experts-dir <dir>] [--dev-layers 0-N] [options]
options:
  --tokenizer <file>  --chat-template <file>  --experts remote|local  --local-experts-gib <g>
  --slots <n>  --max-context <tokens>  --kv-gib <g>  --reserve-gib <g>
  --prefill-rows <r>  --prefill-lanes 1|2
  --drafter <dir>     the DFlash2 drafter: speculative decoding (needs decoder layers 0-43)
  --dev-layers 0-N    DEVELOPMENT: decoder layers 0..=N only; the output is meaningless text";

/// Expert ranks.
pub const RANKS: usize = 4;
/// Rows of one prefill lane at most (one exchange: the wire's request cap).
pub const MAX_LANE_ROWS: usize = 4096;
const GIB: f64 = (1u64 << 30) as f64;

/// Where the routed experts run.
#[derive(Clone, Debug, PartialEq)]
pub enum Experts {
    /// The four ranks, `host:port` in rank order.
    Remote(Vec<String>),
    /// The official FP8 experts on this GPU, loaded from `dir` into `gib` GiB on demand.
    Local { dir: PathBuf, gib: f64 },
}

/// The daemon's options.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub checkpoint: PathBuf,
    pub tokenizer: PathBuf,
    pub chat_template: PathBuf,
    pub experts: Experts,
    pub listen: String,
    pub slots: usize,
    /// None: the model's maximum context.
    pub max_context: Option<usize>,
    /// None: the free device memory less `reserve_gib` and the slots' fixed state.
    pub kv_gib: Option<f64>,
    pub reserve_gib: f64,
    /// Rows of one prefill pass (every lane's together), and its lanes.
    pub prefill_rows: usize,
    pub prefill_lanes: usize,
    /// The DFlash2 drafter's directory (none: no speculative decoding).
    pub drafter: Option<PathBuf>,
    /// Development mode: the number of decoder layers run (`--dev-layers 0-N` gives N + 1).
    pub dev_layers: Option<usize>,
}

/// `0-N`: the first N + 1 decoder layers.
pub fn parse_dev_layers(s: &str) -> Result<usize, String> {
    let bad = || format!("--dev-layers {s:?}: expected 0-N (a prefix of the decoder layers)");
    let (a, b) = s.split_once('-').ok_or_else(bad)?;
    let (a, b): (usize, usize) = (
        a.trim().parse().map_err(|_| bad())?,
        b.trim().parse().map_err(|_| bad())?,
    );
    if a != 0 {
        return Err(format!(
            "--dev-layers {s}: only a prefix of the layers can run (0-{b})"
        ));
    }
    Ok(b + 1)
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
        let mut tokenizer = env("GLM53F_TOKENIZER").map(PathBuf::from);
        let mut chat_template = None;
        let mut experts = "remote".to_string();
        let mut ranks = env("GLM53F_SPARK_ADDRS");
        let mut experts_dir = env("GLM53F_EXPERTS_DIR").map(PathBuf::from);
        let mut local_gib = 4.0;
        let mut listen = env("GLM53F_API_ADDR").unwrap_or_else(|| "127.0.0.1:8100".into());
        let mut slots = match env("GLM53F_MAX_SLOTS") {
            Some(v) => number("GLM53F_MAX_SLOTS", &v)?,
            None => 16,
        };
        let (mut max_context, mut kv_gib, mut reserve_gib) = (None, None, 1.0f64);
        let mut prefill_rows = match env("GLM53F_PREFILL_ROWS") {
            Some(v) => number("GLM53F_PREFILL_ROWS", &v)?,
            None => 4096,
        };
        let mut prefill_lanes = match env("GLM53F_PREFILL_LANES") {
            Some(v) => number("GLM53F_PREFILL_LANES", &v)?,
            None => 2,
        };
        let mut dev_layers = None;
        let mut drafter = env("GLM53F_DFLASH_DIR").map(PathBuf::from);
        let mut it = args.iter();
        while let Some(k) = it.next() {
            let mut val = || it.next().cloned().ok_or(format!("{k} needs a value"));
            match k.as_str() {
                "--checkpoint" => checkpoint = Some(PathBuf::from(val()?)),
                "--tokenizer" => tokenizer = Some(PathBuf::from(val()?)),
                "--chat-template" => chat_template = Some(PathBuf::from(val()?)),
                "--experts" => experts = val()?,
                "--ranks" => ranks = Some(val()?),
                "--experts-dir" => experts_dir = Some(PathBuf::from(val()?)),
                "--local-experts-gib" => local_gib = number(k, &val()?)?,
                "--listen" => listen = val()?,
                "--slots" => slots = number(k, &val()?)?,
                "--max-context" => max_context = Some(number(k, &val()?)?),
                "--kv-gib" => kv_gib = Some(number(k, &val()?)?),
                "--reserve-gib" => reserve_gib = number(k, &val()?)?,
                "--prefill-rows" => prefill_rows = number(k, &val()?)?,
                "--prefill-lanes" => prefill_lanes = number(k, &val()?)?,
                "--drafter" => drafter = Some(PathBuf::from(val()?)),
                "--dev-layers" => dev_layers = Some(parse_dev_layers(&val()?)?),
                other => return Err(format!("unknown argument {other}")),
            }
        }
        let checkpoint = checkpoint
            .ok_or("--checkpoint (or GLM53F_CHECKPOINT_DIR): the coordinator's weights")?;
        let experts = match experts.as_str() {
            "remote" => Experts::Remote(parse_ranks(&ranks.ok_or(
                "--ranks (or GLM53F_SPARK_ADDRS): the four expert ranks' fabric addresses, \
                 host:port in rank order",
            )?)?),
            "local" => Experts::Local {
                dir: experts_dir.unwrap_or_else(|| checkpoint.clone()),
                gib: local_gib,
            },
            other => return Err(format!("--experts {other}: expected remote or local")),
        };
        if !(1..=64).contains(&slots) {
            return Err(format!("--slots {slots}: 1 to 64"));
        }
        if !matches!(prefill_lanes, 1 | 2) {
            return Err(format!("--prefill-lanes {prefill_lanes}: 1 or 2"));
        }
        if !(1..=prefill_lanes * MAX_LANE_ROWS).contains(&prefill_rows) {
            return Err(format!(
                "--prefill-rows {prefill_rows}: 1 to {} with {prefill_lanes} lane(s) (at most \
                 {MAX_LANE_ROWS} per lane)",
                prefill_lanes * MAX_LANE_ROWS
            ));
        }
        if max_context == Some(0) || dev_layers == Some(0) {
            return Err("--max-context and --dev-layers take positive values".into());
        }
        let positive = |g: f64| g.is_finite() && g > 0.0;
        if !kv_gib.is_none_or(positive)
            || !positive(local_gib)
            || reserve_gib.is_nan()
            || reserve_gib < 0.0
        {
            return Err("--kv-gib, --local-experts-gib and --reserve-gib take sizes in GiB".into());
        }
        Ok(Options {
            tokenizer: tokenizer.unwrap_or_else(|| checkpoint.join("tokenizer.json")),
            chat_template: chat_template.unwrap_or_else(|| checkpoint.join("chat_template.jinja")),
            checkpoint,
            experts,
            listen,
            slots,
            max_context,
            kv_gib,
            reserve_gib,
            prefill_rows,
            prefill_lanes,
            drafter,
            dev_layers,
        })
    }
}

/// Physical pages of the KV pool: `kv_gib` when given, else the device's `free` bytes less
/// `reserve_gib` and `fixed` (the slots' positional state and page tables), in pages of
/// `page_bytes`.
pub fn kv_pages(
    kv_gib: Option<f64>,
    free: usize,
    reserve_gib: f64,
    fixed: usize,
    page_bytes: usize,
) -> Result<usize, String> {
    let bytes = match kv_gib {
        Some(g) => (g * GIB) as usize,
        None => free
            .checked_sub((reserve_gib * GIB) as usize + fixed)
            .ok_or_else(|| {
                format!(
                    "{:.2} GiB free on the GPU after the weights: the slots' state ({:.2} GiB) and \
                     the reserve ({reserve_gib} GiB) leave nothing for the KV pages",
                    free as f64 / GIB,
                    fixed as f64 / GIB
                )
            })?,
    };
    let pages = bytes / page_bytes.max(1);
    if pages == 0 {
        return Err(format!("a KV pool of {bytes} bytes holds no page"));
    }
    Ok(pages)
}

/// The development-mode banner: what runs, and that the output means nothing.
pub fn dev_banner(layers: usize, total: usize) -> String {
    let rule = "=".repeat(78);
    format!(
        "{rule}\nDEVELOPMENT MODE (--dev-layers 0-{}): decoder layers 0-{} of {total}, then the head.\n\
         The output is meaningless text by design. Do not serve this to users.\n{rule}",
        layers - 1,
        layers - 1
    )
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
            "GLM53F_MAX_SLOTS" => Some("8".to_string()),
            _ => None,
        };
        let o = Options::parse(&[], &env).unwrap();
        assert_eq!(o.checkpoint, PathBuf::from("/w"));
        assert_eq!(o.tokenizer, PathBuf::from("/w/tokenizer.json"));
        assert_eq!(o.chat_template, PathBuf::from("/w/chat_template.jinja"));
        assert_eq!(o.experts, Experts::Remote(parse_ranks(RANK_LIST).unwrap()));
        assert_eq!((o.listen.as_str(), o.slots), ("127.0.0.1:8100", 8));
        assert_eq!((o.max_context, o.kv_gib, o.reserve_gib), (None, None, 1.0));
        assert_eq!(
            (o.prefill_rows, o.prefill_lanes, o.dev_layers, o.drafter),
            (4096, 2, None, None)
        );
        // Flags win over the environment.
        let o = Options::parse(
            &args("--checkpoint /c --slots 2 --listen 0.0.0.0:9000 --tokenizer /t.json"),
            &env,
        )
        .unwrap();
        assert_eq!((o.checkpoint.to_str(), o.slots), (Some("/c"), 2));
        assert_eq!(o.listen, "0.0.0.0:9000");
        assert_eq!(o.tokenizer, PathBuf::from("/t.json"));
        // The drafter: from its flag, else the environment.
        let with = |k: &str| match k {
            "GLM53F_DFLASH_DIR" => Some("/d".to_string()),
            other => env(other),
        };
        assert_eq!(
            Options::parse(&[], &with).unwrap().drafter,
            Some(PathBuf::from("/d"))
        );
        let o = Options::parse(&args("--drafter /e"), &with).unwrap();
        assert_eq!(o.drafter, Some(PathBuf::from("/e")));
        // The prefill knobs: the environment, and flags over it.
        let env2 = |k: &str| match k {
            "GLM53F_PREFILL_ROWS" => Some("2048".to_string()),
            "GLM53F_PREFILL_LANES" => Some("1".to_string()),
            _ => env(k),
        };
        let o = Options::parse(&[], &env2).unwrap();
        assert_eq!((o.prefill_rows, o.prefill_lanes), (2048, 1));
        let o = Options::parse(&args("--prefill-rows 8192 --prefill-lanes 2"), &env2).unwrap();
        assert_eq!((o.prefill_rows, o.prefill_lanes), (8192, 2));
    }

    #[test]
    fn development_mode_runs_a_prefix_of_the_layers() {
        assert_eq!(parse_dev_layers("0-4"), Ok(5));
        assert_eq!(parse_dev_layers("0-0"), Ok(1));
        assert!(parse_dev_layers("3-4").unwrap_err().contains("prefix"));
        assert!(parse_dev_layers("4").is_err());
        assert!(parse_dev_layers("0-x").is_err());
        let o = Options::parse(
            &args("--checkpoint /c --experts local --dev-layers 0-4"),
            &no_env,
        )
        .unwrap();
        assert_eq!(o.dev_layers, Some(5));
        assert_eq!(
            o.experts,
            Experts::Local {
                dir: PathBuf::from("/c"),
                gib: 4.0
            }
        );
        let b = dev_banner(5, 45);
        assert!(
            b.contains("DEVELOPMENT MODE") && b.contains("0-4 of 45") && b.contains("meaningless")
        );
    }

    #[test]
    fn bad_options_are_refused() {
        let bad = [
            "",                                                    // no checkpoint
            "--checkpoint /c",                                     // remote without ranks
            "--checkpoint /c --ranks a:1,b:2,c:3",                 // three ranks
            "--checkpoint /c --ranks a:1,b:2,c:3,d",               // no port
            "--checkpoint /c --experts gpu",                       // unknown backend
            "--checkpoint /c --experts local --slots 0",           // no slots
            "--checkpoint /c --experts local --slots 65",          // too many
            "--checkpoint /c --experts local --prefill-rows 8193", // over two lanes of 4,096
            "--checkpoint /c --experts local --prefill-rows 5000 --prefill-lanes 1",
            "--checkpoint /c --experts local --prefill-lanes 3",
            "--checkpoint /c --experts local --prefill-rows 0",
            "--checkpoint /c --experts local --kv-gib 0",
            "--checkpoint /c --experts local --reserve-gib -1",
            "--checkpoint /c --experts local --max-context 0",
            "--checkpoint /c --experts local --frobnicate",
            "--checkpoint", // a flag without its value
            "--checkpoint /c --experts local --drafter",
        ];
        for b in bad {
            assert!(
                Options::parse(&args(b), &no_env).is_err(),
                "{b:?} was accepted"
            );
        }
    }

    #[test]
    fn the_kv_pool_takes_what_the_weights_and_the_reserve_leave() {
        let gib = 1usize << 30;
        let page = 394_944;
        // Given: that many GiB of pages.
        assert_eq!(kv_pages(Some(1.0), 0, 4.0, 0, page), Ok(gib / page));
        // Else: free less the reserve and the slots' fixed state.
        assert_eq!(
            kv_pages(None, 16 * gib, 4.0, 2 * gib, page),
            Ok(10 * gib / page)
        );
        assert!(kv_pages(None, 5 * gib, 4.0, 2 * gib, page).is_err());
        assert!(kv_pages(Some(1e-9), 0, 4.0, 0, page).is_err());
    }
}
