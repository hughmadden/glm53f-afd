//! The memory planner reproduces the tables of docs/SIZING.md (sections 2-6)
//! within their rounding. Tensor bytes come from catalogs derived from
//! config.json and, when `GLM53F_TEST_HEADERS_DIR` is set, also from the
//! published checkpoints' headers; both must give the same tables.

mod common;

use common::*;
use glm53f_model::catalog::{Catalog, CheckpointFormat, Coverage, Group};
use glm53f_model::planner::*;

struct Sources {
    what: &'static str,
    official: Catalog,
    exl3: Catalog,
    nvfp4: Catalog,
}

fn sources() -> Vec<Sources> {
    let c = config();
    let mut out = vec![Sources {
        what: "derived from config.json",
        official: Catalog::from_config(c, CheckpointFormat::OfficialFp8),
        exl3: Catalog::from_config(c, CheckpointFormat::Exl3 { bits: 4 }),
        nvfp4: Catalog::from_config(c, CheckpointFormat::Nvfp4),
    }];
    let read = |w| Catalog::from_shards(c, bundle(w).unwrap(), None, Coverage::Complete).unwrap();
    if bundle("official").is_some() {
        out.push(Sources {
            what: "from the published headers",
            official: read("official"),
            exl3: read("tr3"),
            nvfp4: read("nvfp4"),
        });
    }
    out
}

const DEVICE: u64 = 32_607 * MIB;

fn budget(s: &Sources, layout: Layout, slots: u64, runtime_gib: u64) -> GpuBudget {
    let cat = if layout.policy().shipped_bf16 {
        &s.exl3
    } else {
        &s.official
    };
    GpuBudget {
        device: DEVICE,
        weights: resident_weights(cat, layout.policy()).unwrap().total,
        drafter: drafter().weight_bytes(),
        slots,
        slot_state: SlotState::new(&config().text, Some(&drafter())),
        runtime: runtime_gib * GIB,
    }
}

/// Section 2: what grows with context.
#[test]
fn context_bytes_per_token() {
    let t = &config().text;
    let fp8 = KvGeometry::new(t, KvPrecision::Fp8);
    let bf16 = KvGeometry::new(t, KvPrecision::Bf16);
    assert_eq!((fp8.mla_record, fp8.index_record), (528, 132));
    assert_eq!(
        (fp8.mla_per_token(), fp8.index_per_pool() / 4),
        (5_808, 363)
    );
    assert_eq!(fp8.per_token(), 6_171.0);
    assert_eq!(
        (bf16.mla_per_token(), bf16.index_per_pool() / 4),
        (11_264, 704)
    );
    assert_eq!(bf16.per_token(), 11_968.0);
    assert_eq!(fp8.page_bytes(), 64 * 6_171);
    for (g, k256, m1) in [(fp8, "1.51", "6.03"), (bf16, "2.92", "11.69")] {
        assert_rounds_to(gib(g.bytes_for(TOKENS_256K)), k256, "one 256K request");
        assert_rounds_to(gib(g.bytes_for(TOKENS_1M)), m1, "one 1M request");
        assert_eq!(g.request_bytes(TOKENS_1M), g.bytes_for(TOKENS_1M));
    }
}

/// Section 3: what is fixed per active request.
#[test]
fn fixed_state_per_slot() {
    let s = SlotState::new(&config().text, Some(&drafter()));
    assert_eq!(s.kda_state, 136 * MIB);
    assert_rounds_to(mib(s.conv_state), "4.8", "short-convolution state");
    assert_rounds_to(mib(s.draft_kv), "40.2", "DFlash2 draft KV");
    assert_rounds_to(mib(s.total()), "181", "per slot");
    assert_rounds_to(gib(16 * s.total()), "2.8", "16 slots");
    assert_rounds_to(gib(8 * s.total()), "1.4", "8 slots");
    // Section 7: a KDA snapshot (state plus convolution).
    assert_rounds_to(mib(s.kda_snapshot()), "141", "KDA snapshot");
    assert_eq!(SlotState::new(&config().text, None).draft_kv, 0);
}

/// Section 4: coordinator weights.
#[test]
fn coordinator_weights() {
    for s in sources() {
        let w = |l: Layout| resident_weights(&s.official, l.policy()).unwrap();
        let (a, b, c) = (w(Layout::A), w(Layout::B), w(Layout::C));
        let d = resident_weights(&s.exl3, Layout::D.policy()).unwrap();
        let why = s.what;
        let gbs = |r: &ResidentWeights, gs: &[Group]| gb(gs.iter().map(|&g| r.group(g)).sum());
        assert_rounds_to(gbs(&a, &[Group::KdaAttention]), "9.37", why);
        assert_rounds_to(
            gbs(&a, &[Group::DsaAttention, Group::DsaIndexer]),
            "1.64",
            why,
        );
        assert_rounds_to(gbs(&a, &[Group::SharedExperts]), "1.06", why);
        assert_rounds_to(gbs(&a, &[Group::DenseMlp]), "0.45", why);
        assert_rounds_to(
            gbs(&a, &[Group::Routers, Group::Mhc, Group::Norms]),
            "0.17",
            why,
        );
        assert_rounds_to(gbs(&a, &[Group::LmHead]), "1.27", why);
        assert_rounds_to(gbs(&a, &[Group::Embedding]), "1.27", why);
        assert_eq!(b.group(Group::Embedding), 0);
        assert_rounds_to(gib(a.total), "14.18", why);
        assert_rounds_to(gib(b.total), "13.00", why);
        assert_rounds_to(gib(c.total), "8.64", why);
        // FP8 KDA: SIZING says 4.68 GB (half the BF16 bytes). Quantizing the nine
        // projections with 128 x 128 F32 scales and keeping the convolution
        // weights, norms, A_log and dt_bias as shipped gives 4.69 GB.
        assert_eq!(c.group(Group::KdaAttention), 4_688_231_168, "{why}");
        assert_rounds_to(gb(a.group(Group::KdaAttention)) / 2.0, "4.68", why);
        // Section 4 notes: the quantized checkpoints ship every non-expert
        // tensor in BF16, 15.4 GiB without the embedding.
        assert_rounds_to(gib(d.total), "15.4", why);
        assert_eq!(resident_weights(&s.nvfp4, Layout::D.policy()).unwrap(), d);
        assert!(resident_weights(&s.official, Layout::D.policy()).is_err());
        assert!(resident_weights(&s.exl3, Layout::B.policy()).is_err());
        // The knobs combine freely: FP8 KDA with the embedding on the GPU is
        // layout C plus the embedding.
        let fp8_kda_gpu_embed = WeightPolicy {
            embedding_on_gpu: true,
            ..Layout::C.policy()
        };
        let e = resident_weights(&s.official, fp8_kda_gpu_embed).unwrap();
        assert_eq!(e.total, c.total + a.group(Group::Embedding));
        // Optional rows.
        assert_rounds_to(gb(s.official.group_bytes(Group::MtpLayer)), "0.24", why);
        assert_rounds_to(gb(s.official.group_bytes(Group::Vision)), "1.13", why);
    }
    let dflash = drafter().weight_bytes();
    assert_rounds_to(gb(dflash), "2.34", "DFlash2");
    assert_rounds_to(gib(dflash), "2.18", "DFlash2");
}

/// Section 5: the RTX 5090 budget, 16 slots, DFlash2 resident, runtime 5 GiB
/// (low end of each range) and 3 GiB (high end).
#[test]
fn rtx5090_budget_table() {
    // (layout, runtime GiB, KV pool GiB, FP8 tokens M, 256K, 1M, BF16 tokens M)
    let table: [(Layout, u64, &str, &str, u64, u64, &str); 8] = [
        (Layout::A, 5, "7.7", "1.33", 5, 1, "0.69"),
        (Layout::A, 3, "9.7", "1.68", 6, 1, "0.87"),
        (Layout::B, 5, "8.8", "1.54", 5, 1, "0.79"),
        (Layout::B, 3, "10.8", "1.89", 7, 1, "0.97"),
        (Layout::C, 5, "13.2", "2.30", 8, 2, "1.18"),
        (Layout::C, 3, "15.2", "2.64", 10, 2, "1.36"),
        (Layout::D, 5, "6.4", "1.11", 4, 1, "0.57"),
        (Layout::D, 3, "8.4", "1.46", 5, 1, "0.75"),
    ];
    let t = &config().text;
    for s in sources() {
        for &(layout, rt, pool, fp8_m, k256, m1, bf16_m) in &table {
            let why = format!("{} layout {} runtime {rt} GiB", s.what, layout.letter());
            let p = plan_gpu(t, budget(&s, layout, 16, rt)).unwrap();
            assert_rounds_to(gib(p.kv_pool), pool, &why);
            assert_rounds_to(mtok(p.fp8.tokens), fp8_m, &why);
            assert_eq!(
                (p.fp8.requests_256k, p.fp8.requests_1m),
                (k256, m1),
                "{why}"
            );
            assert_rounds_to(mtok(p.bf16.tokens), bf16_m, &why);
            // One request can use the full 1,048,576 tokens with FP8 KV...
            assert_eq!(p.fp8.longest_request, 1_048_576, "{why}");
            // ...while BF16 KV caps it at about 0.6-1.0 M in layouts A, B and D.
            if layout != Layout::C {
                assert!(
                    (550_000..1_000_000).contains(&p.bf16.longest_request),
                    "{why}"
                );
            }
            // With 8 slots instead of 16: about 1.4 GiB and 0.24 M FP8 tokens more.
            let p8 = plan_gpu(t, budget(&s, layout, 8, rt)).unwrap();
            assert_rounds_to(gib(p8.kv_pool - p.kv_pool), "1.4", &why);
            assert_rounds_to(mtok(p8.fp8.tokens - p.fp8.tokens), "0.25", &why);
        }
    }
}

#[test]
fn infeasible_plans_fail() {
    let s = &sources()[0];
    let mut b = budget(s, Layout::A, 16, 4);
    b.device = 16 * GIB;
    let e = plan_gpu(&config().text, b).unwrap_err().to_string();
    assert!(e.contains("the device has 16.00 GiB"), "{e}");
    // Admission never exceeds one request per slot.
    b.device = 80 * GIB;
    b.slots = 2;
    let p = plan_gpu(&config().text, b).unwrap();
    assert_eq!(p.fp8.requests_256k, 2);
}

/// Section 6: the Sparks, per rank at TP4 (layers 3-44; MTP layer 45 apart).
#[test]
fn spark_budget_per_rank() {
    let t = &config().text;
    for s in sources() {
        let why = s.what;
        let exl3 = plan_spark(&s.exl3, t, 4).unwrap();
        let nvfp4 = plan_spark(&s.nvfp4, t, 4).unwrap();
        let fp8 = plan_spark(&s.official, t, 4).unwrap();
        // NVFP4 and FP8 match SIZING. EXL3 does not: SIZING's 38.2 GB (35.5
        // GiB) is a quarter of the checkpoint's expert bytes, but each rank
        // must hold gate/up `suh` and down `svh` whole (see headers.rs).
        assert_eq!(exl3.bytes, 38_385_301_248, "{why}");
        assert_rounds_to(gb(exl3.bytes), "38.4", why);
        assert_rounds_to(gib(exl3.bytes), "35.7", why);
        assert_rounds_to(gb(nvfp4.bytes), "42.8", why);
        assert_rounds_to(gib(nvfp4.bytes), "39.9", why);
        assert_rounds_to(gb(fp8.bytes), "76.1", why);
        assert_rounds_to(gib(fp8.bytes), "70.9", why);
        assert_rounds_to(gb(exl3.mtp_bytes), "0.9", why);
        assert_rounds_to(gb(nvfp4.mtp_bytes), "1.0", why);
        assert_rounds_to(gb(fp8.mtp_bytes), "1.8", why);
        // Read per token at one row: 8 experts in each of 42 layers.
        assert_eq!(exl3.read_per_token, 8 * 42 * exl3.expert_slice);
        assert_rounds_to(gb(exl3.read_per_token), "1.07", why); // SIZING: 1.06 (a quarter)
        assert_rounds_to(gb(nvfp4.read_per_token), "1.19", why);
        assert_rounds_to(gb(fp8.read_per_token), "2.11", why);
        let ms = |b: u64| b as f64 / 230e9 * 1e3;
        assert_rounds_to(ms(exl3.read_per_token), "4.6", why);
        assert_rounds_to(ms(nvfp4.read_per_token), "5.2", why);
        assert_rounds_to(ms(fp8.read_per_token), "9.2", why);
    }
}
