//! CUDA kernels against the CPU reference on random data (feature `cuda`).
//!
//! Run with `cargo test --release --features cuda --test gpu`. The tests use
//! well under 1 GiB of device memory, take turns on the GPU, and skip when less
//! than 2 GiB is free.
#![cfg(feature = "cuda")]

use std::ptr;
use std::sync::{Mutex, MutexGuard};

use glm53f_dsa::cache::{self, PagedLayer, Tail, PAGE_LAYER_BYTES, PAGE_POOLS, PAGE_TOKENS};
use glm53f_dsa::config::DsaConfig;
use glm53f_dsa::ffi::{self, DsaCache, DsaWindow};
use glm53f_dsa::fp8::ScaleMode;
use glm53f_dsa::gpu::{self, check, DeviceBuffer};
use glm53f_dsa::indexer::{self, pool_key};
use glm53f_dsa::mla::{self, MlaWeights};
use glm53f_dsa::num::{bf16_bits_to_f32, bf16_round, dot, f32_to_bf16_bits, layer_norm, rms_norm, Rounding};
use glm53f_dsa::rng::Rng;
use glm53f_dsa::select::{self, top_k};

const MIN_FREE: usize = 2 << 30;

/// The tests take turns on the GPU: a CUDA graph capture (in
/// `index_select_workspace_reuse`) is invalidated by a device-wide
/// synchronization from another thread.
static GPU: Mutex<()> = Mutex::new(());

/// The GPU, or None (skip) when the shared GPU is short of memory.
fn gpu_ready() -> Option<MutexGuard<'static, ()>> {
    let guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    match gpu::mem_info() {
        Ok((free, _)) if free >= MIN_FREE => {
            gpu::init().expect("init");
            Some(guard)
        }
        Ok((free, _)) => {
            eprintln!("skipping: {} MiB free on the GPU", free >> 20);
            None
        }
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

/// A paged cache on the device for one or more requests.
struct DevCache {
    pages: DeviceBuffer,
    tables: DeviceBuffer,
    max_pages: i32,
    n_pages: i32,
    stride: usize,
}

impl DevCache {
    fn upload(layer: &PagedLayer, tables: &[i32], max_pages: usize) -> Self {
        Self {
            pages: DeviceBuffer::from_slice(&layer.bytes).unwrap(),
            tables: DeviceBuffer::from_slice(tables).unwrap(),
            max_pages: max_pages as i32,
            n_pages: layer.pages as i32,
            stride: layer.page_stride,
        }
    }

    fn view(&self) -> DsaCache {
        DsaCache {
            base: self.pages.as_mut_ptr(),
            page_stride: self.stride as i64,
            page_tables: self.tables.as_ptr(),
            max_pages: self.max_pages,
            n_pages: self.n_pages,
        }
    }

    fn download(&self) -> PagedLayer {
        let bytes = self.pages.download::<u8>(self.pages.bytes()).unwrap();
        PagedLayer { bytes, page_stride: self.stride, pages: self.n_pages as usize }
    }
}

/// A shuffled page table covering `tokens` tokens.
fn shuffled_table(tokens: usize, rng: &mut Rng) -> Vec<i32> {
    let n = tokens.div_ceil(PAGE_TOKENS);
    let mut t: Vec<i32> = (0..n as i32).collect();
    for i in (1..n).rev() {
        t.swap(i, rng.below(i + 1));
    }
    t
}

fn random_latent(rng: &mut Rng) -> Vec<f32> {
    let mut v = rng.normals(512, 1.0);
    // A few large channels, like an RMS-normalized latent with uneven norm weights.
    for c in [7usize, 130, 300, 511] {
        v[c] *= 12.0;
    }
    v
}

#[test]
fn latent_write_is_bit_exact() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let mut rng = Rng::new(1);
    let tokens = 300;
    let table = shuffled_table(tokens, &mut rng);
    let layer = PagedLayer::new(table.len(), PAGE_LAYER_BYTES);
    let dev = DevCache::upload(&layer, &table, table.len());
    let rows = 37;
    let pos: Vec<i32> = (0..rows).map(|_| rng.below(tokens) as i32).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let rows = pos.len();
    let lat: Vec<Vec<f32>> = (0..rows).map(|_| random_latent(&mut rng)).collect();
    let flat: Vec<f32> = lat.iter().flatten().copied().collect();
    let d_lat = DeviceBuffer::from_slice(&flat).unwrap();
    let d_pos = DeviceBuffer::from_slice(&pos).unwrap();
    let d_req = DeviceBuffer::from_slice(&vec![0i32; rows]).unwrap();
    // Without the norm: byte-identical to the CPU encoder.
    check(
        unsafe {
            ffi::glm53f_dsa_mla_latent_write(d_lat.as_ptr(), ptr::null(), 0.0, d_pos.as_ptr(), d_req.as_ptr(), rows as i32, dev.view(), ptr::null_mut())
        },
        "latent_write",
    )
    .unwrap();
    gpu::sync().unwrap();
    let out = dev.download();
    for (i, p) in pos.iter().enumerate() {
        let want = cache::encode_latent(&lat[i], ScaleMode::Pow2);
        assert_eq!(out.read_latent(&table, *p as usize), &want[..], "row {i} token {p}");
    }
    // With the fused RMSNorm: within one code step of the CPU (reduction order differs).
    let norm_w: Vec<f32> = (0..512).map(|_| 1.0 + 0.3 * rng.normal()).collect();
    let d_w = DeviceBuffer::from_slice(&norm_w).unwrap();
    check(
        unsafe {
            ffi::glm53f_dsa_mla_latent_write(d_lat.as_ptr(), d_w.as_ptr(), 1e-5, d_pos.as_ptr(), d_req.as_ptr(), rows as i32, dev.view(), ptr::null_mut())
        },
        "latent_write norm",
    )
    .unwrap();
    gpu::sync().unwrap();
    let out = dev.download();
    let (mut same, mut total) = (0usize, 0usize);
    for (i, p) in pos.iter().enumerate() {
        let normed = rms_norm(&lat[i], &norm_w, 1e-5, Rounding::F32);
        let want = cache::encode_latent(&normed, ScaleMode::Pow2);
        let got = out.read_latent(&table, *p as usize);
        let (dw, dg) = (cache::decode_latent(&want), cache::decode_latent(got));
        for c in 0..512 {
            total += 1;
            same += (want[c] == got[c]) as usize;
            let tol = (dw[c].abs() / 8.0).max(cache::latent_scale(&want, c / 128) * 2f32.powi(-8));
            assert!((dw[c] - dg[c]).abs() <= tol, "row {i} channel {c}: {} vs {}", dw[c], dg[c]);
        }
    }
    eprintln!("latent write with RMSNorm: {same}/{total} codes identical to the CPU");
    assert!(same as f64 >= 0.99 * total as f64);
}

/// Pools completed inside a window (some of whose tokens come from the tail),
/// and the tail rewrite after acceptance.
#[test]
fn pool_write_and_tail_commit_match_cpu() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let cfg = DsaConfig::glm53_flash();
    let mut rng = Rng::new(2);
    let d = 128;
    let ln_w: Vec<f32> = (0..d).map(|_| 1.0 + 0.2 * rng.normal()).collect();
    let ln_b: Vec<f32> = rng.normals(d, 0.1);
    let ape: Vec<f32> = rng.normals(4 * d, 0.5);
    // Three requests: committed lengths 10 (tail 2), 12 (tail 0), 7 (tail 3); windows of 9, 1, 8 rows.
    let starts = [10usize, 12, 7];
    let wins = [9usize, 1, 8];
    let accepted = [5usize, 1, 3];
    let max_pages = 2;
    let mut tables = Vec::new();
    for r in 0..3 {
        tables.extend([2 * r as i32 + 1, 2 * r as i32]);
    }
    let layer = PagedLayer::new(6, PAGE_LAYER_BYTES);
    let dev = DevCache::upload(&layer, &tables, max_pages);
    // Old tails (CPU) and window rows.
    let mut tails = Vec::new();
    let mut tail_vals = Vec::new(); // per request: (k, gate) of the tail tokens (BF16 values)
    for s in starts {
        let mut t = Tail::default();
        for _ in 0..s % 4 {
            t.push(&rng.normals(d, 1.0), &rng.normals(d, 1.0));
        }
        tail_vals.push(t.values());
        tails.extend_from_slice(&t.encode());
    }
    let mut windows = Vec::new();
    let mut row_req = Vec::new();
    let mut k_raw = Vec::new();
    let mut gate = Vec::new();
    let mut first = 0;
    for r in 0..3 {
        windows.push(DsaWindow { first_row: first, rows: wins[r] as i32, start: starts[r] as i32, accepted: accepted[r] as i32 });
        for _ in 0..wins[r] {
            row_req.push(r as i32);
            k_raw.extend(rng.normals(d, 2.0));
            gate.extend(rng.normals(d, 1.0));
        }
        first += wins[r] as i32;
    }
    let rows = row_req.len();
    let dk = DeviceBuffer::from_slice(&k_raw).unwrap();
    let dg = DeviceBuffer::from_slice(&gate).unwrap();
    let dlw = DeviceBuffer::from_slice(&ln_w).unwrap();
    let dlb = DeviceBuffer::from_slice(&ln_b).unwrap();
    let dape = DeviceBuffer::from_slice(&ape).unwrap();
    let dtails = DeviceBuffer::from_slice(&tails).unwrap();
    let dwin = DeviceBuffer::from_slice(&windows).unwrap();
    let dreq = DeviceBuffer::from_slice(&row_req).unwrap();
    check(
        unsafe {
            ffi::glm53f_dsa_index_pool_write(
                dk.as_ptr(), dg.as_ptr(), dlw.as_ptr(), dlb.as_ptr(), 1e-6, dape.as_ptr(), dtails.as_ptr(), dwin.as_ptr(), dreq.as_ptr(), rows as i32, dev.view(), ptr::null_mut(),
            )
        },
        "pool_write",
    )
    .unwrap();
    check(
        unsafe {
            ffi::glm53f_dsa_index_tail_commit(dk.as_ptr(), dg.as_ptr(), dlw.as_ptr(), dlb.as_ptr(), 1e-6, dtails.as_mut_ptr(), dwin.as_ptr(), 3, ptr::null_mut())
        },
        "tail_commit",
    )
    .unwrap();
    gpu::sync().unwrap();
    let out = dev.download();
    let new_tails = dtails.download::<u8>(tails.len()).unwrap();
    let mut row = 0;
    let mut checked = 0;
    for r in 0..3 {
        // Token values (BF16 k after LayerNorm, BF16 gate) for positions start - tail .. start + win.
        let mut toks: Vec<(Vec<f32>, Vec<f32>)> = tail_vals[r].clone();
        for i in 0..wins[r] {
            let kr = &k_raw[(row + i) * d..(row + i + 1) * d];
            let k: Vec<f32> = layer_norm(kr, &ln_w, &ln_b, 1e-6).iter().map(|v| bf16_round(*v)).collect();
            let g: Vec<f32> = gate[(row + i) * d..(row + i + 1) * d].iter().map(|v| bf16_round(*v)).collect();
            toks.push((k, g));
        }
        let base = starts[r] - starts[r] % 4;
        let t_tab = &tables[r * max_pages..(r + 1) * max_pages];
        let end = starts[r] + wins[r];
        for pool in starts[r] / 4..end / 4 {
            let idx = |t: usize| t - base;
            let keys: Vec<&[f32]> = (0..4).map(|j| toks[idx(4 * pool + j)].0.as_slice()).collect();
            let gates: Vec<&[f32]> = (0..4).map(|j| toks[idx(4 * pool + j)].1.as_slice()).collect();
            let want = pool_key(&cfg, &ape, &keys, &gates, Rounding::F32);
            let (codes, sc) = out.read_pool(t_tab, pool);
            let got = cache::decode_index_key(codes, sc);
            let (wc, ws) = cache::encode_index_key(&want, ScaleMode::Pow2);
            let wv = cache::decode_index_key(&wc, ws);
            for c in 0..d {
                let tol = (wv[c].abs() / 8.0).max(ws * 2f32.powi(-8));
                assert!((got[c] - wv[c]).abs() <= tol, "req {r} pool {pool} ch {c}: {} vs {}", got[c], wv[c]);
            }
            checked += 1;
        }
        // Tail after acceptance.
        let new_len = starts[r] + accepted[r];
        let t = Tail::decode(&new_tails[r * cache::TAIL_BYTES..(r + 1) * cache::TAIL_BYTES]).unwrap();
        assert_eq!(t.tokens.len(), new_len % 4, "req {r} tail count");
        for (i, (k, g)) in t.values().iter().enumerate() {
            let pos = new_len - new_len % 4 + i;
            let (wk, wg) = &toks[pos - base];
            for c in 0..d {
                assert!((k[c] - wk[c]).abs() <= wk[c].abs() * 2f32.powi(-7) + 1e-6, "req {r} tail {i} key {c}");
                assert_eq!(g[c], wg[c], "req {r} tail {i} gate {c}");
            }
        }
        row += wins[r];
    }
    assert!(checked >= 4, "pools checked: {checked}");
}

struct IndexCase {
    q: Vec<f32>,
    w: Vec<f32>,
    pos: Vec<i32>,
    keys: Vec<Vec<f32>>, // decoded pooled keys
    layer: PagedLayer,
    table: Vec<i32>,
}

fn build_index_case(rng: &mut Rng, pos: Vec<i32>, integer: bool) -> IndexCase {
    let rows = pos.len();
    let max_pos = *pos.iter().max().unwrap() as usize;
    let n_pools = (max_pos + 1) / 4;
    let table = shuffled_table(max_pos + 1, rng);
    let mut layer = PagedLayer::new(table.len(), PAGE_LAYER_BYTES);
    let mut keys = Vec::with_capacity(n_pools);
    for p in 0..n_pools {
        let k: Vec<f32> = if integer {
            (0..128).map(|_| rng.below(7) as f32 - 3.0).collect()
        } else {
            rng.normals(128, 1.0)
        };
        let (codes, s) = cache::encode_index_key(&k, ScaleMode::Pow2);
        layer.write_pool(&table, p, &codes, s);
        keys.push(cache::decode_index_key(&codes, s));
    }
    let (q, w) = if integer {
        ((0..rows * 32 * 128).map(|_| rng.below(5) as f32 - 2.0).collect(), (0..rows * 32).map(|_| rng.below(7) as f32 - 3.0).collect())
    } else {
        (rng.normals(rows * 32 * 128, 1.0), rng.normals(rows * 32, 0.18))
    };
    IndexCase { q, w, pos, keys, layer, table }
}

struct IndexOut {
    pools: Vec<i32>,
    tokens: Vec<i32>,
    counts: Vec<i32>,
    scores: Vec<f32>,
    max_pools: usize,
}

/// `glm53f_dsa_index_select` and its variants (same signature).
type SelectFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    f32,
    *const i32,
    *const i32,
    i32,
    i32,
    DsaCache,
    i32,
    i32,
    *mut core::ffi::c_void,
    u64,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut f32,
    ffi::CudaStream,
) -> i32;

/// Device inputs, workspace and outputs of one selection case.
struct IndexRun {
    dev: DevCache,
    ws: DeviceBuffer,
    ws_bytes: u64,
    q: DeviceBuffer,
    w: DeviceBuffer,
    pos: DeviceBuffer,
    req: DeviceBuffer,
    pools: DeviceBuffer,
    tokens: DeviceBuffer,
    counts: DeviceBuffer,
    scores: DeviceBuffer,
    rows: usize,
    max_pools: usize,
    chunk_pools: i32,
    chunks: i32,
    score_scale: f32,
}

impl IndexRun {
    /// Uploads the case; the plan is glm53f_dsa_index_plan's unless `chunk_pools` is given.
    fn new(case: &IndexCase, score_scale: f32, chunk_pools: Option<i32>) -> Self {
        let rows = case.pos.len();
        let max_pools = ((*case.pos.iter().max().unwrap() + 1) / 4).max(1) as usize;
        let (sms, _, _) = gpu::device_info().unwrap();
        let (mut cp, mut chunks) = (0i32, 0i32);
        unsafe { ffi::glm53f_dsa_index_plan(rows as i32, max_pools as i32, sms, &mut cp, &mut chunks) };
        if let Some(c) = chunk_pools {
            cp = c;
            chunks = (max_pools as i32 + c - 1) / c;
        }
        let ws_bytes = unsafe { ffi::glm53f_dsa_index_workspace_bytes(rows as i32, chunks) };
        Self {
            dev: DevCache::upload(&case.layer, &case.table, case.table.len()),
            ws: DeviceBuffer::alloc(ws_bytes as usize).unwrap(),
            ws_bytes,
            q: DeviceBuffer::from_slice(&case.q).unwrap(),
            w: DeviceBuffer::from_slice(&case.w).unwrap(),
            pos: DeviceBuffer::from_slice(&case.pos).unwrap(),
            req: DeviceBuffer::from_slice(&vec![0i32; rows]).unwrap(),
            pools: DeviceBuffer::alloc(rows * 512 * 4).unwrap(),
            tokens: DeviceBuffer::alloc(rows * 2051 * 4).unwrap(),
            counts: DeviceBuffer::alloc(rows * 2 * 4).unwrap(),
            scores: DeviceBuffer::alloc(rows * max_pools * 4).unwrap(),
            rows,
            max_pools,
            chunk_pools: cp,
            chunks,
            score_scale,
        }
    }

    /// Fills the workspace with one byte value (and waits: launches may use another stream).
    fn fill_workspace(&self, byte: u8) {
        check(unsafe { ffi::cudaMemset(self.ws.ptr(), byte as i32, self.ws.bytes()) }, "memset").unwrap();
        gpu::sync().unwrap();
    }

    /// Clears the outputs (scores to NaN, the rest to 0x7F bytes).
    fn clear_outputs(&self) {
        self.scores.upload(&vec![f32::NAN; self.rows * self.max_pools]).unwrap();
        for b in [&self.pools, &self.tokens, &self.counts] {
            check(unsafe { ffi::cudaMemset(b.ptr(), 0x7F, b.bytes()) }, "memset").unwrap();
        }
        gpu::sync().unwrap();
    }

    /// Launches `f` on `stream` (no synchronization).
    fn launch(&self, f: SelectFn, stream: ffi::CudaStream) -> Result<(), String> {
        check(
            unsafe {
                f(
                    self.q.as_ptr(), self.w.as_ptr(), self.score_scale, self.pos.as_ptr(), self.req.as_ptr(), self.rows as i32, self.max_pools as i32,
                    self.dev.view(), self.chunk_pools, self.chunks, self.ws.ptr(), self.ws_bytes, self.pools.as_mut_ptr(), self.tokens.as_mut_ptr(),
                    self.counts.as_mut_ptr(), self.scores.as_mut_ptr(), stream,
                )
            },
            "index_select",
        )
    }

    fn outputs(&self) -> IndexOut {
        gpu::sync().unwrap();
        IndexOut {
            pools: self.pools.download(self.rows * 512).unwrap(),
            tokens: self.tokens.download(self.rows * 2051).unwrap(),
            counts: self.counts.download(self.rows * 2).unwrap(),
            scores: self.scores.download(self.rows * self.max_pools).unwrap(),
            max_pools: self.max_pools,
        }
    }
}

/// One call of `f` on a workspace filled with `fill` (0 for index_select_prepared).
fn run_index_with(case: &IndexCase, score_scale: f32, chunk_pools: Option<i32>, f: SelectFn, fill: u8) -> IndexOut {
    let run = IndexRun::new(case, score_scale, chunk_pools);
    run.fill_workspace(fill);
    run.clear_outputs();
    run.launch(f, ptr::null_mut()).unwrap();
    run.outputs()
}

/// `glm53f_dsa_index_select` on a workspace that holds garbage.
fn run_index(case: &IndexCase, score_scale: f32, chunk_pools: Option<i32>) -> IndexOut {
    run_index_with(case, score_scale, chunk_pools, ffi::glm53f_dsa_index_select, 0xA5)
}

fn assert_same_selection(got: &IndexOut, want: &IndexOut, what: &str) {
    assert_eq!(got.counts, want.counts, "{what}: counts");
    assert_eq!(got.pools, want.pools, "{what}: pools");
    assert_eq!(got.tokens, want.tokens, "{what}: tokens");
    let same_scores = got.scores.iter().zip(&want.scores).all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(same_scores, "{what}: scores differ");
}

fn cpu_scores(case: &IndexCase, r: usize, scale: f32) -> Vec<f32> {
    let n = (case.pos[r] as usize + 1) / 4;
    let q = &case.q[r * 4096..(r + 1) * 4096];
    let w = &case.w[r * 32..(r + 1) * 32];
    (0..n)
        .map(|p| {
            let mut s = 0.0f32;
            for h in 0..32 {
                s += w[h] * (dot(&q[h * 128..(h + 1) * 128], &case.keys[p]) * scale).max(0.0);
            }
            s
        })
        .collect()
}

/// Check the GPU's per-row output layout against a selection of `pools` (rank order).
fn check_row_layout(out: &IndexOut, r: usize, pos: usize, want_pools: &[u32]) {
    let n = (pos + 1) / 4;
    let mut asc: Vec<u32> = want_pools.to_vec();
    asc.sort_unstable();
    let got_pools: Vec<i32> = out.pools[r * 512..(r + 1) * 512].to_vec();
    let kept = out.counts[2 * r] as usize;
    assert_eq!(kept, asc.len(), "row {r}: kept pools");
    for i in 0..512 {
        let want = if i < kept { asc[i] as i32 } else { -1 };
        assert_eq!(got_pools[i], want, "row {r} pool slot {i}");
    }
    let sel = select::Selection { position: pos, visible_pools: n, pools: want_pools.to_vec(), tail_start: 4 * n, tail_len: pos + 1 - 4 * n };
    let toks = sel.tokens(4);
    assert_eq!(out.counts[2 * r + 1] as usize, toks.len(), "row {r}: token count");
    for i in 0..2051 {
        let want = toks.get(i).map(|t| *t as i32).unwrap_or(-1);
        assert_eq!(out.tokens[r * 2051 + i], want, "row {r} token slot {i}");
    }
}

/// Integer-valued queries, weights and keys make every product and sum exact,
/// so GPU and CPU scores are bit-identical and the selections, including the
/// many ties, must match exactly. Tiny chunks force three merge levels.
#[test]
fn index_select_exact_on_integer_data() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let mut rng = Rng::new(3);
    let pos = vec![19_999, 2050, 2051, 10_002, 3, 1, 2046, 7_777];
    let case = build_index_case(&mut rng, pos.clone(), true);
    for chunk in [Some(64), Some(640), None] {
        let out = run_index(&case, 1.0, chunk);
        for (r, p) in pos.iter().enumerate() {
            let p = *p as usize;
            let scores = cpu_scores(&case, r, 1.0);
            let want = top_k(&scores, 512);
            check_row_layout(&out, r, p, &want);
            if scores.len() > 512 {
                for (i, s) in scores.iter().enumerate() {
                    assert_eq!(out.scores[r * out.max_pools + i], *s, "row {r} pool {i} score");
                }
                let ties = scores.iter().filter(|s| **s == scores[want[511] as usize]).count();
                eprintln!("chunk {chunk:?} row {r}: {} pools, {ties} tied at the boundary score", scores.len());
            }
        }
    }
}

/// Random data at the real scale: scores within a derived f16 bound of the CPU;
/// the kept set is exactly the top 512 of the GPU's own scores, and differs from
/// the CPU's only at near-ties.
#[test]
fn index_select_random_matches_cpu() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let mut rng = Rng::new(4);
    let scale = (128f32).powf(-0.5);
    let pos = vec![40_003, 2051, 9_000, 1_000];
    let case = build_index_case(&mut rng, pos.clone(), false);
    let out = run_index(&case, scale, None);
    let mut worst = 0.0f64;
    let mut boundary = 0usize;
    for (r, p) in pos.iter().enumerate() {
        let p = *p as usize;
        let scores = cpu_scores(&case, r, scale);
        let q = &case.q[r * 4096..(r + 1) * 4096];
        let w = &case.w[r * 32..(r + 1) * 32];
        let gpu_scores: Vec<f32> = out.scores[r * out.max_pools..r * out.max_pools + scores.len()].to_vec();
        if scores.len() > 512 {
            for (i, s) in scores.iter().enumerate() {
                // q is rounded to f16 on the GPU (relative 2^-11 per element).
                let mut bound = 0.0f64;
                for h in 0..32 {
                    bound += (w[h] as f64).abs() * scale as f64 * glm53f_dsa::num::dot_abs(&q[h * 128..(h + 1) * 128], &case.keys[i]);
                }
                let bound = bound * (2f64.powi(-11) + 300.0 * 2f64.powi(-24)) + 1e-7;
                let d = (gpu_scores[i] as f64 - *s as f64).abs();
                assert!(d <= bound, "row {r} pool {i}: {} vs {s} (bound {bound})", gpu_scores[i]);
                worst = worst.max(d / bound);
            }
            // Exactly the top 512 of the GPU's own scores.
            check_row_layout(&out, r, p, &top_k(&gpu_scores, 512));
            let cpu = top_k(&scores, 512);
            let kth = cpu.iter().map(|x| scores[*x as usize]).fold(f32::INFINITY, f32::min);
            let tol = 4e-3 * (1.0 + kth.abs());
            let gsel: Vec<u32> = top_k(&gpu_scores, 512);
            let bad = select::boundary_mismatches(&cpu, &gsel, &scores, tol);
            assert!(bad.is_empty(), "row {r}: pools {bad:?} differ beyond near-ties");
            boundary += cpu.iter().filter(|x| !gsel.contains(x)).count();
        } else {
            check_row_layout(&out, r, p, &top_k(&scores, 512));
        }
    }
    eprintln!("random index scores: worst |gpu - cpu| / bound = {worst:.3}; near-tie swaps {boundary}");
}

/// The fused kernel (`index_select`, `index_select_prepared`) and the first
/// implementation (`index_select_v1`) return identical scores, selections and
/// token lists on random data, dense and sparse rows alike, for the default
/// plan, many small chunks (three merge levels) and one chunk per row (no
/// merge).
#[test]
fn index_select_variants_agree() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let mut rng = Rng::new(9);
    let scale = (128f32).powf(-0.5);
    let pos = vec![60_001, 2051, 9_000, 1_000, 2_050, 33_333, 4_096, 20_480];
    let case = build_index_case(&mut rng, pos, false);
    for chunk in [None, Some(64), Some(15_040)] {
        let v1 = run_index_with(&case, scale, chunk, ffi::glm53f_dsa_index_select_v1, 0);
        let v2 = run_index_with(&case, scale, chunk, ffi::glm53f_dsa_index_select, 0xA5);
        let v2p = run_index_with(&case, scale, chunk, ffi::glm53f_dsa_index_select_prepared, 0);
        assert_same_selection(&v2, &v1, &format!("index_select vs v1, chunk {chunk:?}"));
        assert_same_selection(&v2p, &v1, &format!("index_select_prepared vs v1, chunk {chunk:?}"));
    }
}

/// One workspace across calls: `index_select` on garbage, repeated;
/// `index_select_prepared` after one zero-fill, repeated and interleaved with
/// `index_select`; both captured in CUDA graphs and replayed. Every call gives
/// the first implementation's result.
#[test]
fn index_select_workspace_reuse() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let mut rng = Rng::new(10);
    let scale = (128f32).powf(-0.5);
    let case = build_index_case(&mut rng, vec![50_001, 7_000, 2_051, 12], false);
    let want = run_index_with(&case, scale, Some(64), ffi::glm53f_dsa_index_select_v1, 0);
    let run = IndexRun::new(&case, scale, Some(64));
    let expect = |what: &str| assert_same_selection(&run.outputs(), &want, what);
    run.fill_workspace(0xFF);
    for i in 0..3 {
        run.clear_outputs();
        run.launch(ffi::glm53f_dsa_index_select, ptr::null_mut()).unwrap();
        expect(&format!("index_select call {i} on garbage"));
    }
    run.fill_workspace(0);
    for (i, f) in [ffi::glm53f_dsa_index_select_prepared, ffi::glm53f_dsa_index_select_prepared, ffi::glm53f_dsa_index_select, ffi::glm53f_dsa_index_select_prepared]
        .into_iter()
        .enumerate()
    {
        run.clear_outputs();
        run.launch(f, ptr::null_mut()).unwrap();
        expect(&format!("interleaved call {i}"));
    }
    let stream = gpu::Stream::new().unwrap();
    for f in [ffi::glm53f_dsa_index_select as SelectFn, ffi::glm53f_dsa_index_select_prepared] {
        run.clear_outputs();
        // 4 calls captured in one graph, replayed 3 times.
        gpu::time_graph_us(&stream, 4, 3, |s| run.launch(f, s)).unwrap();
        expect("graph replay");
    }
}

/// 1M-token context (262,144 pools): the kept set is exactly the top 512 of the
/// GPU's scores, sampled scores match the CPU, and the prepared variant and
/// the first implementation return the same outputs.
#[test]
fn index_select_one_million_tokens() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let mut rng = Rng::new(5);
    let n_pools = 262_144usize;
    let pages = n_pools / PAGE_POOLS;
    // Pooled-key blocks only (the latent part of each page stays zero).
    let mut blocks = vec![0u8; pages * (PAGE_LAYER_BYTES - cache::PAGE_POOL_CODES_OFFSET)];
    let block = PAGE_LAYER_BYTES - cache::PAGE_POOL_CODES_OFFSET;
    let mut scales = vec![0f32; n_pools];
    for p in 0..pages {
        let b = &mut blocks[p * block..(p + 1) * block];
        for i in 0..PAGE_POOLS * 128 {
            let mut c = (rng.next_u64() & 0xFF) as u8;
            if c & 0x7F == 0x7F {
                c &= 0xFE; // no NaN codes
            }
            b[i] = c;
        }
        for i in 0..PAGE_POOLS {
            let s = 2f32.powi(rng.below(6) as i32 - 8);
            scales[p * PAGE_POOLS + i] = s;
            b[PAGE_POOLS * 128 + 4 * i..PAGE_POOLS * 128 + 4 * i + 4].copy_from_slice(&s.to_le_bytes());
        }
    }
    let table: Vec<i32> = (0..pages as i32).collect();
    let d_pages = DeviceBuffer::zeroed(pages * PAGE_LAYER_BYTES).unwrap();
    check(
        unsafe {
            ffi::cudaMemcpy2D(
                (d_pages.ptr() as *mut u8).add(cache::PAGE_POOL_CODES_OFFSET) as *mut _,
                PAGE_LAYER_BYTES,
                blocks.as_ptr() as *const _,
                block,
                block,
                pages,
                ffi::MEMCPY_H2D,
            )
        },
        "memcpy2d",
    )
    .unwrap();
    let d_table = DeviceBuffer::from_slice(&table).unwrap();
    let view = DsaCache { base: d_pages.as_mut_ptr(), page_stride: PAGE_LAYER_BYTES as i64, page_tables: d_table.as_ptr(), max_pages: pages as i32, n_pages: pages as i32 };
    let scale = (128f32).powf(-0.5);
    let (sms, _, _) = gpu::device_info().unwrap();
    for rows in [1usize, 8] {
        let pos: Vec<i32> = (0..rows).map(|r| (4 * n_pools - rows + r) as i32).collect();
        let q = rng.normals(rows * 4096, 1.0);
        let w = rng.normals(rows * 32, 0.18);
        let max_pools = n_pools;
        let (mut cp, mut chunks) = (0i32, 0i32);
        unsafe { ffi::glm53f_dsa_index_plan(rows as i32, max_pools as i32, sms, &mut cp, &mut chunks) };
        let ws_bytes = unsafe { ffi::glm53f_dsa_index_workspace_bytes(rows as i32, chunks) };
        let ws = DeviceBuffer::alloc(ws_bytes as usize).unwrap();
        let dq = DeviceBuffer::from_slice(&q).unwrap();
        let dw = DeviceBuffer::from_slice(&w).unwrap();
        let dpos = DeviceBuffer::from_slice(&pos).unwrap();
        let dreq = DeviceBuffer::from_slice(&vec![0i32; rows]).unwrap();
        let pools = DeviceBuffer::zeroed(rows * 512 * 4).unwrap();
        let tokens = DeviceBuffer::zeroed(rows * 2051 * 4).unwrap();
        let counts = DeviceBuffer::zeroed(rows * 8).unwrap();
        let dbg = DeviceBuffer::alloc(rows * max_pools * 4).unwrap();
        check(unsafe { ffi::cudaMemset(dbg.ptr(), 0x7F, dbg.bytes()) }, "memset").unwrap();
        check(
            unsafe {
                ffi::glm53f_dsa_index_select(
                    dq.as_ptr(), dw.as_ptr(), scale, dpos.as_ptr(), dreq.as_ptr(), rows as i32, max_pools as i32, view, cp, chunks, ws.ptr(), ws_bytes,
                    pools.as_mut_ptr(), tokens.as_mut_ptr(), counts.as_mut_ptr(), dbg.as_mut_ptr(), ptr::null_mut(),
                )
            },
            "index_select",
        )
        .unwrap();
        gpu::sync().unwrap();
        let out = IndexOut {
            pools: pools.download(rows * 512).unwrap(),
            tokens: tokens.download(rows * 2051).unwrap(),
            counts: counts.download(rows * 2).unwrap(),
            scores: dbg.download(rows * max_pools).unwrap(),
            max_pools,
        };
        // The prepared variant and the first implementation agree exactly.
        for (name, f) in [("index_select_prepared", ffi::glm53f_dsa_index_select_prepared as SelectFn), ("index_select_v1", ffi::glm53f_dsa_index_select_v1)] {
            ws.zero().unwrap();
            for b in [&pools, &tokens, &counts, &dbg] {
                check(unsafe { ffi::cudaMemset(b.ptr(), 0x7F, b.bytes()) }, "memset").unwrap();
            }
            check(
                unsafe {
                    f(
                        dq.as_ptr(), dw.as_ptr(), scale, dpos.as_ptr(), dreq.as_ptr(), rows as i32, max_pools as i32, view, cp, chunks, ws.ptr(), ws_bytes,
                        pools.as_mut_ptr(), tokens.as_mut_ptr(), counts.as_mut_ptr(), dbg.as_mut_ptr(), ptr::null_mut(),
                    )
                },
                name,
            )
            .unwrap();
            gpu::sync().unwrap();
            let other = IndexOut {
                pools: pools.download(rows * 512).unwrap(),
                tokens: tokens.download(rows * 2051).unwrap(),
                counts: counts.download(rows * 2).unwrap(),
                scores: dbg.download(rows * max_pools).unwrap(),
                max_pools,
            };
            assert_same_selection(&other, &out, &format!("1M context, {rows} row(s): {name}"));
        }
        for r in 0..rows {
            let p = pos[r] as usize;
            let n = (p + 1) / 4;
            let s = &out.scores[r * max_pools..r * max_pools + n];
            check_row_layout(&out, r, p, &top_k(s, 512));
            // Spot-check scores against the CPU.
            let qr = &q[r * 4096..(r + 1) * 4096];
            for _ in 0..500 {
                let i = rng.below(n);
                let page = i / PAGE_POOLS;
                let codes = &blocks[page * block + (i % PAGE_POOLS) * 128..page * block + (i % PAGE_POOLS) * 128 + 128];
                let key = cache::decode_index_key(codes, scales[i]);
                let mut cs = 0.0f32;
                let mut bound = 0.0f64;
                for h in 0..32 {
                    cs += w[r * 32 + h] * (dot(&qr[h * 128..(h + 1) * 128], &key) * scale).max(0.0);
                    bound += (w[r * 32 + h] as f64).abs() * scale as f64 * glm53f_dsa::num::dot_abs(&qr[h * 128..(h + 1) * 128], &key);
                }
                let bound = bound * (2f64.powi(-11) + 300.0 * 2f64.powi(-24)) + 1e-7;
                assert!(((s[i] - cs) as f64).abs() <= bound, "row {r} pool {i}: {} vs {cs}", s[i]);
            }
        }
        eprintln!("1M context, {rows} row(s): plan {chunks} chunks of {cp} pools; selection exact");
    }
}

/// Decoded latents for a paged layer image.
fn cpu_attention(q_abs: &[f32], latents: &[Vec<f32>], toks: &[u32], scale: f32) -> (Vec<f32>, Vec<f32>) {
    let cfg = DsaConfig::glm53_flash();
    assert_eq!(cfg.attn_scale(), scale);
    let lat: Vec<&[f32]> = toks.iter().map(|t| latents[*t as usize].as_slice()).collect();
    mla::attend_absorbed_latent(&cfg, q_abs, &lat)
}

struct AttnCase {
    layer: PagedLayer,
    table: Vec<i32>,
    latents: Vec<Vec<f32>>, // decoded (what the GPU reads)
}

fn build_attn_case(rng: &mut Rng, tokens: usize) -> AttnCase {
    let table = shuffled_table(tokens, rng);
    let mut layer = PagedLayer::new(table.len(), PAGE_LAYER_BYTES);
    let mut latents = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let rec = cache::encode_latent(&random_latent(rng), ScaleMode::Pow2);
        layer.write_latent(&table, t, &rec);
        latents.push(cache::decode_latent(&rec));
    }
    AttnCase { layer, table, latents }
}

fn random_token_list(rng: &mut Rng, tokens: usize, n: usize) -> Vec<u32> {
    let mut all: Vec<u32> = (0..tokens as u32).collect();
    for i in (1..all.len()).rev() {
        all.swap(i, rng.below(i + 1));
    }
    let mut v: Vec<u32> = all[..n].to_vec();
    v.sort_unstable();
    v
}

/// `glm53f_dsa_mla_sparse_attn` and its first implementation (same signature).
type AttnFn = unsafe extern "C" fn(
    *const u16,
    *const i32,
    i32,
    *const i32,
    *const i32,
    i32,
    f32,
    DsaCache,
    i32,
    i32,
    *mut core::ffi::c_void,
    u64,
    *mut f32,
    *mut f32,
    ffi::CudaStream,
) -> i32;

#[allow(clippy::too_many_arguments)]
fn run_attn(case: &AttnCase, f: AttnFn, q_bf16: &[u16], lists: &[Vec<u32>], splits: i32, groups: i32) -> (Vec<f32>, Vec<f32>) {
    let rows = lists.len();
    let dev = DevCache::upload(&case.layer, &case.table, case.table.len());
    let mut toks = vec![-1i32; rows * 2051];
    let mut counts = vec![0i32; rows * 2];
    for (r, l) in lists.iter().enumerate() {
        for (i, t) in l.iter().enumerate() {
            toks[r * 2051 + i] = *t as i32;
        }
        counts[2 * r + 1] = l.len() as i32;
    }
    let dq = DeviceBuffer::from_slice(q_bf16).unwrap();
    let dt = DeviceBuffer::from_slice(&toks).unwrap();
    let dc = DeviceBuffer::from_slice(&counts).unwrap();
    let dreq = DeviceBuffer::from_slice(&vec![0i32; rows]).unwrap();
    let ws_bytes = unsafe { ffi::glm53f_dsa_mla_workspace_bytes(rows as i32, splits) };
    let ws = DeviceBuffer::alloc(ws_bytes as usize).unwrap();
    let o = DeviceBuffer::zeroed(rows * 64 * 512 * 4).unwrap();
    let lse = DeviceBuffer::zeroed(rows * 64 * 4).unwrap();
    check(
        unsafe {
            f(
                dq.as_ptr(), dt.as_ptr(), 2051, dc.as_ptr(), dreq.as_ptr(), rows as i32, 0.0625, dev.view(), splits, groups, ws.ptr(), ws_bytes,
                o.as_mut_ptr(), lse.as_mut_ptr(), ptr::null_mut(),
            )
        },
        "sparse_attn",
    )
    .unwrap();
    gpu::sync().unwrap();
    (o.download(rows * 64 * 512).unwrap(), lse.download(rows * 64).unwrap())
}

/// Sparse attention against the CPU (decoded FP8 latents, BF16 query) for
/// selections of 2,051, 2,048, 777, 64 and 1 tokens, split and unsplit, with 1,
/// 2 and 4 head groups per block and glm53f_dsa_mla_plan's plan: the current
/// kernel and the first implementation, each within the same bounds, and
/// within those bounds of each other.
#[test]
fn sparse_attention_matches_cpu() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let mut rng = Rng::new(6);
    let tokens = 3000;
    let case = build_attn_case(&mut rng, tokens);
    let lists: Vec<Vec<u32>> = [2051usize, 2048, 777, 64, 1].iter().map(|n| random_token_list(&mut rng, tokens, *n)).collect();
    let rows = lists.len();
    let q: Vec<f32> = rng.normals(rows * 64 * 512, 0.5);
    let q_bf16: Vec<u16> = q.iter().map(|v| f32_to_bf16_bits(*v)).collect();
    let qr: Vec<f32> = q_bf16.iter().map(|b| bf16_bits_to_f32(*b)).collect();
    let want: Vec<(Vec<f32>, Vec<f32>)> = (0..rows).map(|r| cpu_attention(&qr[r * 32768..(r + 1) * 32768], &case.latents, &lists[r], 0.0625)).collect();
    let amax = case.latents.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
    let (sms, _, _) = gpu::device_info().unwrap();
    let (mut plan_splits, mut plan_groups) = (0i32, 0i32);
    unsafe { ffi::glm53f_dsa_mla_plan(rows as i32, 2051, sms, &mut plan_splits, &mut plan_groups) };
    assert!((1..=64).contains(&plan_splits) && [1, 2, 4].contains(&plan_groups), "plan {plan_splits} x {plan_groups}");
    let kernels: [(&str, AttnFn); 2] = [("sparse_attn", ffi::glm53f_dsa_mla_sparse_attn), ("sparse_attn_v1", ffi::glm53f_dsa_mla_sparse_attn_v1)];
    for (splits, groups) in [(1, 1), (1, 2), (1, 4), (4, 1), (8, 2), (33, 1), (plan_splits, plan_groups)] {
        let outs: Vec<(Vec<f32>, Vec<f32>)> = kernels.iter().map(|(_, f)| run_attn(&case, *f, &q_bf16, &lists, splits, groups)).collect();
        for ((name, _), (o, lse)) in kernels.iter().zip(&outs) {
            let mut worst_rel = 0.0f64;
            for r in 0..rows {
                for h in 0..64 {
                    let (w, g) = (&want[r].0[h * 512..(h + 1) * 512], &o[(r * 64 + h) * 512..(r * 64 + h + 1) * 512]);
                    let num: f64 = w.iter().zip(g).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
                    let den: f64 = w.iter().map(|a| (*a as f64).powi(2)).sum();
                    let rel = (num / den.max(1e-30)).sqrt();
                    worst_rel = worst_rel.max(rel);
                    // 16-bit probabilities: relative 2^-9 (BF16, v1) or 2^-11 (F16) per weight.
                    assert!(rel < 4e-3, "{name} splits {splits} groups {groups} row {r} head {h}: rel {rel}");
                    for (a, b) in w.iter().zip(g) {
                        assert!((a - b).abs() <= 4e-3 * amax, "{name} row {r} head {h}: {a} vs {b}");
                    }
                    let (wl, gl) = (want[r].1[h], lse[r * 64 + h]);
                    assert!((wl - gl).abs() <= 1e-4 * (1.0 + wl.abs()), "{name} lse row {r} head {h}: {wl} vs {gl}");
                }
            }
            eprintln!("{name} splits {splits} groups {groups}: worst per-head rel L2 vs CPU {worst_rel:.2e}");
        }
        let (a, b) = (&outs[0].0, &outs[1].0);
        for rh in 0..rows * 64 {
            let (x, y) = (&a[rh * 512..(rh + 1) * 512], &b[rh * 512..(rh + 1) * 512]);
            let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
            let den: f64 = y.iter().map(|q| (*q as f64).powi(2)).sum();
            assert!((num / den.max(1e-30)).sqrt() < 4e-3, "current vs first kernel, row-head {rh}");
        }
    }
}

/// Absorb is bit-exact with the CPU (same sequential order, no FMA); the whole
/// absorbed GPU path (absorb, sparse attention, un-absorb) matches the CPU's
/// expanded reference form on FP8-decoded latents.
#[test]
fn absorbed_gpu_path_matches_expanded_cpu() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let cfg = DsaConfig::glm53_flash();
    let mut rng = Rng::new(7);
    let kv_b: Vec<f32> = rng.normals(64 * 512 * 512, 0.05).into_iter().map(bf16_round).collect();
    let kv_b_bits: Vec<u16> = kv_b.iter().map(|v| f32_to_bf16_bits(*v)).collect();
    let w = MlaWeights { q_a_proj: vec![], q_a_norm: vec![], q_b_proj: vec![], kv_a_proj: vec![], kv_a_norm: vec![], kv_b_proj: kv_b.clone(), o_proj: vec![] };
    let rows = 3;
    let q: Vec<f32> = rng.normals(rows * 64 * 256, 1.0);
    let dq = DeviceBuffer::from_slice(&q).unwrap();
    let dkv = DeviceBuffer::from_slice(&kv_b_bits).unwrap();
    let qa16 = DeviceBuffer::zeroed(rows * 64 * 512 * 2).unwrap();
    let qa32 = DeviceBuffer::zeroed(rows * 64 * 512 * 4).unwrap();
    check(unsafe { ffi::glm53f_dsa_mla_absorb_q(dq.as_ptr(), dkv.as_ptr(), rows as i32, qa16.as_mut_ptr(), qa32.as_mut_ptr(), ptr::null_mut()) }, "absorb").unwrap();
    gpu::sync().unwrap();
    let got32: Vec<f32> = qa32.download(rows * 64 * 512).unwrap();
    for r in 0..rows {
        let want = mla::absorb_q(&cfg, &w, &q[r * 16384..(r + 1) * 16384]);
        assert_eq!(&got32[r * 32768..(r + 1) * 32768], &want[..], "absorb row {r} is bit-exact");
    }
    // Attention over a few tokens, then un-absorb, against the expanded CPU form.
    let tokens = 200;
    let case = build_attn_case(&mut rng, tokens);
    let lists: Vec<Vec<u32>> = [48usize, 17, 1].iter().map(|n| random_token_list(&mut rng, tokens, *n)).collect();
    let dev = DevCache::upload(&case.layer, &case.table, case.table.len());
    let mut toks = vec![-1i32; rows * 2051];
    let mut counts = vec![0i32; rows * 2];
    for (r, l) in lists.iter().enumerate() {
        for (i, t) in l.iter().enumerate() {
            toks[r * 2051 + i] = *t as i32;
        }
        counts[2 * r + 1] = l.len() as i32;
    }
    let dt = DeviceBuffer::from_slice(&toks).unwrap();
    let dc = DeviceBuffer::from_slice(&counts).unwrap();
    let dreq = DeviceBuffer::from_slice(&vec![0i32; rows]).unwrap();
    let o_lat = DeviceBuffer::zeroed(rows * 64 * 512 * 4).unwrap();
    let o = DeviceBuffer::zeroed(rows * 64 * 256 * 4).unwrap();
    check(
        unsafe {
            ffi::glm53f_dsa_mla_sparse_attn(
                qa16.as_ptr(), dt.as_ptr(), 2051, dc.as_ptr(), dreq.as_ptr(), rows as i32, 0.0625, dev.view(), 1, 4, ptr::null_mut(), 0, o_lat.as_mut_ptr(), ptr::null_mut(), ptr::null_mut(),
            )
        },
        "sparse_attn",
    )
    .unwrap();
    check(unsafe { ffi::glm53f_dsa_mla_unabsorb_v(o_lat.as_ptr(), dkv.as_ptr(), rows as i32, o.as_mut_ptr(), ptr::null_mut()) }, "unabsorb").unwrap();
    gpu::sync().unwrap();
    let got: Vec<f32> = o.download(rows * 64 * 256).unwrap();
    for r in 0..rows {
        let (ks, vs): (Vec<_>, Vec<_>) = lists[r].iter().map(|t| mla::expand_kv(&cfg, &w, &case.latents[*t as usize], Rounding::F32)).unzip();
        let e = mla::attend_expanded(&cfg, &q[r * 16384..(r + 1) * 16384], &ks, &vs, Rounding::F32);
        let num: f64 = e.out.iter().zip(&got[r * 16384..(r + 1) * 16384]).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = e.out.iter().map(|a| (*a as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        eprintln!("absorbed GPU path vs expanded CPU, row {r} ({} tokens): rel L2 {rel:.2e}", lists[r].len());
        assert!(rel < 1e-2, "row {r}: rel {rel}");
    }
}

/// Absorb is bit-exact for every rows-per-block size (1, 2, 4 and 8, partial
/// blocks included) and its BF16 output is the f32 result rounded; un-absorb is
/// within the summation-order bound of an f64 reference.
#[test]
fn absorb_and_unabsorb_row_counts() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let cfg = DsaConfig::glm53_flash();
    let mut rng = Rng::new(11);
    let kv_b: Vec<f32> = rng.normals(64 * 512 * 512, 0.05).into_iter().map(bf16_round).collect();
    let kv_b_bits: Vec<u16> = kv_b.iter().map(|v| f32_to_bf16_bits(*v)).collect();
    let w = MlaWeights { q_a_proj: vec![], q_a_norm: vec![], q_b_proj: vec![], kv_a_proj: vec![], kv_a_norm: vec![], kv_b_proj: kv_b.clone(), o_proj: vec![] };
    let dkv = DeviceBuffer::from_slice(&kv_b_bits).unwrap();
    let mut worst = 0.0f64;
    for rows in [1usize, 2, 3, 4, 5, 8, 9, 17] {
        let q = rng.normals(rows * 64 * 256, 1.0);
        let o_lat = rng.normals(rows * 64 * 512, 1.0);
        let dq = DeviceBuffer::from_slice(&q).unwrap();
        let qa16 = DeviceBuffer::zeroed(rows * 64 * 512 * 2).unwrap();
        let qa32 = DeviceBuffer::zeroed(rows * 64 * 512 * 4).unwrap();
        check(unsafe { ffi::glm53f_dsa_mla_absorb_q(dq.as_ptr(), dkv.as_ptr(), rows as i32, qa16.as_mut_ptr(), qa32.as_mut_ptr(), ptr::null_mut()) }, "absorb").unwrap();
        let dol = DeviceBuffer::from_slice(&o_lat).unwrap();
        let o = DeviceBuffer::zeroed(rows * 64 * 256 * 4).unwrap();
        check(unsafe { ffi::glm53f_dsa_mla_unabsorb_v(dol.as_ptr(), dkv.as_ptr(), rows as i32, o.as_mut_ptr(), ptr::null_mut()) }, "unabsorb").unwrap();
        gpu::sync().unwrap();
        let got32: Vec<f32> = qa32.download(rows * 64 * 512).unwrap();
        let got16: Vec<u16> = qa16.download(rows * 64 * 512).unwrap();
        let got_o: Vec<f32> = o.download(rows * 64 * 256).unwrap();
        for r in 0..rows {
            let want = mla::absorb_q(&cfg, &w, &q[r * 16384..(r + 1) * 16384]);
            assert_eq!(&got32[r * 32768..(r + 1) * 32768], &want[..], "{rows} rows: absorb row {r} is bit-exact");
            for (i, v) in want.iter().enumerate() {
                assert_eq!(got16[r * 32768 + i], f32_to_bf16_bits(*v), "{rows} rows: BF16 absorb row {r} element {i}");
            }
            for h in 0..64 {
                let ol = &o_lat[(r * 64 + h) * 512..(r * 64 + h + 1) * 512];
                for v in 0..256 {
                    let wv = w.wv_row(&cfg, h, v);
                    let exact: f64 = wv.iter().zip(ol).map(|(a, b)| *a as f64 * *b as f64).sum();
                    let mag: f64 = wv.iter().zip(ol).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
                    let d = (got_o[(r * 64 + h) * 256 + v] as f64 - exact).abs();
                    let bound = glm53f_dsa::num::gamma(512) * mag;
                    assert!(d <= bound, "{rows} rows: un-absorb row {r} head {h} value {v}: off by {d} (bound {bound})");
                    worst = worst.max(d / bound);
                }
            }
        }
    }
    eprintln!("un-absorb: worst error / (gamma_512 sum |w o|) = {worst:.3}");
}

#[test]
fn dense_rows_and_no_pools() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    let mut rng = Rng::new(8);
    // Every row dense: the kernels skip scoring and keep all visible pools.
    let pos = vec![0, 1, 2, 3, 4, 7, 100, 2047, 2050];
    let case = build_index_case(&mut rng, pos.clone(), false);
    let out = run_index(&case, 0.088, None);
    for (r, p) in pos.iter().enumerate() {
        let n = (*p as usize + 1) / 4;
        check_row_layout(&out, r, *p as usize, &(0..n as u32).collect::<Vec<_>>());
        let toks: Vec<i32> = out.tokens[r * 2051..r * 2051 + (*p as usize + 1)].to_vec();
        assert_eq!(toks, (0..=*p).collect::<Vec<i32>>());
    }
    let _ = indexer::KeyFormat::F32;
}

mod common;

/// Layer 3 on the GPU against the oracle fixtures: FP8 latent records written
/// from the fixture latents, the query absorbed with the checkpoint's
/// `kv_b_proj`, attention over the fixture's selections, un-absorbed, and
/// compared with the fixture's per-head output (f32, unquantized latents).
///
/// Needs `GLM53F_CHECKPOINT` and `--release`. Without `layer03-prefill`
/// fixtures, a stand-in set is written from the CPU reference on the real
/// weights and proxy inputs (embedding rows through layer 3's input norm).
#[test]
fn layer3_gpu_against_fixtures() {
    let Some(_gpu) = gpu_ready() else {
        return;
    };
    if cfg!(debug_assertions) {
        eprintln!("skipping: needs --release");
        return;
    }
    let Some(ck) = glm53f_dsa::weights::checkpoint_from_env() else {
        eprintln!("skipping: GLM53F_CHECKPOINT not set");
        return;
    };
    let cfg = DsaConfig::glm53_flash();
    let w = glm53f_dsa::weights::load_dsa_layer(&ck, &cfg, 3).expect("layer 3 weights");
    let found = glm53f_dsa::golden::GoldenSet::discover()
        .into_iter()
        .find(|s| s.dir.file_name().map(|n| n.to_string_lossy().contains("layer03-prefill")).unwrap_or(false));
    let standin_dir = std::env::temp_dir().join(format!("glm53f-dsa-standin-{}", std::process::id()));
    let (set, label) = match found {
        Some(s) => (s, "oracle fixtures"),
        None => {
            // 33 proxy tokens, as the oracle's prompt length.
            let emb = ["model.language_model.embed_tokens.weight", "model.embed_tokens.weight"].into_iter().find(|n| ck.find(n).is_some()).unwrap();
            let norm = ["model.language_model.layers.3.input_layernorm.weight", "model.layers.3.input_layernorm.weight"]
                .into_iter()
                .find(|n| ck.find(n).is_some())
                .unwrap();
            let (w_in, _) = ck.read_f32(norm).unwrap();
            let mut rng = Rng::new(33);
            let x: Vec<Vec<f32>> = (0..33)
                .map(|_| rms_norm(&ck.read_rows(emb, 1000 + rng.below(140_000), 1).unwrap(), &w_in, 1e-5, Rounding::F32))
                .collect();
            let mut opts = glm53f_dsa::layer::LayerOptions::f32_absorbed();
            opts.form = glm53f_dsa::layer::MlaForm::Expanded;
            let mut st = glm53f_dsa::layer::DsaState::new(opts);
            let tr = st.forward(&cfg, &w, &x);
            let mut wr = common::Writer::new(&standin_dir.join("layer03-prefill"));
            wr.add_f32("prefill.attn_norm", &[33, cfg.hidden], &x.concat());
            common::write_phase(&mut wr, &cfg, "prefill.", &tr, &st, true);
            wr.close();
            (glm53f_dsa::golden::GoldenSet::open(&standin_dir.join("layer03-prefill")).unwrap(), "stand-in (CPU reference, proxy inputs)")
        }
    };
    let load = |n: &str| set.load_f32(set.get(n).unwrap_or_else(|| panic!("{n} missing"))).unwrap();
    let latent = load("prefill.mla.latent");
    let q = load("prefill.mla.q");
    let topk = load("prefill.idx.topk");
    let gold = load("prefill.mla.out");
    let t = latent.len() / 512;
    // Cache: token i at position i.
    let table: Vec<i32> = (0..t.div_ceil(PAGE_TOKENS) as i32).collect();
    let layer = PagedLayer::new(table.len(), PAGE_LAYER_BYTES);
    let dev = DevCache::upload(&layer, &table, table.len());
    let pos: Vec<i32> = (0..t as i32).collect();
    let d_lat = DeviceBuffer::from_slice(&latent).unwrap();
    let d_pos = DeviceBuffer::from_slice(&pos).unwrap();
    let d_req = DeviceBuffer::from_slice(&vec![0i32; t]).unwrap();
    check(unsafe { ffi::glm53f_dsa_mla_latent_write(d_lat.as_ptr(), ptr::null(), 0.0, d_pos.as_ptr(), d_req.as_ptr(), t as i32, dev.view(), ptr::null_mut()) }, "latent_write").unwrap();
    let kv_b: Vec<u16> = w.mla.kv_b_proj.iter().map(|v| f32_to_bf16_bits(*v)).collect(); // exact: the checkpoint stores BF16
    let d_kvb = DeviceBuffer::from_slice(&kv_b).unwrap();
    let d_q = DeviceBuffer::from_slice(&q).unwrap();
    let qa = DeviceBuffer::zeroed(t * 64 * 512 * 2).unwrap();
    check(unsafe { ffi::glm53f_dsa_mla_absorb_q(d_q.as_ptr(), d_kvb.as_ptr(), t as i32, qa.as_mut_ptr(), ptr::null_mut(), ptr::null_mut()) }, "absorb").unwrap();
    let mut toks = vec![-1i32; t * 2051];
    let mut counts = vec![0i32; t * 2];
    for r in 0..t {
        let mut v: Vec<i32> = topk[r * 2051..(r + 1) * 2051].iter().filter(|x| **x >= 0.0).map(|x| *x as i32).collect();
        v.sort_unstable();
        toks[r * 2051..r * 2051 + v.len()].copy_from_slice(&v);
        counts[2 * r + 1] = v.len() as i32;
    }
    let dt = DeviceBuffer::from_slice(&toks).unwrap();
    let dc = DeviceBuffer::from_slice(&counts).unwrap();
    let o_lat = DeviceBuffer::zeroed(t * 64 * 512 * 4).unwrap();
    let o = DeviceBuffer::zeroed(t * 64 * 256 * 4).unwrap();
    check(
        unsafe {
            ffi::glm53f_dsa_mla_sparse_attn(qa.as_ptr(), dt.as_ptr(), 2051, dc.as_ptr(), d_req.as_ptr(), t as i32, 0.0625, dev.view(), 1, 4, ptr::null_mut(), 0, o_lat.as_mut_ptr(), ptr::null_mut(), ptr::null_mut())
        },
        "sparse_attn",
    )
    .unwrap();
    check(unsafe { ffi::glm53f_dsa_mla_unabsorb_v(o_lat.as_ptr(), d_kvb.as_ptr(), t as i32, o.as_mut_ptr(), ptr::null_mut()) }, "unabsorb").unwrap();
    gpu::sync().unwrap();
    let got: Vec<f32> = o.download(t * 64 * 256).unwrap();
    // CPU on the same FP8-decoded latents (f32 query, f32 probabilities).
    let pages = dev.download();
    let dec: Vec<Vec<f32>> = (0..t).map(|i| cache::decode_latent(pages.read_latent(&table, i))).collect();
    let (mut worst_gold, mut worst_cpu) = (0.0f64, 0.0f64);
    for r in 0..t {
        let sel: Vec<&[f32]> = toks[r * 2051..r * 2051 + counts[2 * r + 1] as usize].iter().map(|x| dec[*x as usize].as_slice()).collect();
        let a = mla::attend_absorbed(&cfg, &w.mla, &q[r * 16384..(r + 1) * 16384], &sel);
        let g = &got[r * 16384..(r + 1) * 16384];
        let rel = |x: &[f32], y: &[f32]| -> f64 {
            let n: f64 = x.iter().zip(y).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let d: f64 = x.iter().map(|a| (*a as f64).powi(2)).sum();
            (n / d.max(1e-300)).sqrt()
        };
        worst_gold = worst_gold.max(rel(&gold[r * 16384..(r + 1) * 16384], g));
        worst_cpu = worst_cpu.max(rel(&a.out, g));
    }
    eprintln!("layer 3 on the GPU ({label}, {t} rows): worst row rel L2 vs f32 fixture {worst_gold:.2e} (FP8 latents), vs CPU on the same FP8 latents {worst_cpu:.2e}");
    assert!(worst_cpu < 5e-3, "GPU vs CPU on FP8 latents: {worst_cpu}");
    assert!(worst_gold < 6e-2, "GPU vs fixture: {worst_gold}");
    let _ = std::fs::remove_dir_all(&standin_dir);
}
