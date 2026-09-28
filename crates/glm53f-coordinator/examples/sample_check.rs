//! Served sampling kernel check (mimo26f-afd perf reset V3): the GPU sampler (`gpu::Sampler`:
//! argmax, then `glm53f_coord_sample_rows`) against the CPU reference `sampling::select_pick` on
//! synthetic logit rows at GLM-5.3-Flash's width (154,880 LM head rows, 154,856 token ids), and
//! the selection's time at serving shapes.
//!
//!   sample_check [rows (default 2048)]
//!
//! Rows: realistic (a few peaks over a noisy floor), flat (a huge nucleus), tied (logits in steps of
//! 0.5), damaged (NaN and -inf entries), one-hot, each with a strong peak in the padding rows that
//! must never be drawn. Parameters: random temperature / top_p / top_k / min_p per row.
//!
//! PASS: every draw equals the reference's and lies below 154,856. The two differ only in `exp`'s
//! last bit, which can move a draw only when it lands within ~1e-7 of a boundary.

use glm53f_coordinator::gpu::select_rows_host;
use glm53f_coordinator::model::Pick;
use glm53f_coordinator::sampling::{select_pick, Sampling};

const LD: usize = 154_880;
const VOCAB: usize = 154_856;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.f().max(1e-7), self.f());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next() % xs.len() as u64) as usize]
    }
}

fn logits(rng: &mut Rng, kind: usize) -> Vec<f32> {
    let mut l: Vec<f32> = match kind {
        1 => (0..LD).map(|_| 0.1 * rng.normal()).collect(),
        2 => (0..LD).map(|_| (2.0 * (3.0 * rng.normal() - 4.0)).round() * 0.5).collect(),
        4 => vec![-40.0; LD],
        _ => (0..LD).map(|_| 3.0 * rng.normal() - 5.0).collect(),
    };
    let peaks = 1 + (rng.next() % 20) as usize;
    for _ in 0..peaks {
        let i = (rng.next() % VOCAB as u64) as usize;
        l[i] = if kind == 4 { 40.0 } else { 10.0 + 15.0 * rng.f() };
        if kind == 2 {
            l[i] = l[i].round();
        }
    }
    if kind == 3 {
        for _ in 0..50 {
            let i = (rng.next() % VOCAB as u64) as usize;
            l[i] = if rng.next() % 2 == 0 { f32::NAN } else { f32::NEG_INFINITY };
        }
    }
    // A peak in the padding rows: the argmax may take it, a draw never.
    l[VOCAB + (rng.next() % (LD - VOCAB) as u64) as usize] = 60.0;
    l
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(2048);
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    let (mut rows_done, mut bad, mut kernel_ms, mut chunks) = (0usize, 0usize, 0f64, 0usize);
    while rows_done < n {
        let m = (n - rows_done).min(128);
        let mut x = Vec::with_capacity(m * LD);
        let mut picks = Vec::with_capacity(m);
        for r in 0..m {
            x.extend(logits(&mut rng, r % 5));
            let s = Sampling::new(rng.pick(&[0.2, 0.6, 1.0, 1.4, 2.0]), rng.pick(&[1.0, 0.95, 0.9, 0.5, 0.05]),
                rng.pick(&[0usize, 0, 5, 50, 400]), rng.pick(&[0.0, 0.0, 0.02, 0.3]), Some(rng.next()))
                .expect("valid")
                .expect("sampled");
            picks.push(Pick::at(Some(s), rng.next() % 4096));
        }
        let (got, ms) = select_rows_host(&x, LD, VOCAB, &picks).expect("kernel");
        kernel_ms += ms;
        chunks += 1;
        for (r, p) in picks.iter().enumerate() {
            let row = &x[r * LD..(r + 1) * LD];
            let want = select_pick(row, VOCAB, p);
            let g = got[r];
            if want != g || g as usize >= VOCAB {
                bad += 1;
                let (s, pos) = p.draw.expect("sampled");
                eprintln!("row {}: kind {} T {:.2} top_p {} top_k {} min_p {} position {pos}: gpu {g} cpu {want}",
                    rows_done + r, r % 5, s.temperature, s.top_p, s.top_k, s.min_p);
            }
        }
        rows_done += m;
    }
    println!("{n} rows: {bad} mismatches; selection {:.3} ms per 128-row batch", kernel_ms / chunks as f64);
    // Serving shapes: a C1 speculative step (8 rows) and C16 (128 rows), T 0.7 / top_p 0.95.
    for m in [1usize, 8, 128] {
        let mut x = Vec::with_capacity(m * LD);
        let s = Sampling::new(0.7, 0.95, 0, 0.0, Some(7)).unwrap();
        let mut picks = Vec::with_capacity(m);
        for r in 0..m {
            x.extend(logits(&mut rng, 0));
            picks.push(Pick::at(s, r as u64));
        }
        let mut best = f64::MAX;
        for _ in 0..5 {
            best = best.min(select_rows_host(&x, LD, VOCAB, &picks).expect("kernel").1);
        }
        println!("  {m:3} rows, T 0.7 top_p 0.95: {best:.3} ms");
    }
    println!("RESULT: {}", if bad == 0 { "PASS" } else { "FAIL" });
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
