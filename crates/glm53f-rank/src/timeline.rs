//! Cross-host critical-path timeline.
//!
//! Emits `TL <event> [layer=<l>] us=<us>` lines under `GLM53F_TIMELINE=1`, using
//! CLOCK_REALTIME in microseconds. The hosts are not tightly clock-synced, so the
//! coordinator and the Spark each emit from their own clock and the
//! cross-host correlation uses the offset-free four-timestamp method (T1/T4 on the
//! coordinator, T2/T3 on the Spark), with the per-host offsets noted at capture.

/// Wall-clock microsecond timestamp (CLOCK_REALTIME).
pub fn realtime_us() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0)
}

/// Emit one timeline line under `GLM53F_TIMELINE=1` (read once).
pub fn tl(event: &str, layer: Option<u32>) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("GLM53F_TIMELINE").is_some()) {
        let us = realtime_us();
        match layer {
            Some(l) => eprintln!("TL {event} layer={l} us={us}"),
            None => eprintln!("TL {event} us={us}"),
        }
    }
}

/// Per-request stage lines (`ffn …`, `timing …`) under `GLM53F_RANK_TRACE=1`
/// (read once). Off, the serving loop writes nothing per request: at decode a
/// rank serves ~100 requests per step, and four unbuffered lines each cost it
/// about 2% of the step (the log grew ~0.5 GB a day).
pub fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GLM53F_RANK_TRACE").map(|v| v == "1").unwrap_or(false))
}

/// Served-request totals, printed once a minute (the default replacement for
/// the per-request lines).
#[derive(Default)]
pub struct Window {
    start: Option<std::time::Instant>,
    requests: u64,
    rows: u64,
    ffn_ms: f64,
}

impl Window {
    pub fn add(&mut self, rows: usize, ffn_ms: f64) {
        let start = *self.start.get_or_insert_with(std::time::Instant::now);
        self.requests += 1;
        self.rows += rows as u64;
        self.ffn_ms += ffn_ms;
        let secs = start.elapsed().as_secs_f64();
        if secs >= 60.0 {
            eprintln!("stats window={secs:.0}s requests={} rows={} ffn_ms={:.1} (mean {:.3} ms, {:.0}% busy)",
                self.requests, self.rows, self.ffn_ms, self.ffn_ms / self.requests as f64, self.ffn_ms / secs / 10.0);
            *self = Window::default();
        }
    }
}
