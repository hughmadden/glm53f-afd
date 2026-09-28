//! Boot identity readback (ported from mimo26f-afd v1.2.0
//! `crates/mimo26-spark/src/boot.rs`, whose checks lived in
//! `mimo26-repack::identity`).
//!
//! A rank reads its manifest and the layer images resident on local disk and
//! refuses to serve on any mismatch: a missing or truncated image, a SHA-256
//! that is not the manifest's, an image the manifest does not list, or a
//! manifest for another rank, layout or world size. The check is total: every
//! file is verified and every failure named, so the boot log lists every bad
//! image, not only the first. Files are hashed in parallel.

use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::manifest::{Expect, LayerEntry, Manifest};
use crate::sha256::{hex, Sha256};

/// The boot receipt: the readback summary logged at rank start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootReceipt {
    /// Layer images that matched their size and SHA-256.
    pub matched: usize,
    /// Images checked.
    pub total: usize,
    /// Bytes of the matching images.
    pub matched_bytes: u64,
    /// One-line boot-log summary.
    pub summary: String,
}

/// SHA-256 of a file, streamed in 8 MiB chunks, with its length.
pub fn hash_file(path: &Path) -> Result<(u64, String), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, hex(&h.finalize())))
}

fn verify_one(dir: &Path, e: &LayerEntry) -> Result<(), String> {
    let path = dir.join(&e.file);
    let len = std::fs::metadata(&path).map_err(|_| format!("{}: missing", e.file))?.len();
    if len != e.bytes {
        return Err(format!("{}: {len} bytes, manifest says {}", e.file, e.bytes));
    }
    let (_, sha) = hash_file(&path)?;
    if sha != e.sha256 {
        return Err(format!("{}: sha256 {sha}, manifest says {}", e.file, e.sha256));
    }
    Ok(())
}

/// Verify every resident layer image of `rank` in `dir` against the manifest
/// and refuse on any mismatch. The daemon passes [`Expect::SERVING`].
pub fn readback(dir: &Path, rank: usize, expect: &Expect) -> Result<BootReceipt, String> {
    let manifest = Manifest::read(dir)?;
    manifest.validate(rank, expect)?;
    let entries = &manifest.layers;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8).min(entries.len().max(1));
    let next = AtomicUsize::new(0);
    let mut results: Vec<(usize, Result<(), String>)> = Vec::with_capacity(entries.len());
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= entries.len() {
                            break;
                        }
                        out.push((i, verify_one(dir, &entries[i])));
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            results.extend(h.join().expect("readback thread"));
        }
    });
    results.sort_by_key(|(i, _)| *i);
    let mut failures: Vec<String> = results.iter().filter_map(|(_, r)| r.as_ref().err().cloned()).collect();
    // The other direction: images on disk the manifest does not list.
    let listed: std::collections::BTreeSet<&str> = entries.iter().map(|e| e.file.as_str()).collect();
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for ent in rd.flatten() {
        let name = ent.file_name().to_string_lossy().into_owned();
        if name.ends_with(".exl3") && !listed.contains(name.as_str()) {
            failures.push(format!("{name}: resident but not listed in the manifest"));
        }
    }
    if !failures.is_empty() {
        failures.sort();
        return Err(format!("boot readback: refusing to serve, {} problem(s): {}", failures.len(), failures.join("; ")));
    }
    let matched_bytes: u64 = entries.iter().map(|e| e.bytes).sum();
    Ok(BootReceipt {
        matched: entries.len(),
        total: entries.len(),
        matched_bytes,
        summary: format!(
            "identity readback: {}/{} layer images match ({} B), rank {}, layout {}, source {}",
            entries.len(),
            entries.len(),
            matched_bytes,
            manifest.rank,
            manifest.layout,
            manifest.source
        ),
    })
}
