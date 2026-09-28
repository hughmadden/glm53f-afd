//! The rank's resident weights: one image per MoE layer (layout
//! `glm53f-exl3-k4-tp4-e1`, `layout.rs`) under a manifest, read on demand
//! and handed to the kernel's `prepare_layer` (ported from mimo26f-afd v1.2.0
//! `crates/mimo26-spark/src/resident.rs`, which held one file per expert
//! slice). The boot readback has already verified every image from disk.
//!
//! [`write_rank_dir`] cuts a rank directory from an EXL3 checkpoint.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glm53f_model::catalog::{Catalog, Coverage};
use glm53f_model::config::ModelConfig;
use glm53f_model::safetensors::Checkpoint;

use crate::manifest::{file_name, Expect, LayerEntry, Manifest};

/// One rank's layer images, by layer id.
#[derive(Debug)]
pub struct Resident {
    pub dir: PathBuf,
    pub rank: usize,
    /// Layer id -> (file name, bytes).
    pub files: BTreeMap<u32, (String, u64)>,
}

impl Resident {
    /// Index the manifest (no image is read here).
    pub fn load_manifest(dir: &Path, rank: usize, expect: &Expect) -> Result<Self, String> {
        let m = Manifest::read(dir)?;
        m.validate(rank, expect)?;
        let files = m.layers.iter().map(|l| (l.layer, (l.file.clone(), l.bytes))).collect();
        Ok(Self { dir: dir.to_path_buf(), rank, files })
    }

    /// Layer ids held, ascending.
    pub fn layers(&self) -> Vec<u32> {
        self.files.keys().copied().collect()
    }

    /// One layer's image, read from disk.
    pub fn layer_image(&self, layer: u32) -> Result<Vec<u8>, String> {
        let (file, bytes) = self.files.get(&layer).ok_or_else(|| format!("layer {layer} is not resident on rank {}", self.rank))?;
        let path = self.dir.join(file);
        let image = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if image.len() as u64 != *bytes {
            return Err(format!("{}: {} bytes, manifest says {bytes}", path.display(), image.len()));
        }
        Ok(image)
    }

    /// Give back the page cache's copy of `layer`'s image ([`crate::pagecache`]). The rank calls
    /// it once the image is on the device; the file is not changed and can be read again.
    pub fn drop_cache(&self, layer: u32) -> Result<(), String> {
        let (file, _) = self.files.get(&layer).ok_or_else(|| format!("layer {layer} is not resident on rank {}", self.rank))?;
        crate::pagecache::drop_file(&self.dir.join(file))
    }
}

/// Cut rank `rank`'s share of `layers` from the EXL3 checkpoint in
/// `checkpoint` into `out` (one image per layer and a manifest). `source` is
/// recorded in the manifest (for example `repo@revision`). Images are written
/// to a `.part` name and renamed once complete; the manifest is written last.
pub fn write_rank_dir(checkpoint: &Path, rank: usize, layers: &[u32], out: &Path, source: &str) -> Result<Manifest, String> {
    let cfg = ModelConfig::load(&checkpoint.join("config.json")).map_err(|e| e.to_string())?;
    let cp = Checkpoint::open(checkpoint).map_err(|e| e.to_string())?;
    let cat = Catalog::from_shards(&cfg, &cp.shards, None, Coverage::Subset).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut manifest = Manifest::new(rank, source);
    for &layer in layers {
        let t = std::time::Instant::now();
        let image = crate::layout::slice_checkpoint_layer(&cp, &cat, layer, rank)?;
        let sha = crate::sha256::hex(&crate::sha256::sha256(&image));
        let name = file_name(layer, rank);
        let tmp = out.join(format!("{name}.part"));
        std::fs::write(&tmp, &image).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, out.join(&name)).map_err(|e| format!("{name}: {e}"))?;
        eprintln!("rank {rank} layer {layer}: {} bytes, sha256 {sha} ({:.1} s)", image.len(), t.elapsed().as_secs_f64());
        manifest.layers.push(LayerEntry { layer, file: name, bytes: image.len() as u64, sha256: sha });
    }
    manifest.write(out)?;
    Ok(manifest)
}
