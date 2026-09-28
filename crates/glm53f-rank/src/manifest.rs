//! The rank directory's manifest: which layer images a rank holds, with their
//! sizes and SHA-256 digests (after mimo26f-afd's slice manifest, simplified to
//! one image per layer).
//!
//! A plain text file, `manifest.txt`, one record per line:
//!
//! ```text
//! glm53f-rank-manifest 1
//! layout glm53f-exl3-k4-tp4-e1
//! rank 0
//! world 4
//! source <free text: the checkpoint it was cut from>
//! layer 3 L03.r0.exl3 913932288 <sha256, 64 hex>
//! ...
//! layer 44 L44.r0.exl3 913932288 <sha256>
//! ```
//!
//! Every decoder MoE layer (3..=44) must be listed once; layer 45 (the MTP
//! layer's experts) is optional.

use std::path::Path;

use crate::consts::{is_moe_layer, FIRST_MOE_LAYER, LAST_MOE_LAYER, WORLD};
use crate::layout::{LAYER_BYTES, LAYOUT};

/// Manifest file name inside a rank directory.
pub const MANIFEST: &str = "manifest.txt";

/// What a rank directory must hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Expect {
    /// Bytes of every layer image.
    pub layer_bytes: u64,
    /// Whether every decoder MoE layer (3..=44) must be present.
    pub all_layers: bool,
}

impl Expect {
    /// A servable directory: every decoder layer, whole images.
    pub const SERVING: Expect = Expect { layer_bytes: LAYER_BYTES as u64, all_layers: true };
}
const MAGIC: &str = "glm53f-rank-manifest 1";

/// One layer image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerEntry {
    pub layer: u32,
    pub file: String,
    pub bytes: u64,
    /// Lowercase hex SHA-256 of the file.
    pub sha256: String,
}

/// A rank directory's manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub layout: String,
    pub rank: usize,
    pub world: usize,
    pub source: String,
    pub layers: Vec<LayerEntry>,
}

/// The canonical file name of `layer`'s image for `rank`: `L03.r0.exl3`.
pub fn file_name(layer: u32, rank: usize) -> String {
    format!("L{layer:02}.r{rank}.exl3")
}

impl Manifest {
    pub fn new(rank: usize, source: &str) -> Self {
        Self { layout: LAYOUT.into(), rank, world: WORLD, source: source.into(), layers: Vec::new() }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        if lines.next() != Some(MAGIC) {
            return Err(format!("manifest: first line is not {MAGIC:?}"));
        }
        let (mut layout, mut rank, mut world, mut source) = (None, None, None, None);
        let mut layers = Vec::new();
        for (n, line) in lines.enumerate() {
            let bad = |m: &str| format!("manifest line {}: {m}: {line:?}", n + 2);
            let (key, rest) = line.split_once(' ').ok_or_else(|| bad("no value"))?;
            match key {
                "layout" => layout = Some(rest.to_string()),
                "rank" => rank = Some(rest.parse::<usize>().map_err(|_| bad("bad rank"))?),
                "world" => world = Some(rest.parse::<usize>().map_err(|_| bad("bad world"))?),
                "source" => source = Some(rest.to_string()),
                "layer" => {
                    let f: Vec<&str> = rest.split(' ').collect();
                    if f.len() != 4 {
                        return Err(bad("want: layer <id> <file> <bytes> <sha256>"));
                    }
                    crate::sha256::unhex(f[3]).map_err(|e| bad(&e))?;
                    layers.push(LayerEntry {
                        layer: f[0].parse().map_err(|_| bad("bad layer id"))?,
                        file: f[1].to_string(),
                        bytes: f[2].parse().map_err(|_| bad("bad byte count"))?,
                        sha256: f[3].to_ascii_lowercase(),
                    });
                }
                _ => return Err(bad("unknown key")),
            }
        }
        Ok(Self {
            layout: layout.ok_or("manifest: no layout")?,
            rank: rank.ok_or("manifest: no rank")?,
            world: world.ok_or("manifest: no world")?,
            source: source.unwrap_or_default(),
            layers,
        })
    }

    pub fn to_text(&self) -> String {
        let mut s = format!(
            "{MAGIC}\nlayout {}\nrank {}\nworld {}\nsource {}\n",
            self.layout,
            self.rank,
            self.world,
            self.source.replace('\n', " ")
        );
        for l in &self.layers {
            s.push_str(&format!("layer {} {} {} {}\n", l.layer, l.file, l.bytes, l.sha256));
        }
        s
    }

    pub fn read(dir: &Path) -> Result<Self, String> {
        let p = dir.join(MANIFEST);
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        Self::parse(&text)
    }

    pub fn write(&self, dir: &Path) -> Result<(), String> {
        let p = dir.join(MANIFEST);
        let tmp = dir.join(format!("{MANIFEST}.part"));
        std::fs::write(&tmp, self.to_text()).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &p).map_err(|e| format!("{}: {e}", p.display()))
    }

    /// The manifest describes a servable directory for `rank`: this layout and
    /// world, every decoder MoE layer once (MTP optional), canonical names,
    /// whole layer images ([`Expect::SERVING`]; tests use smaller images).
    pub fn validate(&self, rank: usize, expect: &Expect) -> Result<(), String> {
        if self.layout != LAYOUT {
            return Err(format!("manifest: layout {:?}, this rank serves {LAYOUT:?}", self.layout));
        }
        if self.world != WORLD {
            return Err(format!("manifest: world {}, this rank serves TP{WORLD}", self.world));
        }
        if self.rank != rank {
            return Err(format!("manifest: holds rank {}'s share, this is rank {rank}", self.rank));
        }
        let mut seen = std::collections::BTreeSet::new();
        for l in &self.layers {
            if !is_moe_layer(l.layer, true) {
                return Err(format!("manifest: layer {} is not a MoE layer", l.layer));
            }
            if !seen.insert(l.layer) {
                return Err(format!("manifest: layer {} listed twice", l.layer));
            }
            if l.file != file_name(l.layer, rank) {
                return Err(format!("manifest: layer {} file {:?}, want {:?}", l.layer, l.file, file_name(l.layer, rank)));
            }
            if l.bytes != expect.layer_bytes {
                return Err(format!("manifest: layer {} is {} bytes, want {}", l.layer, l.bytes, expect.layer_bytes));
            }
        }
        let missing: Vec<u32> = (FIRST_MOE_LAYER..=LAST_MOE_LAYER).filter(|l| !seen.contains(l)).collect();
        if expect.all_layers && !missing.is_empty() {
            return Err(format!("manifest: missing layers {missing:?}"));
        }
        Ok(())
    }

    pub fn entry(&self, layer: u32) -> Option<&LayerEntry> {
        self.layers.iter().find(|l| l.layer == layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full(rank: usize) -> Manifest {
        let mut m = Manifest::new(rank, "test checkpoint");
        for layer in FIRST_MOE_LAYER..=LAST_MOE_LAYER {
            m.layers.push(LayerEntry {
                layer,
                file: file_name(layer, rank),
                bytes: LAYER_BYTES as u64,
                sha256: "ab".repeat(32),
            });
        }
        m
    }

    #[test]
    fn round_trips_and_validates() {
        let m = full(2);
        let back = Manifest::parse(&m.to_text()).unwrap();
        assert_eq!(back, m);
        back.validate(2, &Expect::SERVING).unwrap();
        assert!(back.validate(1, &Expect::SERVING).is_err(), "another rank's share");
        assert_eq!(file_name(3, 0), "L03.r0.exl3");
    }

    #[test]
    fn incomplete_or_foreign_manifests_are_refused() {
        let mut m = full(0);
        m.layers.pop();
        assert!(m.validate(0, &Expect::SERVING).unwrap_err().contains("missing layers [44]"));
        m.validate(0, &Expect { all_layers: false, ..Expect::SERVING }).expect("a partial directory, when allowed");
        let mut m = full(0);
        m.layers[3].bytes -= 1;
        assert!(m.validate(0, &Expect::SERVING).is_err());
        let mut m = full(0);
        m.layers.push(m.layers[0].clone());
        assert!(m.validate(0, &Expect::SERVING).unwrap_err().contains("twice"));
        let mut m = full(0);
        m.layout = "other".into();
        assert!(m.validate(0, &Expect::SERVING).is_err());
        let mut m = full(0);
        m.layers.push(LayerEntry { layer: 45, file: file_name(45, 0), bytes: LAYER_BYTES as u64, sha256: "00".repeat(32) });
        m.validate(0, &Expect::SERVING).expect("the MTP layer is optional and allowed");
        assert!(Manifest::parse("not a manifest\n").is_err());
        assert!(Manifest::parse(&full(0).to_text().replace(&"ab".repeat(32), "xyz")).is_err());
    }
}
