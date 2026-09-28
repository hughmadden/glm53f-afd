//! Boot identity readback and residency (after mimo26f-afd's
//! `crates/mimo26-spark/tests/boot.rs`). A rank that serves a corrupted,
//! truncated or missing layer image, an image its manifest does not list, or
//! another rank's share, must refuse. Small images stand in for the 0.91 GB
//! layer images (`Expect` carries the size).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use glm53f_rank::boot;
use glm53f_rank::manifest::{file_name, Expect, LayerEntry, Manifest};
use glm53f_rank::resident::Resident;
use glm53f_rank::sha256;

static COUNTER: AtomicUsize = AtomicUsize::new(0);
const BYTES: usize = 4096 + 17;
const SMALL: Expect = Expect { layer_bytes: BYTES as u64, all_layers: false };

fn scratch_dir(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("glm53f-rank-{name}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Write `layers` images for `rank` and their manifest.
fn write_dir(dir: &Path, rank: usize, layers: &[u32]) -> Manifest {
    let mut m = Manifest::new(rank, "test");
    for &layer in layers {
        let bytes: Vec<u8> = (0..BYTES).map(|i| (i as u32).wrapping_mul(layer + 7) as u8).collect();
        let file = file_name(layer, rank);
        std::fs::write(dir.join(&file), &bytes).expect("write image");
        m.layers.push(LayerEntry { layer, file, bytes: BYTES as u64, sha256: sha256::hex(&sha256::sha256(&bytes)) });
    }
    m.write(dir).expect("write manifest");
    m
}

#[test]
fn clean_boot_matches_every_image() {
    let dir = scratch_dir("clean");
    write_dir(&dir, 1, &[3, 4, 44]);
    let r = boot::readback(&dir, 1, &SMALL).expect("clean readback");
    assert_eq!((r.matched, r.total, r.matched_bytes), (3, 3, 3 * BYTES as u64));
    assert!(r.summary.contains("3/3"), "{}", r.summary);
    // A partial directory is not servable.
    assert!(boot::readback(&dir, 1, &Expect { all_layers: true, ..SMALL }).unwrap_err().contains("missing layers"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_corrupt_truncated_or_missing_image_refuses_to_serve_and_is_named() {
    let dir = scratch_dir("corrupt");
    write_dir(&dir, 0, &[3, 4, 5, 6]);
    let p = dir.join(file_name(4, 0));
    let mut b = std::fs::read(&p).unwrap();
    b[1000] ^= 1;
    std::fs::write(&p, &b).unwrap();
    let p = dir.join(file_name(5, 0));
    let b = std::fs::read(&p).unwrap();
    std::fs::write(&p, &b[..BYTES - 1]).unwrap();
    std::fs::remove_file(dir.join(file_name(6, 0))).unwrap();
    let e = boot::readback(&dir, 0, &SMALL).unwrap_err();
    assert!(e.contains("refusing to serve, 3 problem(s)"), "{e}");
    assert!(e.contains("L04.r0.exl3: sha256"), "{e}");
    assert!(e.contains("L05.r0.exl3") && e.contains("bytes"), "{e}");
    assert!(e.contains("L06.r0.exl3: missing"), "{e}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_unlisted_image_or_another_ranks_share_refuses_to_serve() {
    let dir = scratch_dir("unlisted");
    write_dir(&dir, 2, &[3]);
    std::fs::write(dir.join("L07.r2.exl3"), vec![0u8; BYTES]).unwrap();
    assert!(boot::readback(&dir, 2, &SMALL).unwrap_err().contains("not listed"));
    std::fs::remove_file(dir.join("L07.r2.exl3")).unwrap();
    assert!(boot::readback(&dir, 2, &SMALL).is_ok());
    assert!(boot::readback(&dir, 3, &SMALL).unwrap_err().contains("rank 2's share"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn resident_reads_the_listed_images() {
    let dir = scratch_dir("resident");
    write_dir(&dir, 3, &[3, 10]);
    let r = Resident::load_manifest(&dir, 3, &SMALL).unwrap();
    assert_eq!(r.layers(), vec![3, 10]);
    let img = r.layer_image(10).unwrap();
    assert_eq!(img.len(), BYTES);
    assert_eq!(img[1], 17);
    assert!(r.layer_image(4).is_err(), "not resident");
    std::fs::remove_dir_all(&dir).ok();
}

/// The rank drops an image's cached pages once it is on the device: the image reads back the same,
/// a layer the rank does not hold is an error, and a missing file is one on Linux.
#[test]
fn resident_drops_an_images_cache_and_it_reads_back_the_same() {
    let dir = scratch_dir("drop-cache");
    write_dir(&dir, 3, &[3, 10]);
    let r = Resident::load_manifest(&dir, 3, &SMALL).unwrap();
    let before = r.layer_image(10).unwrap();
    r.drop_cache(10).expect("drop");
    assert_eq!(r.layer_image(10).unwrap(), before);
    assert!(r.drop_cache(4).unwrap_err().contains("not resident on rank 3"));
    std::fs::remove_file(dir.join(file_name(3, 3))).unwrap();
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    assert!(r.drop_cache(3).unwrap_err().contains("L03.r3.exl3"));
    std::fs::remove_dir_all(&dir).ok();
}
