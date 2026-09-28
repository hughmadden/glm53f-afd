//! The page-cache hint for the layer images (`pagecache`, written here): what it reads from
//! `/proc/meminfo`, that dropping a file's cached pages leaves its bytes alone, and (Linux) that
//! the pages really leave the cache, counted with `mincore(2)`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use glm53f_rank::pagecache;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn scratch_file(name: &str, len: usize) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join(format!("glm53f-rank-pagecache-{name}-{}-{n}", std::process::id()));
    let bytes: Vec<u8> = (0..len).map(|i| (i as u32).wrapping_mul(2_654_435_761) as u8).collect();
    std::fs::write(&path, &bytes).expect("write");
    // Dirty pages are not dropped: the images are read-only files, so start from clean ones.
    std::fs::File::open(&path).and_then(|f| f.sync_all()).expect("sync");
    path
}

#[test]
fn meminfo_gives_available_and_free() {
    let text = "MemTotal:       124698252 kB\nMemFree:        91234560 kB\nMemAvailable:   100663296 kB\nBuffers:            1234 kB\n";
    assert_eq!(pagecache::parse_meminfo(text), Some((100_663_296, 91_234_560)));
    // Either field missing: nothing to report.
    assert_eq!(pagecache::parse_meminfo("MemFree: 5 kB\n"), None);
    assert_eq!(pagecache::parse_meminfo("MemAvailable: 5 kB\nMemFreeish: 1 kB\n"), None);
    assert_eq!(pagecache::parse_meminfo(""), None);
    #[cfg(target_os = "linux")]
    {
        let line = pagecache::memory_line().expect("/proc/meminfo");
        assert!(line.starts_with("MemAvailable ") && line.contains(" GiB, MemFree ") && line.ends_with(" GiB"), "{line}");
    }
}

#[test]
fn dropping_a_files_cache_leaves_its_bytes() {
    let path = scratch_file("bytes", 3 << 20);
    let before = std::fs::read(&path).unwrap();
    pagecache::drop_file(&path).expect("drop");
    assert_eq!(std::fs::read(&path).unwrap(), before);
    // Twice is fine, and a missing file is an error (a hint that cannot be given), not a panic.
    pagecache::drop_file(&path).expect("drop again");
    std::fs::remove_file(&path).unwrap();
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    assert!(pagecache::drop_file(&path).unwrap_err().contains("glm53f-rank-pagecache-bytes"));
}

#[cfg(target_os = "linux")]
mod resident_pages {
    use super::*;
    use core::ffi::{c_int, c_long, c_void};

    // The kernel's view of a file's pages: map it, ask `mincore`, unmap (a mapped page is not dropped).
    unsafe extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: c_long) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
        fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int;
        fn sysconf(name: c_int) -> c_long;
    }
    const PROT_READ: c_int = 1;
    const MAP_SHARED: c_int = 1;
    const SC_PAGESIZE: c_int = 30;

    /// The fraction of `path`'s pages in the page cache.
    fn cached_fraction(path: &std::path::Path) -> f64 {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open(path).unwrap();
        let len = f.metadata().unwrap().len() as usize;
        // SAFETY: a read-only shared mapping of the whole file, unmapped before returning; `vec`
        // has one byte per page, the size `mincore` writes.
        unsafe {
            let page = sysconf(SC_PAGESIZE) as usize;
            let map = mmap(core::ptr::null_mut(), len, PROT_READ, MAP_SHARED, f.as_raw_fd(), 0);
            assert_ne!(map as isize, -1, "mmap");
            let mut vec = vec![0u8; len.div_ceil(page)];
            let rc = mincore(map, len, vec.as_mut_ptr());
            munmap(map, len);
            assert_eq!(rc, 0, "mincore");
            vec.iter().filter(|&&b| b & 1 == 1).count() as f64 / vec.len() as f64
        }
    }

    /// The filesystem type under `path`, from `/proc/mounts` (the longest mount point that prefixes it).
    fn fs_type(path: &std::path::Path) -> String {
        let path = path.canonicalize().unwrap();
        let mounts = std::fs::read_to_string("/proc/mounts").unwrap_or_default();
        mounts
            .lines()
            .filter_map(|l| {
                let mut f = l.split_whitespace().skip(1);
                let (mount, kind) = (f.next()?, f.next()?);
                path.starts_with(mount).then(|| (mount.len(), kind.to_string()))
            })
            .max()
            .map(|(_, kind)| kind)
            .unwrap_or_default()
    }

    /// The pages of a file that was just read leave the cache. On a filesystem that has no
    /// other copy of the data (tmpfs, or an overlay that caches elsewhere) there is nothing to
    /// drop, and the test says so and passes.
    #[test]
    fn dropping_a_files_cache_empties_it() {
        let path = scratch_file("cached", 64 << 20);
        let kind = fs_type(&path);
        if ["tmpfs", "ramfs", "overlay"].contains(&kind.as_str()) {
            eprintln!("skip: {kind} keeps the pages of a file it holds");
            std::fs::remove_file(&path).unwrap();
            return;
        }
        std::fs::read(&path).unwrap();
        let cached = cached_fraction(&path);
        assert!(cached > 0.9, "a file just read is cached ({cached}, {kind})");
        pagecache::drop_file(&path).expect("drop");
        let left = cached_fraction(&path);
        assert!(left < 0.1, "{left} of the file is still cached after the drop ({kind})");
        std::fs::remove_file(&path).unwrap();
    }
}
