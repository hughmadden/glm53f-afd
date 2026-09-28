//! The page cache and the rank's layer images (written here).
//!
//! A rank reads each layer image twice, both times through the page cache: the boot readback
//! hashes it ([`crate::boot`]) and the upload reads it again ([`crate::resident`]). Nothing gave
//! the pages back, so a rank left about 38 GB of cached images behind. On GB10's unified memory
//! the kernel counts cached pages as available (`MemAvailable`), but a CUDA allocation can fail
//! before it reclaims them: the figure an allocation sees is `MemFree`. [`drop_file`] hands a
//! file's clean pages back with `posix_fadvise(POSIX_FADV_DONTNEED)`, which needs no privilege and
//! changes no byte the rank reads or serves; [`memory_line`] puts both figures in the boot log.
//!
//! The drop is a hint, so a caller logs its error and goes on. Off 64-bit Linux it does nothing.

use std::path::Path;

/// `POSIX_FADV_DONTNEED` (`<fcntl.h>`; 4 on x86-64 and aarch64 Linux).
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
const POSIX_FADV_DONTNEED: core::ffi::c_int = 4;

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
unsafe extern "C" {
    /// `int posix_fadvise(int fd, off_t offset, off_t len, int advice)`: 0, or the error number
    /// itself (it does not set `errno`). `off_t` is `long` on 64-bit Linux.
    fn posix_fadvise(fd: core::ffi::c_int, offset: core::ffi::c_long, len: core::ffi::c_long, advice: core::ffi::c_int) -> core::ffi::c_int;
}

/// Give back the page cache's copy of the file at `path`. Pages not yet written out stay, which
/// the read-only images never are.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub fn drop_file(path: &Path) -> Result<(), String> {
    use std::os::fd::AsRawFd;
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: `f` keeps the descriptor open for the call; offset 0 and length 0 are the whole file.
    let rc = unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
    if rc != 0 {
        return Err(format!("{}: posix_fadvise: {}", path.display(), std::io::Error::from_raw_os_error(rc)));
    }
    Ok(())
}

/// Where the hint is not supported: nothing to do.
#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
pub fn drop_file(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// `MemAvailable` and `MemFree`, in kB, from the text of `/proc/meminfo`.
pub fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let field = |name: &str| {
        text.lines().find_map(|l| l.strip_prefix(name)?.strip_prefix(':')?.split_whitespace().next()?.parse::<u64>().ok())
    };
    Some((field("MemAvailable")?, field("MemFree")?))
}

/// The two figures for the boot log, like `MemAvailable 96.2 GiB, MemFree 88.1 GiB`; `None` where
/// `/proc/meminfo` cannot be read.
pub fn memory_line() -> Option<String> {
    let (avail, free) = parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)?;
    let gib = |kb: u64| kb as f64 / (1u64 << 20) as f64;
    Some(format!("MemAvailable {:.1} GiB, MemFree {:.1} GiB", gib(avail), gib(free)))
}
