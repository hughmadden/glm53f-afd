//! The embedding table in page-locked host RAM (decision D7): 1.27 GB of BF16 that never
//! occupies the GPU. A step gathers only the rows it uses, straight into the 4 mHC streams:
//! the gather kernel reads the mapped host memory over PCIe (8 KiB per row), so token ids that
//! are already on the device (a sampled token) need no host round trip.

use std::fs::File;
use std::os::unix::fs::FileExt;

use glm53f_model::catalog::EMBED;
use glm53f_model::dtype::DType;
use glm53f_model::safetensors::Checkpoint;

use crate::device::{launched, PinnedBuffer, Stream};
use crate::error::{invalid, Error, Result};
use crate::ffi;
use crate::shape::{HIDDEN, VOCAB};

pub struct HostEmbedding {
    table: PinnedBuffer,
    pub vocab: usize,
    pub hidden: usize,
}

impl HostEmbedding {
    /// Read `model.language_model.embed_tokens.weight` into page-locked host memory.
    pub fn load(ckpt: &Checkpoint) -> Result<HostEmbedding> {
        let (shard, e) = ckpt
            .get(EMBED)
            .ok_or_else(|| invalid!("the checkpoint has no {EMBED}"))?;
        if e.dtype != DType::BF16 || e.shape != [VOCAB as u64, HIDDEN as u64] {
            return Err(invalid!("{EMBED}: {} {:?}", e.dtype, e.shape));
        }
        let (_, begin, end) = ckpt.file_range(EMBED)?;
        let table = PinnedBuffer::alloc((end - begin) as usize)?;
        let path = ckpt.dir.join(&shard.file);
        let f =
            File::open(&path).map_err(|err| Error::Other(format!("{}: {err}", path.display())))?;
        f.read_exact_at(table.as_bytes_mut(), begin)
            .map_err(|err| Error::Other(format!("{}: {err}", path.display())))?;
        Ok(HostEmbedding {
            table,
            vocab: VOCAB,
            hidden: HIDDEN,
        })
    }

    /// Bytes of host memory held.
    pub fn bytes(&self) -> usize {
        self.table.bytes()
    }

    /// Row `id` on the host (BF16 bits).
    pub fn row(&self, id: usize) -> &[u16] {
        let b = &self.table.as_bytes()[id * self.hidden * 2..(id + 1) * self.hidden * 2];
        // SAFETY: the pinned allocation is 16-byte aligned and holds whole BF16 values.
        unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u16>(), self.hidden) }
    }

    /// Gather the rows of `ids` (device i32 `[rows]`) into `streams` (device BF16
    /// `[rows][4][hidden]`), and into `rows_out` (`[rows][hidden]`) when given.
    ///
    /// # Safety
    ///
    /// `ids`, `streams` and `rows_out` (when non-null) are device pointers to live buffers of
    /// `rows` rows.
    pub unsafe fn gather(
        &self,
        ids: *const i32,
        rows: usize,
        streams: *mut u16,
        rows_out: *mut u16,
        stream: &Stream,
    ) -> Result<()> {
        // SAFETY: device buffers sized by the caller; the table is mapped host memory.
        let code = unsafe {
            ffi::glm53f_fwd_embed_gather(
                self.table.device_ptr(),
                self.vocab as i64,
                ids,
                rows as i32,
                self.hidden as i32,
                streams,
                rows_out,
                stream.raw(),
            )
        };
        launched(code, "glm53f_fwd_embed_gather")
    }
}
