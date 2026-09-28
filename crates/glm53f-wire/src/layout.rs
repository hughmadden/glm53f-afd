//! DS41RTE3 v3 wire layout — field offsets, wire codes, row accounting.
//!
//! Format derivation (copy-in source, READ-ONLY): tpurtell/ds41rt v10
//! `rust/crates/ds41rt-transport/src/protocol_v2.rs` — magic/version `:6-7`,
//! row/route lengths `:11-16`, dtype codes `:45-60`, source kinds `:122-128`,
//! header/row/route structs `:261-292`, row/route byte codecs `:580-620`. Wire
//! magic stays `DS41RTE3` (ARCHITECTURE.md §9 item 2, skeleton default "keep").
//!
//! Row accounting (ADVISOR-I4 §3.2 items 5+6, ARCHITECTURE.md §11.5):
//!
//! * request row = 40-B row descriptor + 4,224-B `Fp8E4m3Ue8m0K32` hidden row
//!   (4,096 E4M3 payload + 128 UE8M0 K32 scales) + 8 x 12-B route entries
//!   = **4,360 B** per token per layer per Spark (4360 x 42 x 4 = 0.73 MB
//!   out/token over GLM-5.3-Flash's 42 MoE layers).
//! * compact return row = 4,096 BF16 = **8,192 B** per token per layer per
//!   Spark, the Spark pre-summing its 8 weighted route partials (8192 x 42 x 4
//!   = 1.38 MB in/token). NEVER per-route FP32 returns
//!   (the 24.6 MB/token wall, ADVISOR-I4 A6).
//!
//! The constants below carry compile-time guards; `tests/wire_layout.rs` pins
//! them again at runtime and against real encoded frame bodies.

use crate::error::WireError;
use crate::naive::WireNaive;

/// Frame magic (ARCHITECTURE.md §9 item 2: keep `DS41RTE3`).
pub const MAGIC: [u8; 8] = *b"DS41RTE3";
/// Frame format version (DS41RTE3 v3).
pub const VERSION: u16 = 3;
/// Version 4: version 3 plus the prefill reduce-scatter (`crate::row_shard`). The header and
/// its L4 tail are unchanged. A frame is version 4 only when it uses the extension (a
/// reduce-scattered request, a row-slice return, an exchange frame), so every other frame stays
/// version 3 and a version-3 peer refuses exactly the frames it cannot read.
pub const VERSION_ROW_SHARD: u16 = 4;
/// Message kind: coordinator -> Spark request frame.
pub const KIND_REQUEST: u16 = 1;
/// Message kind: Spark -> coordinator compact return frame.
pub const KIND_RETURN: u16 = 2;
/// Message kind (version 4 only): the rank-to-rank exchange frame of the prefill reduce-scatter.
pub const KIND_EXCHANGE: u16 = 3;

/// Unified frame header length (v3 L4-extended: seq + CRC32C in the tail).
pub const HEADER_LEN: usize = 128;
/// Row descriptor length (ds41rt `EXPERT_PROTOCOL_V2_ROW_DESCRIPTOR_LEN`).
pub const ROW_DESCRIPTOR_LEN: usize = 40;
/// Route entry length (ds41rt `EXPERT_PROTOCOL_V2_ROUTE_ENTRY_LEN`).
pub const ROUTE_ENTRY_LEN: usize = 12;

/// Hidden width (GLM-5.3-Flash, as MiMo-V2.6-Flash; also the return width).
pub const HIDDEN: usize = 4096;
/// Router top-k (sigmoid + `e_score_correction_bias` + norm-topk, top-8 of
/// GLM-5.3-Flash's 288 routed experts; expert ids are `u32` on the wire).
pub const TOPK: usize = 8;
/// Spark ranks (TP4EP1 quarter slices; coordinator sums 4 partials in FP32).
pub const SPARKS: usize = 4;
/// UE8M0 scale block width (`Fp8E4m3Ue8m0K32`: one scale byte per 32 values).
pub const K32: usize = 32;
/// Naive scale block width (the K-slip trap).
pub const K16: usize = 16;

/// Canonical hidden row: 4,096 E4M3 payload + 128 UE8M0 K32 scales = 4,224 B.
pub const HIDDEN_ROW_BYTES: usize = HIDDEN + HIDDEN / K32;

/// **Request row contract: 4,360 B** (descriptor + hidden row + top-8 routes).
pub const REQUEST_ROW_BYTES: usize =
    ROW_DESCRIPTOR_LEN + HIDDEN_ROW_BYTES + TOPK * ROUTE_ENTRY_LEN;

/// **Return row contract: 8,192 B** BF16 per Spark per token (pre-summed).
pub const RETURN_ROW_BYTES: usize = HIDDEN * 2;

// Static layout guards: these fail the BUILD if the row contracts drift.
const _: [(); 4224] = [(); HIDDEN_ROW_BYTES];
const _: [(); 4360] = [(); REQUEST_ROW_BYTES];
const _: [(); 8192] = [(); RETURN_ROW_BYTES];

/// Unified 128-B frame header field offsets (little-endian).
pub mod hdr {
    pub const MAGIC: usize = 0; // [u8; 8]
    pub const VERSION: usize = 8; // u16
    pub const KIND: usize = 10; // u16
    pub const HEADER_LEN: usize = 12; // u32
    pub const REQUEST_ID: usize = 16; // u64
    pub const PLACEMENT_VERSION: usize = 24; // u64
    pub const LAYER_ID: usize = 32; // u32
    pub const ROW_COUNT: usize = 36; // u32
    pub const DIM: usize = 40; // u32 (hidden_dim / output_dim)
    pub const PAYLOAD_DTYPE: usize = 44; // u16
    pub const SOURCE_KIND: usize = 46; // u16
    pub const ROUTE_COUNT: usize = 48; // u32
    pub const ROW_DESCRIPTOR_BYTES: usize = 52; // u32
    pub const ROUTE_BYTES: usize = 56; // u32
    pub const PAYLOAD_BYTES: usize = 60; // u64
    pub const LOGICAL_PAYLOAD_BYTES: usize = 68; // u64
    pub const WIRE_BYTES: usize = 76; // u64
    pub const FLAGS: usize = 84; // u32
    pub const ROW_STRIDE_BYTES: usize = 88; // u32
    pub const STATUS: usize = 92; // u32
    pub const EXECUTOR_ID: usize = 96; // u64
    pub const TOKEN_POSITION: usize = 104; // u64
    pub const SEQ: usize = 112; // u64 (L4 sequence)
    pub const CRC32C: usize = 120; // u32 (L4 checksum; zeroed while computing)
    pub const RESERVED: usize = 124; // u32 (must be 0; an exchange frame's row-shard word)
}

/// Row descriptor field offsets (40 B, byte-exact ds41rt v3).
pub mod desc {
    pub const ROW_ID: usize = 0; // u64
    pub const SOURCE_KIND: usize = 8; // u16
    pub const PAD0: usize = 10; // u16 (0)
    pub const SOURCE_REQUEST_ID: usize = 12; // u64
    pub const TOKEN_POSITION: usize = 20; // u64
    pub const ROUTE_OFFSET: usize = 28; // u32
    pub const ROUTE_COUNT: usize = 32; // u32
    pub const PAD1: usize = 36; // u32 (0)
}

/// Route entry field offsets (12 B, byte-exact ds41rt v3).
pub mod route {
    pub const ROW_INDEX: usize = 0; // u32
    pub const EXPERT_ID: usize = 4; // u32
    pub const GATE_WEIGHT: usize = 8; // f32 (LE)
}

/// Payload dtype wire codes (ds41rt `ExpertV2Dtype`, protocol_v2.rs:45-60).
/// Codes are a WIRE contract — renumbering breaks the seam silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Dtype {
    Bf16 = 1,
    F16 = 2,
    Fp8Debug = 3,
    Nvfp4E2m1Fp8E4m3 = 4,
    /// E4M3 payload + one FP32 dequant scale per row.
    Fp8E4m3RowScaled = 5,
    F32 = 6,
    /// E4M3 payload + one UE8M0 scale byte per contiguous 32 values.
    Fp8E4m3Ue8m0K32 = 7,
}

impl Dtype {
    /// Wire code (DS41RTE3 v3 numbering).
    pub fn code(self, naive: WireNaive) -> u16 {
        let raw = self as u16;
        if naive.has(WireNaive::DTYPE_RECODE) {
            raw.wrapping_add(100)
        } else {
            raw
        }
    }

    pub fn from_code(code: u16, naive: WireNaive) -> Result<Self, WireError> {
        let raw = if naive.has(WireNaive::DTYPE_RECODE) {
            code.wrapping_sub(100)
        } else {
            code
        };
        match raw {
            1 => Ok(Self::Bf16),
            2 => Ok(Self::F16),
            3 => Ok(Self::Fp8Debug),
            4 => Ok(Self::Nvfp4E2m1Fp8E4m3),
            5 => Ok(Self::Fp8E4m3RowScaled),
            6 => Ok(Self::F32),
            7 => Ok(Self::Fp8E4m3Ue8m0K32),
            _ => Err(WireError::BadCode { field: "payload_dtype", code: code as u32 }),
        }
    }
}

/// Row/traffic source class (ds41rt `ExpertV2SourceKind`, protocol_v2.rs:122-128).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum SourceKind {
    Decode = 1,
    Prefill = 2,
    MtpVerify = 3,
    Benchmark = 4,
}

impl SourceKind {
    pub fn code(self) -> u16 {
        self as u16
    }

    pub fn from_code(code: u16) -> Result<Self, WireError> {
        match code {
            1 => Ok(Self::Decode),
            2 => Ok(Self::Prefill),
            3 => Ok(Self::MtpVerify),
            4 => Ok(Self::Benchmark),
            _ => Err(WireError::BadCode { field: "source_kind", code: code as u32 }),
        }
    }
}

/// Frame status (ds41rt `ExpertProtocolV2Status`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Status {
    Ok = 0,
    Error = 1,
}

impl Status {
    pub fn code(self) -> u32 {
        self as u32
    }

    pub fn from_code(code: u32) -> Result<Self, WireError> {
        match code {
            0 => Ok(Self::Ok),
            1 => Ok(Self::Error),
            _ => Err(WireError::BadCode { field: "status", code }),
        }
    }
}

/// UE8M0-K32 row byte count (ds41rt `row_bytes` formula: elements +
/// elements/32; width must be a nonzero multiple of 32 — fail loud, never
/// silently truncate a scale grid).
pub fn ue8m0_k32_row_bytes(elements: usize) -> Result<usize, WireError> {
    if elements == 0 || elements % K32 != 0 {
        return Err(WireError::DimMismatch {
            field: "ue8m0_k32_row_width",
            want: K32,
            got: elements,
        });
    }
    Ok(elements + elements / K32)
}

/// Scale block width on the wire (32 canonical; 16 in the K-slip trap).
pub fn scale_k(naive: WireNaive) -> usize {
    if naive.has(WireNaive::K16_SCALES) {
        K16
    } else {
        K32
    }
}

/// Hidden row bytes including the scale region (4,224 canonical).
pub fn hidden_row_bytes(naive: WireNaive) -> usize {
    if naive.has(WireNaive::ROW_SCALED_DTYPE) {
        HIDDEN + 4
    } else {
        HIDDEN + HIDDEN / scale_k(naive)
    }
}

/// **Request row bytes: 4,360 canonical** (descriptor + hidden row + top-8
/// route entries).
pub fn request_row_bytes(naive: WireNaive) -> usize {
    ROW_DESCRIPTOR_LEN + hidden_row_bytes(naive) + TOPK * ROUTE_ENTRY_LEN
}

/// **Return row bytes per token per Spark: 8,192 canonical** — independent of
/// `route_count` because the Spark pre-sums its weighted route partials
/// (A6: per-route FP32 returns are the 24.6 MB/token wall).
pub fn return_row_bytes(route_count: usize, naive: WireNaive) -> usize {
    if naive.has(WireNaive::PER_ROUTE_RETURN) {
        route_count * HIDDEN * 2
    } else if naive.has(WireNaive::FP32_RETURN) {
        HIDDEN * 4
    } else {
        RETURN_ROW_BYTES
    }
}

/// Env-default layout helpers (NEGATIVE tests call these; they drift when the
/// naive run selects a layout trap).
pub fn request_row_bytes_env() -> usize {
    request_row_bytes(crate::naive::naive_from_env())
}

pub fn return_row_bytes_env(route_count: usize) -> usize {
    return_row_bytes(route_count, crate::naive::naive_from_env())
}
