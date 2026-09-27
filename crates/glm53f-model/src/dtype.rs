//! Safetensors element types.

use std::fmt;

/// A safetensors element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    /// FP8 E4M3 (OCP "fn" variant, as safetensors `F8_E4M3`).
    F8E4M3,
    F8E5M2,
    /// FP8 E8M0: a power-of-two scale.
    F8E8M0,
}

impl DType {
    /// Parse the safetensors dtype string.
    pub fn parse(s: &str) -> Option<DType> {
        Some(match s {
            "BOOL" => DType::Bool,
            "U8" => DType::U8,
            "I8" => DType::I8,
            "U16" => DType::U16,
            "I16" => DType::I16,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "U32" => DType::U32,
            "I32" => DType::I32,
            "F32" => DType::F32,
            "U64" => DType::U64,
            "I64" => DType::I64,
            "F64" => DType::F64,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E5M2" => DType::F8E5M2,
            "F8_E8M0" => DType::F8E8M0,
            _ => return None,
        })
    }

    /// The safetensors dtype string.
    pub fn as_str(self) -> &'static str {
        match self {
            DType::Bool => "BOOL",
            DType::U8 => "U8",
            DType::I8 => "I8",
            DType::U16 => "U16",
            DType::I16 => "I16",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::U32 => "U32",
            DType::I32 => "I32",
            DType::F32 => "F32",
            DType::U64 => "U64",
            DType::I64 => "I64",
            DType::F64 => "F64",
            DType::F8E4M3 => "F8_E4M3",
            DType::F8E5M2 => "F8_E5M2",
            DType::F8E8M0 => "F8_E8M0",
        }
    }

    /// Bytes per element.
    pub fn size(self) -> u64 {
        match self {
            DType::Bool | DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E5M2 | DType::F8E8M0 => {
                1
            }
            DType::U16 | DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Number of elements of a shape (1 for a scalar).
pub fn numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_round_trips() {
        for s in [
            "BOOL", "U8", "I8", "U16", "I16", "F16", "BF16", "U32", "I32", "F32", "U64", "I64",
            "F64", "F8_E4M3", "F8_E5M2", "F8_E8M0",
        ] {
            assert_eq!(DType::parse(s).unwrap().as_str(), s);
        }
        assert_eq!(DType::parse("F4"), None);
        assert_eq!(numel(&[]), 1);
        assert_eq!(numel(&[2, 3, 0]), 0);
    }
}
