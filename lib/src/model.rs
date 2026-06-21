//! Format-agnostic model trait and common types.

use crate::Result;
use std::borrow::Cow;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    Onnx,
    Unknown,
}

impl ModelFormat {
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("gguf") => Self::Gguf,
            Some(ext)
                if ext.eq_ignore_ascii_case("safetensors")
                    || ext.eq_ignore_ascii_case("st") =>
            {
                Self::Safetensors
            }
            Some(ext) if ext.eq_ignore_ascii_case("onnx") => Self::Onnx,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Safetensors => "safetensors",
            Self::Onnx => "onnx",
            Self::Unknown => "unknown",
        }
    }
}

/// Format-agnostic tensor descriptor.
///
/// # Alignment
///
/// Tensor data should be aligned to at least 32 bytes when possible to enable
/// AVX2 SIMD operations (`_mm256_load_ps`, `_mm256_store_ps`). When using
/// memory-mapped files (GGUF format), alignment depends on the file offset and
/// the `general.alignment` metadata field (default 32 bytes). For in-memory
/// tensors, use `AlignedVec` or ensure allocations meet the alignment requirement.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub byte_size: u64,
    pub data_offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    F32,
    F16,
    Bf16,
    F64,
    I8,
    I16,
    I32,
    I64,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    Iq1M,
    Tq1_0,
    Tq2_0,
    Unknown(u32),
}

impl TensorDtype {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq1S => "IQ1_S",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2S => "IQ2_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::Iq1M => "IQ1_M",
            Self::Tq1_0 => "TQ1_0",
            Self::Tq2_0 => "TQ2_0",
            Self::Unknown(_) => "?",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockRef {
    pub index: i32,
    pub label: &'static str,
}

#[derive(Debug, Clone)]
pub enum MetadataValue<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(&'a str),
    U64(u64),
    I64(i64),
    F64(f64),
}

pub trait Model: Send + Sync {
    fn format(&self) -> ModelFormat;
    fn name(&self) -> Option<&str>;
    fn architecture(&self) -> Option<&str>;
    fn block_count(&self) -> Option<usize>;
    fn tensors(&self) -> &[Tensor];
    fn tensor(&self, name: &str) -> Option<&Tensor>;
    fn metadata(&self, key: &str) -> Option<MetadataValue<'_>>;
    fn read_tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>>;
}
