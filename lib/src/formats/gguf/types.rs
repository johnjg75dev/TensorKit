//! GGUF core types: tensor element types, metadata values, tensor info.
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use crate::model::TensorDtype;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const DEFAULT_ALIGNMENT: usize = 32;

/// GGML tensor element / block types. Values from ggml.h.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Unknown(u32),
}

impl GgmlType {
    #[inline]
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            other => Self::Unknown(other),
        }
    }

    #[inline]
    pub fn to_tensor_dtype(self) -> TensorDtype {
        match self {
            Self::F32 => TensorDtype::F32,
            Self::F16 => TensorDtype::F16,
            Self::Bf16 => TensorDtype::Bf16,
            Self::F64 => TensorDtype::F64,
            Self::I8 => TensorDtype::I8,
            Self::I16 => TensorDtype::I16,
            Self::I32 => TensorDtype::I32,
            Self::I64 => TensorDtype::I64,
            Self::Q4_0 => TensorDtype::Q4_0,
            Self::Q4_1 => TensorDtype::Q4_1,
            Self::Q5_0 => TensorDtype::Q5_0,
            Self::Q5_1 => TensorDtype::Q5_1,
            Self::Q8_0 => TensorDtype::Q8_0,
            Self::Q8_1 => TensorDtype::Q8_1,
            Self::Q2K => TensorDtype::Q2K,
            Self::Q3K => TensorDtype::Q3K,
            Self::Q4K => TensorDtype::Q4K,
            Self::Q5K => TensorDtype::Q5K,
            Self::Q6K => TensorDtype::Q6K,
            Self::Q8K => TensorDtype::Q8K,
            Self::Iq2Xxs => TensorDtype::Iq2Xxs,
            Self::Iq2Xs => TensorDtype::Iq2Xs,
            Self::Iq3Xxs => TensorDtype::Iq3Xxs,
            Self::Iq1S => TensorDtype::Iq1S,
            Self::Iq4Nl => TensorDtype::Iq4Nl,
            Self::Iq3S => TensorDtype::Iq3S,
            Self::Iq2S => TensorDtype::Iq2S,
            Self::Iq4Xs => TensorDtype::Iq4Xs,
            Self::Iq1M => TensorDtype::Iq1M,
            Self::Tq1_0 => TensorDtype::Tq1_0,
            Self::Tq2_0 => TensorDtype::Tq2_0,
            Self::Unknown(v) => TensorDtype::Unknown(v),
        }
    }

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

    /// Quantized block size in elements. 1 for non-quantized scalar types.
    #[inline]
    pub const fn block_size(self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::Bf16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::Iq4Nl => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq4Xs
            | Self::Iq1M
            | Self::Tq1_0
            | Self::Tq2_0 => 256,
            Self::Unknown(_) => 1,
        }
    }

    /// Bytes per quantized block. Returns None for unknown / unsupported types.
    #[inline]
    pub const fn block_bytes(self) -> Option<usize> {
        Some(match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Bf16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::Q2K => 82,  // 256/16 * (2+2+2+16) = 82
            Self::Q3K => 110, // 256/16 * (2+1+16) + 4 = 110
            Self::Iq1S => 50,
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq3Xxs => 98,
            Self::Iq3S => 110,
            Self::Iq2S => 82,
            Self::Iq4Nl => 18,
            Self::Iq4Xs => 136,
            Self::Iq1M => 56,
            Self::Tq1_0 => 38,
            Self::Tq2_0 => 66,
            _ => return None,
        })
    }

    #[inline]
    pub fn is_dequantizable(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::Bf16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::Q4K
                | Self::Q5K
                | Self::Q6K
                | Self::Q8K
                | Self::Q2K
                | Self::Q3K
                | Self::Iq1S
                | Self::Iq2Xxs
                | Self::Iq2Xs
                | Self::Iq3Xxs
                | Self::Iq3S
                | Self::Iq2S
                | Self::Iq4Nl
                | Self::Iq4Xs
                | Self::Iq1M
                | Self::Tq1_0
                | Self::Tq2_0
        )
    }

    /// Raw on-disk u32 value (for the writer).
    #[inline]
    pub fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::Iq1M => 29,
            Self::Bf16 => 30,
            Self::Tq1_0 => 34,
            Self::Tq2_0 => 35,
            Self::Unknown(v) => v,
        }
    }
}

/// One tensor slot in the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: [u64; 4],
    pub ggml_type: GgmlType,
    /// Offset relative to the start of the tensor data section.
    pub offset: u64,
    pub n_elements: u64,
    pub byte_size: u64,
}

#[derive(Debug, Clone)]
pub struct MetadataKv {
    pub key: String,
    pub value_type: u32,
    pub value: MetaValue,
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(ArrayValue),
}

#[derive(Debug, Clone)]
pub struct ArrayValue {
    pub elem_type: u32,
    pub elements: Vec<MetaValue>,
}

#[inline]
pub fn dims_product(dims: &[u64; 4], n_dims: u32) -> u64 {
    let mut p = 1u64;
    for i in 0..n_dims as usize {
        p = p.saturating_mul(dims[i].max(1));
    }
    p
}

#[inline]
pub fn byte_size_for(n_elements: u64, ty: GgmlType) -> u64 {
    let block_elems = ty.block_size() as u64;
    match ty.block_bytes() {
        Some(b) => {
            let n_blocks = n_elements.div_ceil(block_elems);
            n_blocks * b as u64
        }
        None => n_elements * ty_size_guess(ty) as u64,
    }
}

#[inline]
fn ty_size_guess(ty: GgmlType) -> usize {
    match ty {
        GgmlType::F32 => 4,
        GgmlType::F16 => 2,
        GgmlType::Bf16 => 2,
        GgmlType::F64 => 8,
        GgmlType::I8 => 1,
        GgmlType::I16 => 2,
        GgmlType::I32 => 4,
        GgmlType::I64 => 8,
        _ => 1,
    }
}
