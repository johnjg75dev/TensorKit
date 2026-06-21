//! Pure-Rust GGML-style weight quantizers.
//!
//! The companion to `crate::formats::gguf::dequant`. Each submodule implements
//! a single block quantizer; `quantize` is the public dispatch entry point.
//!
//! Scope:
//!   - Per-block: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
//!   - K-quants:  Q4_K, Q5_K, Q6_K (Q8_K dequant exists but no quantizer
//!     — it's only useful for round-trip and is rarely used as
//!     a storage format since it's not smaller than F16).
//!   - I-quants:  not yet implemented.
//!
//! The K-quant quantizers match the canonical llama.cpp on-disk layout
//! (12-byte scale tables with the "shared low-4" or "shared high-2" packing
//! tricks). Quantization is per-sub-block with a simple max-absolute-value
//! scale; the encoder's 6-bit values stay in [0, 63] so the coupling
//! constraints of the canonical layout are satisfied trivially.

pub mod apply;
pub mod iq1_m;
pub mod iq1_s;
pub mod iq2_s;
pub mod iq2_xs;
pub mod iq2_xxs;
pub mod iq3_s;
pub mod iq3_xxs;
pub mod iq4_nl;
pub mod iq4_xs;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;
pub mod q8_1;
pub mod q8_k;
pub mod tq1_0;
pub mod tq2_0;
#[cfg(target_arch = "x86_64")]
mod simd;

use crate::formats::gguf::types::GgmlType;

/// Quantize a row-major `f32` buffer to the given GGML block type.
///
/// `src.len()` must be a multiple of the type's block size (32 for
/// Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, 256 for K-quants). Returns the raw little-endian
/// block bytes; the caller is responsible for any alignment padding required
/// by the container (GGUF uses 32-byte alignment between tensors).
///
/// # Panics
/// Panics if `ty` is not a supported quant type, or if `src.len()` is not a
/// multiple of the block size.
pub fn quantize(src: &[f32], ty: GgmlType) -> Vec<u8> {
    let block = ty.block_size();
    if block > 1 {
        assert!(
            src.len().is_multiple_of(block),
            "input length {} is not a multiple of block size {} for {:?}",
            src.len(),
            block,
            ty,
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
            && let Some(v) = unsafe { simd::try_quantize(src, ty) } {
                return v;
            }
    }

    dispatch_quantize(src, ty)
}

#[inline(never)]
pub fn dispatch_quantize(src: &[f32], ty: GgmlType) -> Vec<u8> {
    match ty {
        GgmlType::Q2K => q2_k::quantize(src),
        GgmlType::Q3K => q3_k::quantize(src),
        GgmlType::Q4_0 => q4_0::quantize(src),
        GgmlType::Q4_1 => q4_1::quantize(src),
        GgmlType::Q4K => q4_k::quantize(src),
        GgmlType::Q5_0 => q5_0::quantize(src),
        GgmlType::Q5_1 => q5_1::quantize(src),
        GgmlType::Q5K => q5_k::quantize(src),
        GgmlType::Q6K => q6_k::quantize(src),
        GgmlType::Q8_0 => q8_0::quantize(src),
        GgmlType::Q8_1 => q8_1::quantize(src),
        GgmlType::Q8K => q8_k::quantize(src),
        GgmlType::Iq2S => iq2_s::quantize(src),
        GgmlType::Iq2Xs => iq2_xs::quantize(src),
        GgmlType::Iq2Xxs => iq2_xxs::quantize(src),
        GgmlType::Iq3Xxs => iq3_xxs::quantize(src),
        GgmlType::Iq3S => iq3_s::quantize(src),
        GgmlType::Iq4Nl => iq4_nl::quantize(src),
        GgmlType::Iq4Xs => iq4_xs::quantize(src),
        GgmlType::Iq1M => iq1_m::quantize(src),
        GgmlType::Iq1S => iq1_s::quantize(src),
        GgmlType::Tq1_0 => tq1_0::quantize(src),
        GgmlType::Tq2_0 => tq2_0::quantize(src),
        other => panic!("quantize: unsupported type {:?}", other),
    }
}

/// Returns true if `ty` is a quant type accepted by `quantize`.
pub fn is_quantizable(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q4K
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8_0
            | GgmlType::Q8_1
            | GgmlType::Q8K
            | GgmlType::Iq2S
            | GgmlType::Iq2Xs
            | GgmlType::Iq2Xxs
            | GgmlType::Iq3Xxs
            | GgmlType::Iq3S
            | GgmlType::Iq4Nl
            | GgmlType::Iq4Xs
            | GgmlType::Iq1M
            | GgmlType::Iq1S
            | GgmlType::Tq1_0
            | GgmlType::Tq2_0
    )
}

/// Parallel variant of [`quantize`]. Splits `src` into per-block chunks and
/// quantizes each chunk in parallel via rayon, then concatenates the
/// resulting bytes in the original order. Output is byte-identical to
/// `quantize` because each block is independent and produces a
/// deterministic layout. Falls back to `quantize` for single-block inputs.
pub fn quantize_par(src: &[f32], ty: GgmlType) -> Vec<u8> {
    use rayon::prelude::*;
    let block = ty.block_size();
    if block <= 1 || src.len() < block * 2 {
        return quantize(src, ty);
    }
    let chunks: Vec<&[f32]> = src.chunks_exact(block).collect();
    let parts: Vec<Vec<u8>> = match ty {
        GgmlType::Q2K => chunks.par_iter().map(|c| q2_k::quantize(c)).collect(),
        GgmlType::Q3K => chunks.par_iter().map(|c| q3_k::quantize(c)).collect(),
        GgmlType::Q4_0 => chunks.par_iter().map(|c| q4_0::quantize(c)).collect(),
        GgmlType::Q4_1 => chunks.par_iter().map(|c| q4_1::quantize(c)).collect(),
        GgmlType::Q4K => chunks.par_iter().map(|c| q4_k::quantize(c)).collect(),
        GgmlType::Q5_0 => chunks.par_iter().map(|c| q5_0::quantize(c)).collect(),
        GgmlType::Q5_1 => chunks.par_iter().map(|c| q5_1::quantize(c)).collect(),
        GgmlType::Q5K => chunks.par_iter().map(|c| q5_k::quantize(c)).collect(),
        GgmlType::Q6K => chunks.par_iter().map(|c| q6_k::quantize(c)).collect(),
        GgmlType::Q8_0 => chunks.par_iter().map(|c| q8_0::quantize(c)).collect(),
        GgmlType::Q8_1 => chunks.par_iter().map(|c| q8_1::quantize(c)).collect(),
        GgmlType::Q8K => chunks.par_iter().map(|c| q8_k::quantize(c)).collect(),
        GgmlType::Iq2S => chunks.par_iter().map(|c| iq2_s::quantize(c)).collect(),
        GgmlType::Iq2Xs => chunks.par_iter().map(|c| iq2_xs::quantize(c)).collect(),
        GgmlType::Iq2Xxs => chunks.par_iter().map(|c| iq2_xxs::quantize(c)).collect(),
        GgmlType::Iq3Xxs => chunks.par_iter().map(|c| iq3_xxs::quantize(c)).collect(),
        GgmlType::Iq3S => chunks.par_iter().map(|c| iq3_s::quantize(c)).collect(),
        GgmlType::Iq4Nl => chunks.par_iter().map(|c| iq4_nl::quantize(c)).collect(),
        GgmlType::Iq4Xs => chunks.par_iter().map(|c| iq4_xs::quantize(c)).collect(),
        GgmlType::Iq1M => chunks.par_iter().map(|c| iq1_m::quantize(c)).collect(),
        GgmlType::Iq1S => chunks.par_iter().map(|c| iq1_s::quantize(c)).collect(),
        GgmlType::Tq1_0 => chunks.par_iter().map(|c| tq1_0::quantize(c)).collect(),
        GgmlType::Tq2_0 => chunks.par_iter().map(|c| tq2_0::quantize(c)).collect(),
        other => panic!("quantize_par: unsupported type {:?}", other),
    };
    let total: usize = parts.iter().map(|v| v.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in parts {
        out.extend(p);
    }
    out
}

/// f32 -> binary16 bit pattern (round-to-nearest-even, handles inf/nan).
#[inline]
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if v.is_infinite() {
        return sign | 0x7c00;
    }
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = (mant | 0x400) >> (1 - exp);
        return sign | (mant as u16);
    }
    sign | ((exp as u16) << 10) | (mant as u16)
}
