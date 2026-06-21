//! x86_64 SIMD dequantization. Requires AVX2 + F16C (Haswell+ era, ubiquitous
//! on anything still running). Falls back to scalar when the features aren't
//! available or for types we haven't vectorized yet.
//!
//! Function-level `#[target_feature(enable = "avx2,f16c")]` keeps these
//! functions callable from the dispatch site via a feature-detected branch.

use super::scalar::{bf16_to_f32, f16_to_f32};
use crate::formats::gguf::types::GgmlType;

/// Try to dequantize with SIMD. Returns `None` if the type isn't vectorized
/// here (caller should fall back to scalar).
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn try_dequant(ty: GgmlType, bytes: &[u8], max: usize) -> Option<Vec<f32>> {
    Some(match ty {
        GgmlType::F32 => unsafe { dequant_f32_avx2(bytes, max) },
        GgmlType::F16 => unsafe { dequant_f16_avx2(bytes, max) },
        GgmlType::Bf16 => unsafe { dequant_bf16_avx2(bytes, max) },
        GgmlType::Q4_0 => unsafe { dequant_q4_0_avx2(bytes, max) },
        GgmlType::Q8_0 => unsafe { dequant_q8_0_avx2(bytes, max) },
        GgmlType::Q8_1 => unsafe { dequant_q8_1_avx2(bytes, max) },
        // K-quants and I-quants aren't vectorized yet; scalar handles them.
        _ => return None,
    })
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// -- F32 -> F32 (trivially a 256-bit memcpy) ---------------------------------

#[target_feature(enable = "avx2")]
unsafe fn dequant_f32_avx2(bytes: &[u8], max: usize) -> Vec<f32> { unsafe {
    let n = bytes.len() / 4;
    let mut out: Vec<f32> = Vec::with_capacity(n);
    let mut i = 0;
    while i + 32 <= bytes.len() {
        let v = _mm256_loadu_ps(bytes.as_ptr().add(i) as *const f32);
        // Re-fetch as_mut_ptr() each iter; set_len may have reallocated.
        _mm256_storeu_ps(out.as_mut_ptr().add(out.len()), v);
        out.set_len(out.len() + 8);
        i += 32;
    }
    // Tail: 0..3 f32s left.
    if i < bytes.len() {
        for c in bytes[i..].chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
    }
    if out.len() > max {
        out.truncate(max);
    }
    out
}}

// -- F16 -> F32 (8 per instruction via F16C `_mm256_cvtph_ps`) ---------------

#[target_feature(enable = "avx2,f16c")]
unsafe fn dequant_f16_avx2(bytes: &[u8], max: usize) -> Vec<f32> { unsafe {
    let n = bytes.len() / 2;
    let mut out: Vec<f32> = Vec::with_capacity(n);
    let mut i = 0;
    while i + 16 <= bytes.len() {
        let v_i16 = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);
        let v_f32 = _mm256_cvtph_ps(v_i16);
        _mm256_storeu_ps(out.as_mut_ptr().add(out.len()), v_f32);
        out.set_len(out.len() + 8);
        i += 16;
    }
    if i < bytes.len() {
        for c in bytes[i..].chunks_exact(2) {
            out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
        }
    }
    if out.len() > max {
        out.truncate(max);
    }
    out
}}

// -- BF16 -> F32 (zero-extend to i32, shift left 16, reinterpret) ------------

#[target_feature(enable = "avx2")]
unsafe fn dequant_bf16_avx2(bytes: &[u8], max: usize) -> Vec<f32> { unsafe {
    let n = bytes.len() / 2;
    let mut out: Vec<f32> = Vec::with_capacity(n);
    let mut i = 0;
    while i + 16 <= bytes.len() {
        let v_i16 = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);
        // Widen 8 u16 -> 8 i32 (zero-extend) and shift left 16 to make BF16 -> f32 bits.
        let v_i32 = _mm256_cvtepu16_epi32(v_i16);
        let v_shifted = _mm256_slli_epi32(v_i32, 16);
        let v_f32 = _mm256_castsi256_ps(v_shifted);
        _mm256_storeu_ps(out.as_mut_ptr().add(out.len()), v_f32);
        out.set_len(out.len() + 8);
        i += 16;
    }
    if i < bytes.len() {
        for c in bytes[i..].chunks_exact(2) {
            out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
        }
    }
    if out.len() > max {
        out.truncate(max);
    }
    out
}}

// -- Q4_0 (32 elements / 18 bytes) -------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn dequant_q4_0_avx2(bytes: &[u8], max: usize) -> Vec<f32> { unsafe {
    let n_blocks = bytes.len() / 18;
    let mut out: Vec<f32> = Vec::with_capacity(n_blocks * 32);
    let mask8 = _mm_set1_epi8(0x0F);
    let eight = _mm_set1_epi8(8);

    for blk in bytes.chunks_exact(18) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..18]; // 16 bytes

        // Load 16 bytes containing 32 packed 4-bit values.
        let v = _mm_loadu_si128(qs.as_ptr() as *const __m128i);

        // Split into low and high nibbles (16 nibbles in each).
        let lo = _mm_and_si128(v, mask8);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask8);

        // Interleave lo and hi to get the 32 nibbles as i8 lanes.
        let i_lo = _mm_unpacklo_epi8(lo, hi); // 16 i8s: first 16 elements
        let i_hi = _mm_unpackhi_epi8(lo, hi); // 16 i8s: next  16 elements

        // Subtract 8 (zero point).
        let c_lo = _mm_sub_epi8(i_lo, eight);
        let c_hi = _mm_sub_epi8(i_hi, eight);

        // Widen to i32, 8 at a time (need 4 calls to get all 32).
        let i32_0 = _mm256_cvtepi8_epi32(c_lo);
        let i32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(c_lo, 8));
        let i32_2 = _mm256_cvtepi8_epi32(c_hi);
        let i32_3 = _mm256_cvtepi8_epi32(_mm_srli_si128(c_hi, 8));

        // Convert to f32 and multiply by d.
        let d_v = _mm256_set1_ps(d);
        let f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_0), d_v);
        let f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_1), d_v);
        let f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_2), d_v);
        let f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_3), d_v);

        let base = out.as_mut_ptr().add(out.len());
        _mm256_storeu_ps(base, f0);
        _mm256_storeu_ps(base.add(8), f1);
        _mm256_storeu_ps(base.add(16), f2);
        _mm256_storeu_ps(base.add(24), f3);
        out.set_len(out.len() + 32);
    }
    if out.len() > max {
        out.truncate(max);
    }
    out
}}

// -- Q8_0 (32 elements / 34 bytes) -------------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn dequant_q8_0_avx2(bytes: &[u8], max: usize) -> Vec<f32> { unsafe {
    dequant_q8_block_avx2(bytes, max, 34, 2)
}}

// -- Q8_1 (32 elements / 36 bytes) -------------------------------------------
// Identical to Q8_0 except the block header is 4 bytes (d + sum) instead of 2.

#[target_feature(enable = "avx2")]
unsafe fn dequant_q8_1_avx2(bytes: &[u8], max: usize) -> Vec<f32> { unsafe {
    dequant_q8_block_avx2(bytes, max, 36, 4)
}}

/// Shared AVX2 dequant for Q8_0 / Q8_1 block formats.
/// `stride` is bytes per block (34 for Q8_0, 36 for Q8_1).
/// `qs_off` is offset to the byte array of 32 i8 quants (2 for Q8_0, 4 for Q8_1).
#[target_feature(enable = "avx2")]
unsafe fn dequant_q8_block_avx2(bytes: &[u8], max: usize, stride: usize, qs_off: usize) -> Vec<f32> { unsafe {
    let n_blocks = bytes.len() / stride;
    let mut out: Vec<f32> = Vec::with_capacity(n_blocks * 32);

    for blk in bytes.chunks_exact(stride) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[qs_off..qs_off + 32];
        let d_v = _mm256_set1_ps(d);

        let v_i8 = _mm256_loadu_si256(qs.as_ptr() as *const __m256i);
        let lo = _mm256_castsi256_si128(v_i8);
        let hi = _mm256_extracti128_si256(v_i8, 1);
        let lo_hi = _mm_srli_si128(lo, 8);
        let hi_hi = _mm_srli_si128(hi, 8);

        let f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), d_v);
        let f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_hi)), d_v);
        let f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), d_v);
        let f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_hi)), d_v);

        let base = out.as_mut_ptr().add(out.len());
        _mm256_storeu_ps(base, f0);
        _mm256_storeu_ps(base.add(8), f1);
        _mm256_storeu_ps(base.add(16), f2);
        _mm256_storeu_ps(base.add(24), f3);
        out.set_len(out.len() + 32);
    }
    if out.len() > max {
        out.truncate(max);
    }
    out
}}
