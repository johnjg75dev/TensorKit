//! x86_64 SIMD quantization for hot quantizer types (Q4_0, Q8_0).
//!
//! Function-level `#[target_feature(enable = "avx2,fma")]` keeps these
//! callable from the dispatch site after a runtime feature check.
//! All intrinsics are inside `unsafe fn` bodies; this module is consumed
//! from the dispatch only after `is_x86_feature_detected!("avx2")`.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Try to quantize with AVX2+FMA. Returns `None` if the type isn't vectorized.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn try_quantize(src: &[f32], ty: GgmlType) -> Option<Vec<u8>> {
    Some(match ty {
        GgmlType::Q4_0 => unsafe { quantize_q4_0_avx2(src) },
        GgmlType::Q8_0 => unsafe { quantize_q8_0_avx2(src) },
        _ => return None,
    })
}

// -- helpers ----------------------------------------------------------------

/// Horizontal max of all 8 f32 lanes in a `__m256`.
#[target_feature(enable = "avx2")]
unsafe fn hmax_ps(v: __m256) -> f32 {
    let hi = _mm256_permute2f128_ps(v, v, 1);
    let mx = _mm256_max_ps(v, hi);
    let sh = _mm256_shuffle_ps(mx, mx, 0b10_11_00_01);
    let mx = _mm256_max_ps(mx, sh);
    let sh = _mm256_shuffle_ps(mx, mx, 0b01_00_11_10);
    let mx = _mm256_max_ps(mx, sh);
    _mm256_cvtss_f32(mx)
}

/// Max absolute value across 32 f32 values.
#[target_feature(enable = "avx2")]
unsafe fn max_abs_32(src: &[f32]) -> f32 {
    let sign_msk = _mm256_set1_ps(-0.0f32);
    let a0 = _mm256_andnot_ps(sign_msk, _mm256_loadu_ps(src.as_ptr()));
    let a1 = _mm256_andnot_ps(sign_msk, _mm256_loadu_ps(src.as_ptr().add(8)));
    let a2 = _mm256_andnot_ps(sign_msk, _mm256_loadu_ps(src.as_ptr().add(16)));
    let a3 = _mm256_andnot_ps(sign_msk, _mm256_loadu_ps(src.as_ptr().add(24)));
    let mx = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
    hmax_ps(mx)
}

// -- Q4_0 (32 elements / 18 bytes) -----------------------------------------
//
// AVX2+FMA is used for the hot path: max-abs reduction + scale/round/clamp.
// The final nibble pack is scalar (8 iters/block, not worth SIMD).

#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_q4_0_avx2(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(32));
    let n_blocks = src.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 18);

    for blk in src.chunks_exact(32) {
        let amax = max_abs_32(blk);
        let d = if amax == 0.0 { 0.0 } else { amax / 8.0 };
        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());

        if d == 0.0 {
            out.extend(std::iter::repeat_n(0u8, 16));
            continue;
        }

        let inv_d = 1.0 / d;
        let inv_bc = _mm256_set1_ps(inv_d);
        let eight = _mm256_set1_ps(8.0);
        let zero_i = _mm256_setzero_si256();
        let fifteen = _mm256_set1_epi32(15);

        macro_rules! quant_lane {
            ($ptr:expr) => {{
                let v = _mm256_loadu_ps($ptr);
                let scaled = _mm256_fmadd_ps(v, inv_bc, eight);
                let i = _mm256_cvtps_epi32(scaled);
                _mm256_min_epi32(_mm256_max_epi32(i, zero_i), fifteen)
            }};
        }

        let q0 = quant_lane!(blk.as_ptr());
        let q1 = quant_lane!(blk.as_ptr().add(8));
        let q2 = quant_lane!(blk.as_ptr().add(16));
        let q3 = quant_lane!(blk.as_ptr().add(24));

        // i32 -> u8 via pack (lane-scrambled), then fix lane order.
        let p01 = _mm256_packus_epi32(q0, q1);
        let p23 = _mm256_packus_epi32(q2, q3);
        let p = _mm256_packus_epi16(p01, p23);

        // Fix lane-scrambled byte order: extract 128-bit lanes, deinterleave
        // 32-bit groups, reassemble.
        let lo = _mm256_castsi256_si128(p);
        let hi = _mm256_extracti128_si256(p, 1);
        let i_lo = _mm_unpacklo_epi32(lo, hi);
        let i_hi = _mm_unpackhi_epi32(lo, hi);
        let fixed = _mm256_set_m128i(i_hi, i_lo);

        // Store and nibble-pack in linear order.
        let mut buf = [0u8; 32];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, fixed);
        let base = out.as_mut_ptr().add(out.len());
        for j in 0..16 {
            unsafe { *base.add(j) = buf[2 * j] | (buf[2 * j + 1] << 4); }
        }
        out.set_len(out.len() + 16);
    }
    out
}

// -- Q8_0 (32 elements / 34 bytes) -----------------------------------------

#[target_feature(enable = "avx2")]
unsafe fn quantize_q8_0_avx2(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(32));
    let n_blocks = src.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 34);

    for blk in src.chunks_exact(32) {
        let amax = max_abs_32(blk);
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());

        if d == 0.0 {
            out.extend(std::iter::repeat_n(0u8, 32));
            continue;
        }

        let inv_d = 1.0 / d;
        let inv_bc = _mm256_set1_ps(inv_d);
        let min_i8 = _mm256_set1_epi32(-128i32);
        let max_i8 = _mm256_set1_epi32(127);

        macro_rules! quant_lane_i8 {
            ($ptr:expr) => {{
                let v = _mm256_loadu_ps($ptr);
                let scaled = _mm256_mul_ps(v, inv_bc);
                let i = _mm256_cvtps_epi32(scaled);
                _mm256_min_epi32(_mm256_max_epi32(i, min_i8), max_i8)
            }};
        }

        let q0 = quant_lane_i8!(blk.as_ptr());
        let q1 = quant_lane_i8!(blk.as_ptr().add(8));
        let q2 = quant_lane_i8!(blk.as_ptr().add(16));
        let q3 = quant_lane_i8!(blk.as_ptr().add(24));

        // i32 -> i16 -> i8 with lane-scrambled order, then permute back.
        let p01 = _mm256_packs_epi32(q0, q1);
        let p23 = _mm256_packs_epi32(q2, q3);
        let p = _mm256_packs_epi16(p01, p23);

        let lo = _mm256_castsi256_si128(p);
        let hi = _mm256_extracti128_si256(p, 1);
        let i_lo = _mm_unpacklo_epi32(lo, hi);
        let i_hi = _mm_unpackhi_epi32(lo, hi);
        let fixed = _mm256_set_m128i(i_hi, i_lo);

        let mut buf = [0u8; 32];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, fixed);
        out.extend_from_slice(&buf);
    }
    out
}
