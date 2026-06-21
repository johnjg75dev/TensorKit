//! Scalar dequantization fallbacks. Used on non-x86_64 and when AVX2+F16C
//! aren't both available.
//!
//! All hot loops are marked `#[inline]` for the compiler; LLVM unrolls the
//! 32-element block loops aggressively at `-O3`.
//!
//! For multi-block types the parallel entry point `dequantize_par` splits
//! input bytes into per-block chunks and processes them in parallel via rayon.

use super::lookup::{IQ1S_DELTA, IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, KVALUES_IQ4NL};
use super::truncate_to;
use crate::formats::gguf::types::GgmlType;

pub fn dequantize(ty: GgmlType, bytes: &[u8], max: usize) -> Option<Vec<f32>> {
    Some(truncate_to(
        match ty {
            GgmlType::F32 => scan_f32(bytes),
            GgmlType::F16 => scan_f16(bytes),
            GgmlType::Bf16 => scan_bf16(bytes),
            GgmlType::Q4_0 => dequant_q4_0(bytes),
            GgmlType::Q4_1 => dequant_q4_1(bytes),
            GgmlType::Q5_0 => dequant_q5_0(bytes),
            GgmlType::Q5_1 => dequant_q5_1(bytes),
            GgmlType::Q8_0 => dequant_q8_0(bytes),
            GgmlType::Q8_1 => dequant_q8_1(bytes),
            GgmlType::Q4K => dequant_q4_k(bytes),
            GgmlType::Q5K => dequant_q5_k(bytes),
            GgmlType::Q6K => dequant_q6_k(bytes),
            GgmlType::Q8K => dequant_q8_k(bytes),
            GgmlType::Q2K => dequant_q2_k(bytes),
            GgmlType::Q3K => dequant_q3_k(bytes),
            GgmlType::Iq1S => dequant_iq1_s(bytes),
            GgmlType::Iq2Xxs => dequant_iq2_xxs(bytes),
            GgmlType::Iq2Xs => dequant_iq2_xs(bytes),
            GgmlType::Iq2S => dequant_iq2_s(bytes),
            GgmlType::Iq1M => dequant_iq1_m(bytes),
            GgmlType::Iq3Xxs => dequant_iq3_xxs(bytes),
            GgmlType::Iq3S => dequant_iq3_s(bytes),
            GgmlType::Iq4Nl => dequant_iq4_nl(bytes),
            GgmlType::Iq4Xs => dequant_iq4_xs(bytes),
            GgmlType::Tq1_0 => dequant_tq1_0(bytes),
            GgmlType::Tq2_0 => dequant_tq2_0(bytes),
            GgmlType::I8 => scan_i8(bytes),
            GgmlType::I16 => scan_i16(bytes),
            GgmlType::I32 => scan_i32(bytes),
            GgmlType::I64 => scan_i64(bytes),
            GgmlType::F64 => scan_f64(bytes),
            _ => return None,
        },
        max,
    ))
}

// -- scalar types -------------------------------------------------------------

#[inline]
pub fn scan_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    out
}

#[inline]
pub fn scan_f16(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
    }
    out
}

#[inline]
pub fn scan_bf16(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
    }
    out
}

/// IEEE 754 binary16 -> binary32. Marked inline so the SIMD path
/// (F16C `_mm256_cvtph_ps`) can replace all callers under `target_feature`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        let val = (mant as f32) / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        if mant == 0 {
            return f32::INFINITY * (if sign == 1 { -1.0 } else { 1.0 });
        }
        return f32::NAN;
    }
    let new_exp = (exp as i32 - 15 + 127) as u32;
    let bits32 = (sign << 31) | (new_exp << 23) | (mant << 13);
    f32::from_bits(bits32)
}

/// bfloat16 -> binary32. Pure left-shift, compiles to one instruction.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// -- Q4_0 ---------------------------------------------------------------------

#[inline]
pub fn dequant_q4_0(bytes: &[u8]) -> Vec<f32> {
    // 18 bytes / 32 elements: f16 d, 16 bytes of 4-bit quants
    let n_blocks = bytes.len() / 18;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(18) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for pair in 0..16usize {
            let q = blk[2 + pair];
            let x0 = (q & 0x0F) as i32 - 8;
            let x1 = ((q >> 4) & 0x0F) as i32 - 8;
            out.push(d * x0 as f32);
            out.push(d * x1 as f32);
        }
    }
    out
}

// -- Q4_1 ---------------------------------------------------------------------

#[inline]
pub fn dequant_q4_1(bytes: &[u8]) -> Vec<f32> {
    // 20 bytes / 32 elements: f16 d, f16 m, 16 bytes of 4-bit quants
    let n_blocks = bytes.len() / 20;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(20) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        for pair in 0..16usize {
            let q = blk[4 + pair];
            let x0 = (q & 0x0F) as f32;
            let x1 = ((q >> 4) & 0x0F) as f32;
            out.push(x0 * d + m);
            out.push(x1 * d + m);
        }
    }
    out
}

// -- Q5_0 ---------------------------------------------------------------------

#[inline]
pub fn dequant_q5_0(bytes: &[u8]) -> Vec<f32> {
    // 22 bytes / 32 elements: f16 d, u32 qh, 16 bytes qs
    let n_blocks = bytes.len() / 22;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(22) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        for pair in 0..16usize {
            let q = blk[6 + pair];
            let xh0 = ((qh >> (pair * 2)) & 1) << 4;
            let xh1 = ((qh >> (pair * 2 + 1)) & 1) << 4;
            let x0 = (q & 0x0F) as i32 | xh0 as i32;
            let x1 = ((q >> 4) & 0x0F) as i32 | xh1 as i32;
            out.push(d * (x0 as f32 - 16.0));
            out.push(d * (x1 as f32 - 16.0));
        }
    }
    out
}

// -- Q5_1 ---------------------------------------------------------------------

#[inline]
pub fn dequant_q5_1(bytes: &[u8]) -> Vec<f32> {
    // 24 bytes / 32 elements: f16 d, f16 m, u32 qh, 16 bytes qs
    let n_blocks = bytes.len() / 24;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(24) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        for pair in 0..16usize {
            let q = blk[8 + pair];
            let xh0 = ((qh >> (pair * 2)) & 1) << 4;
            let xh1 = ((qh >> (pair * 2 + 1)) & 1) << 4;
            let x0 = (q & 0x0F) as f32 + xh0 as f32;
            let x1 = ((q >> 4) & 0x0F) as f32 + xh1 as f32;
            out.push(x0 * d + m);
            out.push(x1 * d + m);
        }
    }
    out
}

// -- Q8_0 ---------------------------------------------------------------------

#[inline]
pub fn dequant_q8_0(bytes: &[u8]) -> Vec<f32> {
    // 34 bytes / 32 elements: f16 d, 32 i8 qs
    let n_blocks = bytes.len() / 34;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(34) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..32 {
            let q = blk[2 + j] as i8 as f32;
            out.push(d * q);
        }
    }
    out
}

// -- K-quants -----------------------------------------------------------------

const QK_K: usize = 256;

#[inline]
fn get_scale_min_k4(scales: &[u8; 12], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[4 + j] & 0x3F)
    } else {
        // Canonical llama.cpp Q4_K layout (ggml-quants.c):
        //   sc[j] = (scales[8 + (j-4)] & 0x0F) | ((scales[j-4] >> 6) << 4)
        //   mn[j] = (scales[8 + (j-4)] & 0x0F) | ((scales[j]   >> 6) << 4)
        // The "shared low-4" trick: the low 4 bits of sc[j] and mn[j] are
        // the same byte (scales[8 + (j-4)]), while the high 2 bits come
        // from scales[j-4] (for sc) and scales[j] (for mn). This means
        // the encoder must satisfy the coupling: the 2-bit high part of
        // sc[j] is stored in the 6-bit slot of sc[j-4] and the 2-bit high
        // part of mn[j] is stored in the 6-bit slot of mn[j-4].
        let lbits_idx = 8 + (j - 4);
        let lo = scales[lbits_idx] & 0x0F;
        let sc = lo | ((scales[j - 4] >> 6) << 4);
        let mn = lo | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

#[inline]
pub fn dequant_q4_k(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / 144;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    let mut sc = [0u8; 12];
    for blk in bytes.chunks_exact(144) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        sc.copy_from_slice(&blk[4..16]);
        let qs = &blk[16..144];
        for j in 0..QK_K {
            let sub = j / 32;
            let (sc1, mn1) = get_scale_min_k4(&sc, sub);
            let d_sc = d * sc1 as f32;
            let m_sc = dmin * mn1 as f32;
            let q = if j < QK_K / 2 {
                (qs[j] & 0x0F) as f32
            } else {
                ((qs[j - QK_K / 2] >> 4) & 0x0F) as f32
            };
            out.push(d_sc * q - m_sc);
        }
    }
    out
}

#[inline]
pub fn dequant_q5_k(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / 176;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    let mut sc = [0u8; 12];
    for blk in bytes.chunks_exact(176) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        sc.copy_from_slice(&blk[4..16]);
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        for j in 0..QK_K {
            let sub = j / 32;
            let (sc1, mn1) = get_scale_min_k4(&sc, sub);
            let d_sc = d * sc1 as f32;
            let m_sc = dmin * mn1 as f32;
            let h = (qh[j / 8] >> (j % 8)) & 1;
            let q = if j < QK_K / 2 {
                (qs[j] & 0x0F) | (h << 4)
            } else {
                ((qs[j - QK_K / 2] >> 4) & 0x0F) | (h << 4)
            };
            out.push(d_sc * q as f32 - m_sc);
        }
    }
    out
}

#[inline]
pub fn dequant_q6_k(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / 210;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(210) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let ql = &blk[2..130];
        let qh = &blk[130..194];
        // 16 i8 scales, one per 16-element sub-block of the 256-element super-block.
        let sc = &blk[194..210];
        for j in 0..QK_K {
            let ql_byte = ql[j / 2];
            let ql_val = if j % 2 == 0 {
                ql_byte & 0x0F
            } else {
                (ql_byte >> 4) & 0x0F
            };
            let qh_byte = qh[j / 4];
            let shift = (j % 4) * 2;
            let qh_val = (qh_byte >> shift) & 0x03;
            let q = ((ql_val | (qh_val << 4)) as i32) - 32;
            let s = sc[j / 16] as i8 as f32;
            out.push(d * s * q as f32);
        }
    }
    out
}

#[inline]
pub fn dequant_q8_k(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / 292;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(292) {
        let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        for j in 0..QK_K {
            let q = blk[4 + j] as i8 as f32;
            out.push(d * q);
        }
    }
    out
}

// -- Q8_1 ---------------------------------------------------------------------

/// Q8_1: 36 bytes / 32 elements.  f16 d, f16 _sum, 32× u8 quants.
/// Dequant: `xi = d * qi` where qi is the u8 quant (signed via i8 cast).
#[inline]
pub fn dequant_q8_1(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 36;
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        // blk[2..4] is f16 sum (unused in dequant)
        for j in 0..32 {
            let q = blk[4 + j] as i8 as f32;
            out.push(d * q);
        }
    }
    out
}

// -- Q2_K ---------------------------------------------------------------------

/// Q2_K: 82 bytes / 256 elements.
///
/// Layout per super-block (llama.cpp `block_q2_K`):
/// ```text
///   qs[64]       — 64 bytes, each holding 4× 2-bit quants
///   d            — 2 bytes f16 super-block scale
///   scales[16]   — 16 bytes; each byte packs:
///                    low 4 bits = sub-block scale
///                    high 4 bits = sub-block min
/// ```
///
/// Dequant formula: `d * (q * sc - mn)` where
/// `q` is the 2-bit quant value, `sc` and `mn` are per-sub-block.
#[inline]
pub fn dequant_q2_k(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 82; // 64 qs + 2 d + 16 scales
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let qs = &blk[0..64];
        let d = f16_to_f32(u16::from_le_bytes([blk[64], blk[65]]));
        let scales = &blk[66..82]; // 16 bytes

        let mut block_out = [0.0f32; QK_K];
        let mut is = 0usize;
        // Two halves of 128 elements each, using qs[0..32] and qs[32..64].
        for half in 0..2 {
            let q_base = half * 32;
            let mut shift = 0usize;
            for _j in 0..4 {
                // First sub-block of 16 elements.
                {
                    let sc = scales[is];
                    let dl = d * (sc & 0x0F) as f32; // lower nibble = scale
                    let ml = d * (sc >> 4) as f32; // upper nibble = min
                    is += 1;
                    for l in 0..16 {
                        let qi = ((qs[q_base + l] >> shift) & 3) as f32;
                        let idx = half * 128 + _j * 32 + l;
                        block_out[idx] = dl * qi - ml;
                    }
                }
                // Second sub-block of 16 elements.
                {
                    let sc = scales[is];
                    let dl = d * (sc & 0x0F) as f32;
                    let ml = d * (sc >> 4) as f32;
                    is += 1;
                    for l in 0..16 {
                        let qi = ((qs[q_base + 16 + l] >> shift) & 3) as f32;
                        let idx = half * 128 + _j * 32 + 16 + l;
                        block_out[idx] = dl * qi - ml;
                    }
                }
                shift += 2;
            }
        }
        out.extend_from_slice(&block_out);
    }
    out
}

// -- Q3_K ---------------------------------------------------------------------

/// Q3_K: 110 bytes / 256 elements.
///
/// Layout per super-block (llama.cpp `block_q3_K`):
/// ```text
///   hmask[32]    — 32 bytes, 1-bit high parts (256 bits)
///   qs[64]       — 64 bytes, 2-bit low parts (4 quants per byte)
///   d            — 2 bytes f16 super-block scale
///   scales[12]   — 12-byte scale table (6-bit sc + 2-bit high packed, like Q4_K)
/// ```
///
/// Dequant formula: `d * q * sc` where `q` is the 3-bit quant
/// (2 low bits from qs, 1 high bit from hmask) and `sc` is the sub-block scale.
#[inline]
pub fn dequant_q3_k(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 110; // 32 hmask + 64 qs + 2 d + 12 scales
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    let mut sc_buf = [0u8; 12];
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let d = f16_to_f32(u16::from_le_bytes([blk[96], blk[97]]));
        sc_buf.copy_from_slice(&blk[98..110]);

        for j in 0..QK_K {
            let sub = j / 32; // 8 sub-blocks of 32 elements
            let sc = get_q3_k_scale(&sc_buf, sub);
            let d_sc = d * sc as f32;

            // Low 2 bits from qs (packed like Q4_K: lower nibble first half, upper nibble second half)
            let ql = if j < QK_K / 2 {
                (qs[j] & 0x0F) as i32
            } else {
                ((qs[j - QK_K / 2] >> 4) & 0x0F) as i32
            };
            // High 1 bit from hmask
            let h = ((hmask[j / 8] >> (j % 8)) & 1) as i32;
            let q = ql - 16 + h * 2;
            out.push(d_sc * q as f32);
        }
    }
    out
}

/// Extract the scale for sub-block `j` (0..7) from a Q3_K 12-byte scale table.
/// Q3_K scales use the same "shared high-2" packing as Q5_K:
///   sc[j] = (scales[j] & 0x3F) | ((scales[8 + (j >> 2)] >> ((j & 3) * 2)) & 3) << 6
#[inline]
fn get_q3_k_scale(scales: &[u8; 12], j: usize) -> u8 {
    let lo = scales[j] & 0x3F;
    let hi = (scales[8 + (j >> 2)] >> ((j & 3) * 2)) & 3;
    lo | (hi << 6)
}

// -- I8 / I16 / I32 / I64 / F64 scalar types --------------------------------

#[inline]
pub fn scan_i8(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| (b as i8) as f32).collect()
}

#[inline]
pub fn scan_i16(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        out.push(i16::from_le_bytes([c[0], c[1]]) as f32);
    }
    out
}

#[inline]
pub fn scan_i32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32);
    }
    out
}

#[inline]
pub fn scan_i64(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for c in bytes.chunks_exact(8) {
        out.push(i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32);
    }
    out
}

#[inline]
pub fn scan_f64(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for c in bytes.chunks_exact(8) {
        out.push(f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32);
    }
    out
}

// -- IQ4_NL (4-bit, 32 elements/block) --------------------------------------

/// IQ4_NL: 18 bytes / 32 elements. Same block layout as Q4_0 but
/// the 16 4-bit values index into the KVALUES_IQ4NL non-linear table
/// (which includes negative values, so no -8 offset is needed).
#[inline]
pub fn dequant_iq4_nl(bytes: &[u8]) -> Vec<f32> {
    let n_blocks = bytes.len() / 18;
    let mut out = Vec::with_capacity(n_blocks * 32);
    for blk in bytes.chunks_exact(18) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..16 {
            let lo = (blk[2 + j] & 0x0F) as usize;
            let hi = ((blk[2 + j] >> 4) & 0x0F) as usize;
            out.push(d * KVALUES_IQ4NL[lo] as f32);
            out.push(d * KVALUES_IQ4NL[hi] as f32);
        }
    }
    out
}

// -- IQ4_XS (4-bit, 256 elements/block) -------------------------------------

/// IQ4_XS block layout: 2 bytes d (f16), 4 bytes scales_h/scales_l (2 bytes each),
/// 128 bytes of qs (16 sub-blocks × 16 bytes of 4-bit quants indexed into KVALUES_IQ4NL).
/// Block size: 2 + 4 + 128 = 134 bytes per 256 elements. Wait — actually
/// the IQ4_XS block is 136 bytes. Per-block: 2 d, 2 scales_l, 1 scales_h, 1 unused, 128 qs.
#[inline]
pub fn dequant_iq4_xs(bytes: &[u8]) -> Vec<f32> {
    // IQ4_XS block: d[2] scales_l[4] scales_h[2] qs[128] = 136 bytes
    const BLOCK_SIZE: usize = 136;
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let scales_l = &blk[2..6];
        let scales_h = u16::from_le_bytes([blk[6], blk[7]]);
        let qs = &blk[8..136];
        for ib in 0..8 {
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) as u16 & 0x0F)
                | (((scales_h >> (2 * ib)) & 0x03) << 4);
            let dl = d * (ls as i32 - 32) as f32;
            let qb = &qs[ib * 16..ib * 16 + 16];
            for j in 0..16 {
                let lo = (qb[j] & 0x0F) as usize;
                let hi = ((qb[j] >> 4) & 0x0F) as usize;
                out.push(dl * KVALUES_IQ4NL[lo] as f32);
                out.push(dl * KVALUES_IQ4NL[hi] as f32);
            }
        }
    }
    out
}

// -- IQ2_XXS (256 elements/block) -------------------------------------------

/// IQ2_XXS block layout: 2 d (f16), 64 qs bytes = 66 bytes / 256 elements.
/// qs is read as 32 uint32s: each uint32 packs (idx_low[9b] in aux8[0..1],
/// idx_high packed bits, plus the scale/sigs in aux32[1]).
/// Reference: llama.cpp `dequantize_row_iq2_xxs`.
#[inline]
pub fn dequant_iq2_xxs(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 66; // 2 d + 64 qs
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66];
        for ib32 in 0..8 {
            let aux32_off = ib32 * 8;
            let a0 = u32::from_le_bytes([qs[aux32_off], qs[aux32_off + 1], qs[aux32_off + 2], qs[aux32_off + 3]]);
            let a1 = u32::from_le_bytes([qs[aux32_off + 4], qs[aux32_off + 5], qs[aux32_off + 6], qs[aux32_off + 7]]);
            let db = d * (0.5 + ((a1 >> 28) as f32)) * 0.25;
            for l in 0..4 {
                let grid_off = ((a0 >> (8 * l)) & 0xFF) as usize;
                let grid = unsafe {
                    std::slice::from_raw_parts(
                        (IQ2XXS_GRID.as_ptr() as *const u8).add(grid_off),
                        8,
                    )
                };
                let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 0x7F) as usize];
                for j in 0..8 {
                    let v = grid[j] as f32;
                    let sgn = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out.push(db * v * sgn);
                }
            }
        }
    }
    out
}

// -- IQ2_XS (256 elements/block) -------------------------------------------

/// IQ2_XS block layout: 2 d (f16), 8 scales, 64 qs = 74 bytes / 256 elements.
#[inline]
pub fn dequant_iq2_xs(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 74; // 2 d + 8 scales + 64 qs
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let scales = &blk[2..10];
        let qs = &blk[10..74];
        let mut db = [0.0f32; 2];
        for ib32 in 0..8 {
            db[0] = d * (0.5 + (scales[ib32] & 0x0F) as f32) * 0.25;
            db[1] = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
            for l in 0..4 {
                let q_byte = qs[ib32 * 4 + l];
                let grid_off = ((q_byte as u16) & 0x1FF) as usize;
                let signs = KSIGNS_IQ2XS[((q_byte as u16) >> 9) as usize];
                let grid = unsafe {
                    std::slice::from_raw_parts(
                        (IQ2XS_GRID.as_ptr() as *const u8).add(grid_off),
                        8,
                    )
                };
                let dl = db[l / 2];
                for j in 0..8 {
                    let v = grid[j] as f32;
                    let sgn = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out.push(dl * v * sgn);
                }
            }
        }
    }
    out
}

// -- IQ3_XXS (256 elements/block) ------------------------------------------

/// IQ3_XXS block layout: 2 d (f16), 64 qs (which also embeds scales/signs
/// in the upper 4 bits of every 4th byte group) = 66 bytes / 256 elements.
#[inline]
pub fn dequant_iq3_xxs(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 98; // 2 d + 96 qs (qs+scales_and_signs)
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66];
        let scs = &blk[66..98];
        for ib32 in 0..8 {
            let aux32 = u32::from_le_bytes([scs[ib32 * 4], scs[ib32 * 4 + 1], scs[ib32 * 4 + 2], scs[ib32 * 4 + 3]]);
            let db = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 0x7F) as usize];
                let g1 = qs[ib32 * 8 + 2 * l] as usize;
                let g2 = qs[ib32 * 8 + 2 * l + 1] as usize;
                let grid1 = unsafe {
                    std::slice::from_raw_parts(
                        (IQ3XXS_GRID.as_ptr() as *const u8).add(g1),
                        4,
                    )
                };
                let grid2 = unsafe {
                    std::slice::from_raw_parts(
                        (IQ3XXS_GRID.as_ptr() as *const u8).add(g2),
                        4,
                    )
                };
                for j in 0..4 {
                    let s1 = if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    out.push(db * (grid1[j] as f32) * s1);
                    out.push(db * (grid2[j] as f32) * s2);
                }
            }
        }
    }
    out
}

// -- IQ3_S (256 elements/block) --------------------------------------------

/// IQ3_S block layout: 2 d (f16), 64 qs, 8 qh, 32 signs, 4 scales = 110 bytes.
/// Matches ggml-common.h block_iq3_s (d, qs, qh, signs, scales).
#[inline]
pub fn dequant_iq3_s(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 110;
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66];
        let qh = &blk[66..74];
        let signs = &blk[74..106];
        let scales = &blk[106..110];
        for ib32 in (0..8).step_by(2) {
            let db1 = d * (1.0 + 2.0 * (scales[ib32 / 2] & 0x0F) as f32);
            let db2 = d * (1.0 + 2.0 * (scales[ib32 / 2] >> 4) as f32);
            for l in 0..4 {
                let g1 = (qs[ib32 * 8 + 2 * l] as usize)
                    | (((qh[ib32] as u16) << (8 - 2 * l) & 0x100) as usize);
                let g2 = (qs[ib32 * 8 + 2 * l + 1] as usize)
                    | (((qh[ib32] as u16) << (7 - 2 * l) & 0x100) as usize);
                let grid1 = unsafe {
                    std::slice::from_raw_parts(
                        (IQ3S_GRID.as_ptr() as *const u8).add(g1),
                        4,
                    )
                };
                let grid2 = unsafe {
                    std::slice::from_raw_parts(
                        (IQ3S_GRID.as_ptr() as *const u8).add(g2),
                        4,
                    )
                };
                let s_byte = signs[ib32 * 4 + l];
                for j in 0..4 {
                    let s1 = if s_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s2 = if s_byte & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    out.push(db1 * (grid1[j] as f32) * s1);
                    out.push(db1 * (grid2[j] as f32) * s2);
                }
            }
            for l in 0..4 {
                let g1 = (qs[(ib32 + 1) * 8 + 2 * l] as usize)
                    | (((qh[ib32 + 1] as u16) << (8 - 2 * l) & 0x100) as usize);
                let g2 = (qs[(ib32 + 1) * 8 + 2 * l + 1] as usize)
                    | (((qh[ib32 + 1] as u16) << (7 - 2 * l) & 0x100) as usize);
                let grid1 = unsafe {
                    std::slice::from_raw_parts(
                        (IQ3S_GRID.as_ptr() as *const u8).add(g1),
                        4,
                    )
                };
                let grid2 = unsafe {
                    std::slice::from_raw_parts(
                        (IQ3S_GRID.as_ptr() as *const u8).add(g2),
                        4,
                    )
                };
                let s_byte = signs[(ib32 + 1) * 4 + l];
                for j in 0..4 {
                    let s1 = if s_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    let s2 = if s_byte & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                    out.push(db2 * (grid1[j] as f32) * s1);
                    out.push(db2 * (grid2[j] as f32) * s2);
                }
            }
        }
    }
    out
}

// -- IQ1_S (256 elements/block) --------------------------------------------

/// IQ1_S block layout: 2 d (f16), 16 qh, 8 qs_padding + 64 qs = ~66 bytes / 256 elements.
/// Actually: 2 d + 16 qh + 8 padding/qs_low = 26 bytes plus 32 bytes more
/// (qs in 8 groups of 4 bytes for 256 elements). Final block = 50 bytes.
#[inline]
pub fn dequant_iq1_s(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 50; // 2 d + 16 qh + 32 qs
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh_bytes = &blk[2..18];
        let qh: &[u16] = unsafe {
            std::slice::from_raw_parts(qh_bytes.as_ptr() as *const u16, 8)
        };
        let qs = &blk[18..50];
        for ib in 0..8 {
            let dl = d * (2.0 * ((qh[ib] >> 12) & 7) as f32 + 1.0);
            let delta = if qh[ib] & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            for l in 0..4 {
                let grid_idx = (qs[ib * 4 + l] as usize) | (((qh[ib] >> (3 * l)) & 7) << 8) as usize;
                let grid = unsafe {
                    std::slice::from_raw_parts(
                        (IQ1S_GRID.as_ptr() as *const i8).add(grid_idx),
                        8,
                    )
                };
                for j in 0..8 {
                    out.push(dl * (grid[j] as f32 + delta));
                }
            }
        }
    }
    out
}

// -- IQ2_S (256 elements/block) --------------------------------------------

/// IQ2_S block layout: 82 bytes / 256 elements
///   d[2] + qs[64] + qh[8] + scales[8]
/// qs stores grid indices packed as pairs: low_byte | (high_byte << 8) & 0x3ff
/// qh stores sign bits (1 byte per 8-element sub-group)
/// scales stores 2 × 4-bit per-sub-block scales
#[inline]
pub fn dequant_iq2_s(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 82;
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66];
        let qh = &blk[66..74];
        let scales = &blk[74..82];
        for ib32 in 0..8 {
            let db0 = d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25;
            let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
            for l in 0..4 {
                let dl = if l < 2 { db0 } else { db1 };
                let idx = (qs[ib32 * 4 + l] as u16)
                    | ((qh[ib32] as u16) << (8 - 2 * l) & 0x300);
                let grid = unsafe {
                    std::slice::from_raw_parts(
                        (IQ2S_GRID.as_ptr() as *const u8).add(idx as usize),
                        8,
                    )
                };
                let signs_byte = qs[32 + ib32 * 4 + l];
                for j in 0..8 {
                    let sgn = if signs_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                    out.push(dl * grid[j] as f32 * sgn);
                }
            }
        }
    }
    out
}

// -- IQ1_M (256 elements/block) --------------------------------------------

/// IQ1_M block layout: 56 bytes / 256 elements (NO separate d field)
///   qs[32] + qh[16] + scales[8]
/// d is packed as a 12-bit FP16 spread across the high bits of 4 u16 in scales[0..8]
/// scale unpack: (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
/// Uses the same IQ1S_GRID as IQ1_S (reuses 2048-entry codebook).
#[inline]
pub fn dequant_iq1_m(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 56;
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let qs = &blk[0..32];
        let qh = &blk[32..48];
        let scales_bytes = &blk[48..56];
        let sc_u16 = unsafe {
            std::slice::from_raw_parts(scales_bytes.as_ptr() as *const u16, 4)
        };
        let d_bits = (sc_u16[0] >> 12)
            | ((sc_u16[1] >> 8) & 0x00f0)
            | ((sc_u16[2] >> 4) & 0x0f00)
            | (sc_u16[3] & 0xf000);
        let d = f16_to_f32(d_bits);
        for ib in 0..8 {
            let dl1 = d * (2.0 * ((sc_u16[ib / 2] >> (6 * (ib % 2))) & 0x7) as f32 + 1.0);
            let dl2 = d * (2.0 * ((sc_u16[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0);
            let idx0 = qs[ib * 4] as u16 | ((qh[ib * 2] as u16) << 8) & 0x700;
            let idx1 = qs[ib * 4 + 1] as u16 | ((qh[ib * 2] as u16) << 4) & 0x700;
            let idx2 = qs[ib * 4 + 2] as u16 | ((qh[ib * 2 + 1] as u16) << 8) & 0x700;
            let idx3 = qs[ib * 4 + 3] as u16 | ((qh[ib * 2 + 1] as u16) << 4) & 0x700;
            let delta0 = if qh[ib * 2] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            let delta1 = if qh[ib * 2] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            let delta2 = if qh[ib * 2 + 1] & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            let delta3 = if qh[ib * 2 + 1] & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            for l in 0..2 {
                let (idx, delta, dl) = if l == 0 {
                    (idx0, delta0, dl1)
                } else {
                    (idx1, delta1, dl1)
                };
                let grid = unsafe {
                    std::slice::from_raw_parts(
                        (IQ1S_GRID.as_ptr() as *const i8).add(idx as usize),
                        8,
                    )
                };
                for j in 0..8 {
                    out.push(dl * (grid[j] as f32 + delta));
                }
            }
            for l in 2..4 {
                let (idx, delta, dl) = if l == 2 {
                    (idx2, delta2, dl2)
                } else {
                    (idx3, delta3, dl2)
                };
                let grid = unsafe {
                    std::slice::from_raw_parts(
                        (IQ1S_GRID.as_ptr() as *const i8).add(idx as usize),
                        8,
                    )
                };
                for j in 0..8 {
                    out.push(dl * (grid[j] as f32 + delta));
                }
            }
        }
    }
    out
}

// -- TQ1_0 (256 elements/block) --------------------------------------------

/// TQ1_0: ternary quantization with delta steps of 1/3. Block layout:
/// 2 d (f16), 32 qs (each u8 * 3^shift → 0..1..2 in steps), 4 qh (high groups).
/// Total block = 38 bytes / 256 elements.
#[inline]
pub fn dequant_tq1_0(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 38; // 2 d + 32 qs + 4 qh
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    let pow3: [u8; 5] = [1, 3, 9, 27, 81];
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..34];
        let qh = &blk[34..38];
        // First 32 bytes of qs: 5 ternary expansions × 32 elements = 160 elements
        for j in 0..32 {
            for n in 0..5 {
                let q = qs[j].wrapping_mul(pow3[n]);
                let xi = ((q as u16).wrapping_mul(3) >> 8) as i16;
                out.push((xi as f32 - 1.0) * d);
            }
        }
        // Last 4 bytes of qh: 4 ternary expansions × 4 elements = 16 elements
        for n in 0..4 {
            for j in 0..4 {
                let q = qh[j].wrapping_mul(pow3[n]);
                let xi = ((q as u16).wrapping_mul(3) >> 8) as i16;
                out.push((xi as f32 - 1.0) * d);
            }
        }
        // Remaining elements: any leftover (none for full blocks)
        let _ = qs;
    }
    out
}

// -- TQ2_0 (256 elements/block) --------------------------------------------

/// TQ2_0: 2-bit ternary (values -1, 0, 1). Block layout:
/// 2 d (f16), 64 qs (each byte holds 4 × 2-bit quants) = 66 bytes / 256 elements.
#[inline]
pub fn dequant_tq2_0(bytes: &[u8]) -> Vec<f32> {
    const BLOCK_SIZE: usize = 66; // 2 d + 64 qs
    let n_blocks = bytes.len() / BLOCK_SIZE;
    let mut out = Vec::with_capacity(n_blocks * QK_K);
    for blk in bytes.chunks_exact(BLOCK_SIZE) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66];
        for j in 0..64 {
            for l in 0..4 {
                let q = ((qs[j] >> (l * 2)) & 3) as i8;
                out.push((q as f32 - 1.0) * d);
            }
        }
    }
    out
}



