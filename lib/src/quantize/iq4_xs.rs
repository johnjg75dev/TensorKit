//! IQ4_XS quantizer: 256 elements per super-block, 8 sub-blocks of 32, 4-bit.
//!
//! On-disk: 2 d (f16), 4 scales_l, 2 scales_h (u16), 128 qs = 136 bytes / 256 el.
//! Dequant: x = d * (ls - 32) * KVALUES_IQ4NL[idx]
//!
//! ## Scale packing (6 bits per sub-block Ã— 8 = 48 bits)
//!
//! Sub-block s has a 6-bit signed value ls âˆˆ [0, 63]; the effective scale is
//! (ls - 32). The low 4 bits of ls[s] are packed into `scales_l[s/2]` at
//! nibble position 4*(s%2). The high 2 bits are packed into `scales_h` at
//! bit position 2*s.
//!
//! | Range      | Bytes | Content                          |
//! |------------|-------|----------------------------------|
//! | [0,  2)    | 2     | d (f16)                          |
//! | [2,  6)    | 4     | scales_l (low nibbles Ã— 8)       |
//! | [6,  8)    | 2     | scales_h (high 2 bits Ã— 8 as u16)|
//! | [8, 136)   | 128   | qs (4-bit indices, 2 per byte)   |

use crate::formats::gguf::dequant;
use crate::formats::gguf::dequant::lookup::KVALUES_IQ4NL;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const BLOCK_LEN: usize = 256;
const SUB_LEN: usize = 32;
const N_SUB: usize = BLOCK_LEN / SUB_LEN;
const BLOCK_BYTES: usize = 136;

/// Find the nearest index in KVALUES_IQ4NL for a target value.
fn nearest_kvalue(target: f32) -> u8 {
    let mut best = 0u8;
    let mut best_dist = f32::MAX;
    for (i, &v) in KVALUES_IQ4NL.iter().enumerate() {
        let d = (target - v as f32).abs();
        if d < best_dist {
            best_dist = d;
            best = i as u8;
        }
    }
    best
}

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK_LEN));
    let mut out = Vec::with_capacity(src.len() / BLOCK_LEN * BLOCK_BYTES);
    for blk in src.chunks_exact(BLOCK_LEN) {
        let mut sub_max = [0.0f32; N_SUB];
        let mut block_max_abs = f32::NEG_INFINITY;
        let mut block_sum = 0.0f32;
        for s in 0..N_SUB {
            let sub = &blk[s * SUB_LEN..(s + 1) * SUB_LEN];
            let (mx, mn, sum) = sub.iter()
                .fold((f32::NEG_INFINITY, f32::INFINITY, 0.0f32), |(mx, mn, s), &v| {
                    (mx.max(v), mn.min(v), s + v)
                });
            let max_abs = mx.abs().max(mn.abs());
            sub_max[s] = max_abs;
            if max_abs > block_max_abs { block_max_abs = max_abs; }
            block_sum += sum;
        }

        let d_sign = if block_sum >= 0.0 { 1.0 } else { -1.0 };
        let d = if block_max_abs == 0.0 {
            0.0
        } else {
            d_sign * block_max_abs / (31.0 * 113.0)
        };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut ls = [0u8; N_SUB];
        for s in 0..N_SUB {
            if d == 0.0 || sub_max[s] == 0.0 {
                ls[s] = 32;
            } else {
                let ideal = (sub_max[s] * inv_d / 113.0 + 32.0).round() as i32;
                ls[s] = ideal.clamp(0, 63) as u8;
            }
        }

        let mut qs = [0u8; 128];
        for s in 0..N_SUB {
            let dl = d * (ls[s] as i32 - 32) as f32;
            let inv_dl = if dl == 0.0 { 0.0 } else { 1.0 / dl };
            let sub = &blk[s * SUB_LEN..(s + 1) * SUB_LEN];
            let qb = &mut qs[s * 16..(s + 1) * 16];
            for pair in 0..16 {
                let v0 = sub[pair * 2];
                let v1 = sub[pair * 2 + 1];
                let idx0 = if dl == 0.0 { 0 } else { nearest_kvalue(v0 * inv_dl) };
                let idx1 = if dl == 0.0 { 0 } else { nearest_kvalue(v1 * inv_dl) };
                qb[pair] = idx0 | (idx1 << 4);
            }
        }

        let mut scales_l = [0u8; 4];
        let mut scales_h: u16 = 0;
        for s in 0..N_SUB {
            let lo = ls[s] & 0x0F;
            let hi = (ls[s] >> 4) & 0x03;
            scales_l[s / 2] |= lo << (4 * (s % 2));
            scales_h |= (hi as u16) << (2 * s);
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&scales_l);
        out.extend_from_slice(&scales_h.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Iq4Xs, bytes, None).unwrap()
}

#[cfg(test)]
#[path = "../../tests/unit/quantize/iq4_xs.rs"]
mod tests;
