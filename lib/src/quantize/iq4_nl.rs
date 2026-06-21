//! IQ4_NL quantizer: 32 elements per block, 4-bit non-linear.
//!
//! Same on-disk layout as Q4_0 (18 bytes: f16 d + 16 bytes qs), but the
//! 4-bit values index into KVALUES_IQ4NL non-linear lookup table (which
//! includes negative values, so no -8 offset is needed).
//!
//! Dequant: x = d * KVALUES_IQ4NL[idx]

use crate::formats::gguf::dequant;
use crate::formats::gguf::dequant::lookup::KVALUES_IQ4NL;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const BLOCK_LEN: usize = 32;
const BLOCK_BYTES: usize = 18;

/// Find the nearest index in KVALUES_IQ4NL for a target value.
fn nearest_idx(target: f32) -> u8 {
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
        let block_max = blk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let block_min = blk.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let d_pos = if block_max <= 0.0 { 0.0 } else { block_max / 113.0 };
        let d_neg = if block_min >= 0.0 { 0.0 } else { -block_min / 127.0 };
        let d = d_pos.max(d_neg);
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; 16];
        for pair in 0..16 {
            let v0 = blk[pair * 2];
            let v1 = blk[pair * 2 + 1];
            let idx0 = if d == 0.0 {
                0
            } else {
                nearest_idx(v0 * inv_d)
            };
            let idx1 = if d == 0.0 {
                0
            } else {
                nearest_idx(v1 * inv_d)
            };
            qs[pair] = idx0 | (idx1 << 4);
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Iq4Nl, bytes, None).unwrap()
}

#[cfg(test)]
#[path = "../../tests/unit/quantize/iq4_nl.rs"]
mod tests;
