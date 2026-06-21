//! Q2_K quantizer: 256 elements per super-block, 16 sub-blocks of 16 elements,
//! 2-bit per element.
//!
//! On-disk layout (82 bytes / 256 elements), matching llama.cpp `block_q2_K`:
//! - 64 bytes: qs[] — quantized values, each byte holds 4× 2-bit quants (0-3)
//! - 2 bytes: f16 d — super-block scale
//! - 16 bytes: scales[] — each byte packs:
//!   lower 4 bits = sub-block scale (sc)
//!   upper 4 bits = sub-block min (mn)
//!
//! Dequant formula: x = d * (q * sc - mn)

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const SUB: usize = 16;
const N_SUB: usize = QK_K / SUB;
const BLOCK_BYTES: usize = 82;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(QK_K));
    let mut out = Vec::with_capacity(src.len() / QK_K * BLOCK_BYTES);
    for blk in src.chunks_exact(QK_K) {
        let overall_max = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = if overall_max == 0.0 { 0.0 } else { overall_max / 3.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; 64];
        let mut scales = [0u8; N_SUB];

        for s in 0..N_SUB {
            let sub = &blk[s * SUB..(s + 1) * SUB];
            let (sub_max, sub_min) = sub.iter().fold(
                (f32::NEG_INFINITY, f32::INFINITY),
                |(mx, mn), &v| (mx.max(v), mn.min(v)),
            );
            let range = sub_max - sub_min;

            let (sc, mn) = if d == 0.0 {
                (0u8, 0u8)
            } else {
                let sc_f = range * inv_d / 3.0;
                let mn_f = -sub_min * inv_d;
                let sc = if range > 0.0 {
                    (sc_f.ceil() as u8).clamp(1, 15)
                } else {
                    1
                };
                let mn = (mn_f.round() as u8).clamp(0, 15);
                (sc, mn)
            };

            scales[s] = (mn << 4) | sc;

            let d_sc = d * sc as f32;
            let d_mn = d * mn as f32;
            let denom = d_sc;

            if denom == 0.0 {
                for j in 0..SUB {
                    let idx = s * SUB + j;
                    qs[idx / 4] |= 0 << ((idx % 4) * 2);
                }
            } else {
                for j in 0..SUB {
                    let idx = s * SUB + j;
                    let x = blk[idx];
                    let qf = ((x + d_mn) / denom).round();
                    let q = (qf as i32).clamp(0, 3) as u8;
                    qs[idx / 4] |= q << ((idx % 4) * 2);
                }
            }
        }

        out.extend_from_slice(&qs);
        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&scales);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Q2K, bytes, None).unwrap()
}