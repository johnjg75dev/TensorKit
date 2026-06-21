//! Q4_1 quantizer: 32 elements per block, affine, 4-bit.
//!
//! On-disk: 2 bytes f16 d, 2 bytes f16 m, 16 bytes qs (32 nibbles). Total 20 B / 32 el.
//! Dequant: x = d * n + m; n in [0, 15] so range is [m, m + 15d].

use crate::formats::gguf::dequant::f16_to_f32;
use crate::quantize::f32_to_f16_bits;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 20;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK));
    let mut out = Vec::with_capacity(src.len() / BLOCK * BLOCK_BYTES);
    for blk in src.chunks_exact(BLOCK) {
        let (lo, hi) = blk
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        let range = hi - lo;
        let d = if range == 0.0 { 0.0 } else { range / 15.0 };
        let m = lo;
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16_bits(m).to_le_bytes());
        for pair in 0..BLOCK / 2 {
            let lo_n = quant_one(blk[pair * 2], m, inv_d);
            let hi_n = quant_one(blk[pair * 2 + 1], m, inv_d);
            out.push(lo_n | (hi_n << 4));
        }
    }
    out
}

#[inline]
fn quant_one(v: f32, m: f32, inv_d: f32) -> u8 {
    let n = ((v - m) * inv_d).round();
    n.clamp(0.0, 15.0) as u8
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * BLOCK);
    for blk in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        for pair in 0..BLOCK / 2 {
            let q = blk[4 + pair];
            out.push(d * (q & 0x0F) as f32 + m);
            out.push(d * ((q >> 4) & 0x0F) as f32 + m);
        }
    }
    out
}
