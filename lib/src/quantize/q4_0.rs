//! Q4_0 quantizer: 32 elements per block, symmetric, 4-bit.
//!
//! On-disk: 2 bytes f16 d, 16 bytes qs (32 nibbles, low first). Total 18 B / 32 el.
//! Dequant: x = d * (n - 8); n in [0, 15] so the representable range is
//! [-8d, 7d]. d = max(|x|) / 8 is the convention; values that exceed +7d clip.

use crate::formats::gguf::dequant::f16_to_f32;
use crate::quantize::f32_to_f16_bits;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 18;
const MAX_VAL: f32 = 8.0;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK));
    let mut out = Vec::with_capacity(src.len() / BLOCK * BLOCK_BYTES);
    for blk in src.chunks_exact(BLOCK) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / MAX_VAL };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for pair in 0..BLOCK / 2 {
            let lo = quant_one(blk[pair * 2], inv_d);
            let hi = quant_one(blk[pair * 2 + 1], inv_d);
            out.push(lo | (hi << 4));
        }
    }
    out
}

#[inline]
fn quant_one(v: f32, inv_d: f32) -> u8 {
    let n = (v * inv_d + 8.0).round();
    n.clamp(0.0, 15.0) as u8
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * BLOCK);
    for blk in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for pair in 0..BLOCK / 2 {
            let q = blk[2 + pair];
            out.push(d * ((q & 0x0F) as f32 - 8.0));
            out.push(d * (((q >> 4) & 0x0F) as f32 - 8.0));
        }
    }
    out
}
