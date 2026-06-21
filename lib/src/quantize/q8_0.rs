//! Q8_0 quantizer: 32 elements per block, symmetric, 8-bit.
//!
//! On-disk: 2 bytes f16 d, 32 bytes i8 qs. Total 34 B / 32 el.
//! Dequant: x = d * q.

use crate::formats::gguf::dequant::f16_to_f32;
use crate::quantize::f32_to_f16_bits;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 34;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK));
    let mut out = Vec::with_capacity(src.len() / BLOCK * BLOCK_BYTES);
    for blk in src.chunks_exact(BLOCK) {
        // d = max(|x|) / 127; zero if the block is all zero.
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for &v in blk {
            let q = (v * inv_d).round();
            let q = q.clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

/// Inverse of `quantize` (mirrors the dequant in `crate::formats::gguf::dequant`).
/// Exposed for round-trip tests; the real reader path goes through
/// `dequant::dequantize`.
#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * BLOCK);
    for blk in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..BLOCK {
            out.push(d * (blk[2 + j] as i8 as f32));
        }
    }
    out
}
