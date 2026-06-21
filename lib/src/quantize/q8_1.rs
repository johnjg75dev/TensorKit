//! Q8_1 quantizer: 32 elements per block, symmetric, 8-bit,
//! with an extra f16 sum field.
//!
//! On-disk: 2 bytes f16 d, 2 bytes f16 s, 32 bytes i8 qs.
//! Total 36 B / 32 el.  Dequant: x = d * q.

use crate::formats::gguf::dequant::f16_to_f32;
use crate::quantize::f32_to_f16_bits;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 36;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK));
    let mut out = Vec::with_capacity(src.len() / BLOCK * BLOCK_BYTES);
    for blk in src.chunks_exact(BLOCK) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        // placeholder for sum_s, written below
        let sum_offset = out.len();
        out.extend_from_slice(&[0u8; 2]);

        let mut sum_s: i32 = 0;
        for &v in blk {
            let q = (v * inv_d).round();
            let q = q.clamp(-128.0, 127.0) as i8;
            sum_s += q as i32;
            out.push(q as u8);
        }

        let sum_bytes = f32_to_f16_bits(sum_s as f32).to_le_bytes();
        out[sum_offset] = sum_bytes[0];
        out[sum_offset + 1] = sum_bytes[1];
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * BLOCK);
    for blk in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..BLOCK {
            out.push(d * (blk[4 + j] as i8 as f32));
        }
    }
    out
}