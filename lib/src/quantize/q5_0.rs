//! Q5_0 quantizer: 32 elements per block, symmetric, 5-bit.
//!
//! On-disk: 2 bytes f16 d, 4 bytes u32 qh, 16 bytes qs. Total 22 B / 32 el.
//!
//! Layout: element j in a block of 32 lives in pair p = j / 2 and slot s = j % 2.
//!   - qs[p] holds the low 4 bits of both elements of the pair (lo nibble = slot 0).
//!   - qh holds the high bit: bit 2*p is the high bit of slot 0, bit 2*p + 1
//!     is the high bit of slot 1.
//!     Dequant: x = d * (n - 16); n in [0, 31] so range is [-16d, 15d].

use crate::formats::gguf::dequant::f16_to_f32;
use crate::quantize::f32_to_f16_bits;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 22;
const MAX_VAL: f32 = 16.0;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK));
    let mut out = Vec::with_capacity(src.len() / BLOCK * BLOCK_BYTES);
    for blk in src.chunks_exact(BLOCK) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / MAX_VAL };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; 16];
        let mut qh: u32 = 0;
        for j in 0..BLOCK {
            let n = quant_one(blk[j], inv_d) as u32;
            let lo = (n & 0x0F) as u8;
            let hi = (n >> 4) & 0x01;
            let p = j / 2;
            let s = j % 2;
            if s == 0 {
                qs[p] |= lo;
            } else {
                qs[p] |= lo << 4;
            }
            qh |= hi << (2 * p + s);
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

#[inline]
fn quant_one(v: f32, inv_d: f32) -> u8 {
    let n = (v * inv_d + 16.0).round();
    n.clamp(0.0, 31.0) as u8
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * BLOCK);
    for blk in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        for pair in 0..BLOCK / 2 {
            let q = blk[6 + pair];
            let xh0 = ((qh >> (pair * 2)) & 1) << 4;
            let xh1 = ((qh >> (pair * 2 + 1)) & 1) << 4;
            out.push(d * ((q & 0x0F) as f32 - 16.0 + xh0 as f32));
            out.push(d * (((q >> 4) & 0x0F) as f32 - 16.0 + xh1 as f32));
        }
    }
    out
}
