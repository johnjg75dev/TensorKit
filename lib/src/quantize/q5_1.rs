//! Q5_1 quantizer: 32 elements per block, affine, 5-bit.
//!
//! On-disk: 2 bytes f16 d, 2 bytes f16 m, 4 bytes u32 qh, 16 bytes qs. Total 24 B / 32 el.
//! Element j in pair p = j / 2, slot s = j % 2:
//!   - qs[p] holds the low 4 bits of both (lo nibble = slot 0).
//!   - qh bit 2*p is high bit of slot 0, bit 2*p + 1 is high bit of slot 1.
//!     Dequant: x = d * n + m; n in [0, 31] so range is [m, m + 31d].

use crate::formats::gguf::dequant::f16_to_f32;
use crate::quantize::f32_to_f16_bits;

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 24;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK));
    let mut out = Vec::with_capacity(src.len() / BLOCK * BLOCK_BYTES);
    for blk in src.chunks_exact(BLOCK) {
        let (lo_v, hi_v) = blk
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        let range = hi_v - lo_v;
        let d = if range == 0.0 { 0.0 } else { range / 31.0 };
        let m = lo_v;
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut qs = [0u8; 16];
        let mut qh: u32 = 0;
        for j in 0..BLOCK {
            let n = quant_one(blk[j], m, inv_d) as u32;
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
        out.extend_from_slice(&f32_to_f16_bits(m).to_le_bytes());
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

#[inline]
fn quant_one(v: f32, m: f32, inv_d: f32) -> u8 {
    let n = ((v - m) * inv_d).round();
    n.clamp(0.0, 31.0) as u8
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * BLOCK);
    for blk in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        for pair in 0..BLOCK / 2 {
            let q = blk[8 + pair];
            let xh0 = ((qh >> (pair * 2)) & 1) << 4;
            let xh1 = ((qh >> (pair * 2 + 1)) & 1) << 4;
            out.push(((q & 0x0F) as f32 + xh0 as f32) * d + m);
            out.push((((q >> 4) & 0x0F) as f32 + xh1 as f32) * d + m);
        }
    }
    out
}
