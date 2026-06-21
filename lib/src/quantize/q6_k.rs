//! Q6_K quantizer: 256 elements per super-block, 16 sub-blocks of 16, 6-bit.
//!
//! On-disk: 2 bytes f16 d, 128 bytes ql, 64 bytes qh, 16 bytes i8 scales.
//! Total 210 B / 256 el. Dequant: x = d * s * q where s is the per-16 i8
//! scale and q is a 6-bit signed value in [-32, 31] = `(ql_low4 - 32) +
//! qh_2bit * 4`.
//!
//! Layout:
//!   - ql[j/2]: low nibble = element j's low 4 bits, high nibble = element
//!     (j+1)'s low 4 bits (for even j).
//!   - qh[j/4]: 2 bits per element, packed 4 per byte. Forms the high 2 bits
//!     of the 6-bit value.
//!   - sc[j/16]: i8 scale for the 16-element sub-block containing j.

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const SUB: usize = 16;
const N_SUB: usize = QK_K / SUB;
const BLOCK_BYTES: usize = 210;
const GRID_MAX: f32 = 31.0; // max positive representable (q in [-32, 31])

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(QK_K));
    let mut out = Vec::with_capacity(src.len() / QK_K * BLOCK_BYTES);
    for blk in src.chunks_exact(QK_K) {
        // Pick a shared d for the super-block such that the worst-case
        // sub-block fills d * 127 * 31. Then for each sub-block pick a
        // positive i8 scale s so the sub-block's range fits.
        let mut block_amax = 0.0f32;
        for s in 0..N_SUB {
            let sub = &blk[s * SUB..(s + 1) * SUB];
            let amax = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            if amax > block_amax {
                block_amax = amax;
            }
        }
        let d = if block_amax == 0.0 {
            0.0
        } else {
            block_amax / (127.0 * GRID_MAX)
        };

        let mut scales = [0i8; N_SUB];
        for s in 0..N_SUB {
            let sub = &blk[s * SUB..(s + 1) * SUB];
            let amax = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let s_f = if d == 0.0 { 0.0 } else { amax / (d * GRID_MAX) };
            // If the sub-block has non-zero magnitude, ensure at least 1
            // grid step (otherwise s=0 collapses the representable range).
            scales[s] = if amax > 0.0 {
                s_f.round().clamp(1.0, 127.0) as i8
            } else {
                0
            };
        }

        // Build ql, qh.
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for j in 0..QK_K {
            let sub = j / SUB;
            let s = scales[sub] as f32;
            let v = blk[j];
            // Round v / (d * s) to nearest 6-bit signed value in [-32, 31].
            let denom = d * s;
            let q = if denom == 0.0 {
                0.0
            } else {
                (v / denom).round()
            };
            let q = q.clamp(-32.0, 31.0) as i32;
            // Split: ql = low 4 of (q + 32), qh = high 2 of (q + 32).
            let biased = q + 32;
            let ql_v = (biased & 0x0F) as u8;
            let qh_v = ((biased >> 4) & 0x03) as u8;
            if j % 2 == 0 {
                ql[j / 2] |= ql_v;
            } else {
                ql[j / 2] |= ql_v << 4;
            }
            qh[j / 4] |= qh_v << ((j % 4) * 2);
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        for &s in &scales {
            out.push(s as u8);
        }
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Q6K, bytes, None).unwrap()
}
