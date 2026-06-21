//! Q5_K quantizer: 256 elements per super-block, 8 sub-blocks of 32, 5-bit.
//!
//! On-disk: 2 bytes f16 d, 2 bytes f16 dmin, 12 bytes scales, 32 bytes qh,
//! 128 bytes qs. Total 176 B / 256 el. Dequant: x = d * sc1 * q - dmin * mn1.
//!
//! ## Scale table layout (12 bytes, canonical Q4_K / Q5_K)
//!
//! For sub-block j < 4:
//!   - sc = scales[j] & 0x3F            (low 6 bits of byte j)
//!   - mn = scales[4 + j] & 0x3F        (low 6 bits of byte 4+j)
//!
//! For sub-block j >= 4 (let s = j - 4):
//!   - sc = (scales[8 + s] & 0x0F) | ((scales[s] >> 6) << 4)
//!   - mn = (scales[8 + s] & 0x0F) | ((scales[4 + s] >> 6) << 4)
//!
//! The "shared low-4" trick: sc[4+s] and mn[4+s] share the same low
//! nibble from scales[8 + s], while their high 2 bits come from
//! scales[s] and scales[4 + s] respectively. The encoder must satisfy
//! sc[4+s] & 0x0F == mn[4+s] & 0x0F.
//!
//! ## qs + qh layout
//!
//! q is a 5-bit value in [0, 31]. Low 4 bits in qs (packed nibbles, same
//! as Q4_K), high bit in qh at bit position (j % 8) of byte qh[j / 8].

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const SUB: usize = 32;
const N_SUB: usize = QK_K / SUB;
const BLOCK_BYTES: usize = 176;

#[inline]
fn pack_qs(qs: &mut [u8; 128], j: usize, n: u8) {
    if j < QK_K / 2 {
        qs[j] |= n & 0x0F;
    } else {
        qs[j - QK_K / 2] |= (n & 0x0F) << 4;
    }
}

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(QK_K));
    let mut out = Vec::with_capacity(src.len() / QK_K * BLOCK_BYTES);
    for blk in src.chunks_exact(QK_K) {
        let (block_max, block_min) = blk
            .iter()
            .fold((f32::NEG_INFINITY, f32::INFINITY), |(mx, mn), &v| {
                (mx.max(v), mn.min(v))
            });
        let d = if block_max == 0.0 {
            0.0
        } else {
            block_max / 31.0
        };
        let dmin = if block_min >= 0.0 {
            0.0
        } else {
            -block_min / 31.0
        };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
        let inv_dmin = if dmin == 0.0 { 0.0 } else { 1.0 / dmin };

        let mut sc = [0u8; N_SUB];
        let mut mn = [0u8; N_SUB];
        for s in 0..N_SUB {
            let sub = &blk[s * SUB..(s + 1) * SUB];
            let (smax, smin) = sub
                .iter()
                .fold((f32::NEG_INFINITY, f32::INFINITY), |(mx, mn), &v| {
                    (mx.max(v), mn.min(v))
                });
            let range = smax - smin;
            let sc_f = (range * inv_d / 31.0).round();
            // If the sub-block has non-zero variation, ensure at least 1
            // grid step.
            sc[s] = if range > 0.0 {
                sc_f.clamp(1.0, 63.0) as u8
            } else {
                0
            };
            mn[s] = (-smin * inv_dmin).round().clamp(0.0, 63.0) as u8;
        }

        // Enforce coupling: sc[4+s] and mn[4+s] share low nibble.
        // Search all (lo, sc_h, mn_h) combos to minimize reconstruction error.
        for s in 4..N_SUB {
            if sc[s] == 0 && mn[s] == 0 {
                continue;
            }
            let sub = &blk[s * SUB..(s + 1) * SUB];
            let mut best_lo: u8 = 1;
            let mut best_sc_h: u8 = 0;
            let mut best_mn_h: u8 = 0;
            let mut best_err = f32::INFINITY;
            for lo in 1u8..=15u8 {
                for sc_h in 0u8..=3u8 {
                    for mn_h in 0u8..=3u8 {
                        let sc_val = lo | (sc_h << 4);
                        let mn_val = lo | (mn_h << 4);
                        let d_sc = d * sc_val as f32;
                        let m_sc = dmin * mn_val as f32;
                        let mut err_max = 0.0f32;
                        for &v in sub {
                            if d_sc == 0.0 {
                                continue;
                            }
                            let q = ((v + m_sc) / d_sc).round().clamp(0.0, 31.0);
                            let recon = d_sc * q - m_sc;
                            let e = (v - recon).abs();
                            if e > err_max {
                                err_max = e;
                            }
                        }
                        if err_max < best_err {
                            best_err = err_max;
                            best_lo = lo;
                            best_sc_h = sc_h;
                            best_mn_h = mn_h;
                        }
                    }
                }
            }
            sc[s] = best_lo | (best_sc_h << 4);
            mn[s] = best_lo | (best_mn_h << 4);
        }

        // Pack 12-byte scale table (canonical Q4_K / Q5_K format).
        let mut packed = [0u8; 12];
        for s in 0..4 {
            packed[s] = sc[s] | (((sc[4 + s] >> 4) & 0x03) << 6);
            packed[4 + s] = mn[s] | (((mn[4 + s] >> 4) & 0x03) << 6);
            packed[8 + s] = sc[4 + s] & 0x0F;
        }

        // Quantize qs + qh.
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        for j in 0..QK_K {
            let s = j / SUB;
            let d_sc = d * sc[s] as f32;
            let m_sc = dmin * mn[s] as f32;
            let denom = d_sc;
            let n = if denom == 0.0 {
                0.0
            } else {
                ((blk[j] + m_sc) / denom).round()
            };
            let n = n.clamp(0.0, 31.0) as u8;
            let lo = n & 0x0F;
            let hi = (n >> 4) & 0x01;
            pack_qs(&mut qs, j, lo);
            qh[j / 8] |= hi << (j % 8);
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16_bits(dmin).to_le_bytes());
        out.extend_from_slice(&packed);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Q5K, bytes, None).unwrap()
}
