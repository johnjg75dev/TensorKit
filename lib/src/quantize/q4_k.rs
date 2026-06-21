//! Q4_K quantizer: 256 elements per super-block, 8 sub-blocks of 32, 4-bit.
//!
//! On-disk: 2 bytes f16 d, 2 bytes f16 dmin, 12 bytes scales, 128 bytes qs.
//! Total 144 B / 256 el. Dequant: x = d * sc1 * q - dmin * mn1.
//!
//! ## Scale table layout (12 bytes, canonical llama.cpp Q4_K)
//!
//! For sub-blocks 0..3 (j < 4):
//!   - sc[j] = scales[j] & 0x3F           (low 6 bits of byte j)
//!   - mn[j] = scales[4 + j] & 0x3F       (low 6 bits of byte 4+j)
//!     The high 2 bits of scales[0..3] and scales[4..7] are "spillover" slots
//!     holding the high 2 bits of the next-half sc[4..7] and mn[4..7].
//!
//! For sub-blocks 4..7 (j >= 4), the "shared low-4" trick: the low 4 bits
//! of sc[j] and mn[j] are the SAME byte (scales[8 + (j-4)]), while the
//! high 2 bits come from scales[j-4] (for sc) and scales[j] (for mn):
//!   - sc[j] = (scales[8 + (j-4)] & 0x0F) | ((scales[j-4] >> 6) << 4)
//!   - mn[j] = (scales[8 + (j-4)] & 0x0F) | ((scales[j]   >> 6) << 4)
//!
//! This means the encoder must satisfy the coupling: the high 2 bits of
//! sc[4+j] are stored in the high 2 bits of scales[j] (the 6-bit slot for
//! sc[j]), and the high 2 bits of mn[4+j] are stored in the high 2 bits
//! of scales[4+j] (the 6-bit slot for mn[j]). For 6-bit sub-block values,
//! the high 2 bits are non-zero when the value exceeds 63, which the simple
//! per-sub-block range quantizer avoids; we still allow it for fidelity.

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const SUB: usize = 32;
const N_SUB: usize = QK_K / SUB;
const BLOCK_BYTES: usize = 144;

/// Encode a sub-block's 4-bit qs byte: j < 128 -> low nibble; j >= 128 ->
/// high nibble of qs[j - 128].
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
        // Pick d, dmin from the block: d covers the positive side, dmin
        // covers the negative side. With per-sub-block sc, mn, the
        // representable range is [-dmin*mn, d*sc*15 - dmin*mn].
        let (block_max, block_min) = blk
            .iter()
            .fold((f32::NEG_INFINITY, f32::INFINITY), |(mx, mn), &v| {
                (mx.max(v), mn.min(v))
            });
        let d = if block_max == 0.0 {
            0.0
        } else {
            block_max / 15.0
        };
        let dmin = if block_min >= 0.0 {
            0.0
        } else {
            -block_min / 15.0
        };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
        let inv_dmin = if dmin == 0.0 { 0.0 } else { 1.0 / dmin };

        // Per sub-block: pick (sc, mn) so the representable range covers
        // [sub_min, sub_max]. With q in [0, 15]:
        //   x = d*sc*q - dmin*mn
        // We want -dmin*mn ≤ sub_min and d*sc*15 - dmin*mn ≥ sub_max.
        // Tightest fit: -dmin*mn = sub_min and d*sc*15 = sub_max - sub_min.
        // So:
        //   sc[s] = (sub_max - sub_min) / (d * 15)
        //   mn[s] = -sub_min / dmin
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
            let sc_f = if d == 0.0 { 0.0 } else { range * inv_d / 15.0 };
            let mn_f = if dmin == 0.0 { 0.0 } else { -smin * inv_dmin };
            let sc_r = sc_f.round();
            // If the sub-block has non-zero variation, ensure at least 1
            // grid step (otherwise sc=0 collapses the representable range
            // and we lose the ability to distinguish values).
            sc[s] = if range > 0.0 {
                sc_r.clamp(1.0, 63.0) as u8
            } else {
                0
            };
            mn[s] = mn_f.round().clamp(0.0, 63.0) as u8;
        }

        // Enforce canonical Q4_K layout coupling: for sub-blocks 4..7, the
        // low 4 bits of sc[s] and mn[s] MUST be the same (they share one
        // byte), while the high 2 bits (bits 4-5) are stored independently
        // in the high 2 bits of the 6-bit scale slots for sub-blocks s-4
        // (sc) and s (mn). Search over all 15×4×4 = 240 combinations of
        // (shared_lo, sc_high_nibble, mn_high_nibble) to find the best fit.
        // The high nibble is only 2 bits (range 0..3) because sc, mn are
        // 6-bit values; packing shifts them into bits 6-7 of the first-half
        // scale bytes so that `get_scale_min_k4` recovers them correctly.
        for s in 4..N_SUB {
            if sc[s] == 0 && mn[s] == 0 {
                continue;
            }
            let lo_sc = sc[s] & 0x0F;
            let lo_mn = mn[s] & 0x0F;
            if lo_sc == lo_mn {
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
                            let q = ((v + m_sc) / d_sc).round().clamp(0.0, 15.0);
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

        let mut packed = [0u8; 12];
        for s in 0..4 {
            packed[s] = sc[s] | (((sc[4 + s] >> 4) & 0x03) << 6);
            packed[4 + s] = mn[s] | (((mn[4 + s] >> 4) & 0x03) << 6);
        }
        for s in 4..N_SUB {
            let j = s - 4;
            let lo = sc[s] & 0x0F;
            packed[8 + j] = lo | ((mn[s] & 0x0F) << 4);
        }

        // Quantize qs.
        let mut qs = [0u8; 128];
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
            let n = n.clamp(0.0, 15.0) as u8;
            pack_qs(&mut qs, j, n);
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16_bits(dmin).to_le_bytes());
        out.extend_from_slice(&packed);
        out.extend_from_slice(&qs);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Q4K, bytes, None).unwrap()
}
