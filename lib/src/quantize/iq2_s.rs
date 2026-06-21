//! IQ2_S quantizer: 256 elements per super-block.
//! Grid values: {8, 25, 43}, unsigned bytes packed as 8 bytes per u64 entry.
//! On-disk (82 bytes): d (f16) + qs[64] + qh[8] + scales[8].

use crate::formats::gguf::dequant;
use crate::formats::gguf::dequant::lookup::{IQ2S_GRID, KMASK_IQ2XS};
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 82;
const MAX_GRID_VAL: f32 = 43.0;

fn grid_flat() -> &'static [u8] {
    unsafe {
        std::slice::from_raw_parts(
            IQ2S_GRID.as_ptr() as *const u8,
            IQ2S_GRID.len() * 8,
        )
    }
}

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(QK_K));
    let mut out = Vec::with_capacity(src.len() / QK_K * BLOCK_BYTES);
    let flat = grid_flat();

    for blk in src.chunks_exact(QK_K) {
        let mut block_max_abs = 0.0f32;
        for &v in blk {
            let a = v.abs();
            if a > block_max_abs { block_max_abs = a; }
        }

        let d = if block_max_abs == 0.0 {
            0.0
        } else {
            block_max_abs / ((0.5 + 15.0) * 0.25 * MAX_GRID_VAL)
        };

        let mut qs = [0u8; 64];
        let mut qh = [0u8; 8];
        let mut scales = [0u8; 8];

        for ib32 in 0..8 {
            let group_start = ib32 * 32;

            let group_max_abs = blk[group_start..group_start + 32]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);

            let max_scale = if d == 0.0 || group_max_abs == 0.0 {
                0u8
            } else {
                let raw = (group_max_abs / (d.abs() * MAX_GRID_VAL) - 0.5) / 0.25;
                (raw as u8).min(15)
            };

            let scale0 = max_scale.min(15);
            let scale1 = max_scale.min(15);
            scales[ib32] = scale0 | (scale1 << 4);

            let db0 = d * (0.5 + scale0 as f32) * 0.25;
            let db1 = d * (0.5 + scale1 as f32) * 0.25;

            for l in 0..4 {
                let dl = if l < 2 { db0 } else { db1 };
                let sub = &blk[group_start + l * 8..group_start + (l + 1) * 8];

                let mut best_idx = 0u16;
                let mut best_signs = 0u8;
                let mut best_err = f32::MAX;

                if d == 0.0 {
                    qs[ib32 * 4 + l] = 0;
                    qs[32 + ib32 * 4 + l] = 0;
                    continue;
                }

                for idx in 0..1024u16 {
                    let off = idx as usize * 8;
                    let g0 = flat[off] as f32;
                    let g1 = flat[off + 1] as f32;
                    let g2 = flat[off + 2] as f32;
                    let g3 = flat[off + 3] as f32;
                    let g4 = flat[off + 4] as f32;
                    let g5 = flat[off + 5] as f32;
                    let g6 = flat[off + 6] as f32;
                    let g7 = flat[off + 7] as f32;
                    let gvals = [g0, g1, g2, g3, g4, g5, g6, g7];

                    for sign_bits in 0..=255u16 {
                        let sbits = sign_bits as u8;
                        let mut err = 0.0f32;
                        for j in 0..8 {
                            let sgn = if sbits & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let recon = dl * gvals[j] * sgn;
                            let diff = sub[j] - recon;
                            err += diff * diff;
                        }

                        if err < best_err {
                            best_err = err;
                            best_idx = idx;
                            best_signs = sbits;
                        }
                    }
                }

                qs[ib32 * 4 + l] = best_idx as u8;
                qh[ib32] |= ((best_idx >> 8) as u8 & 3) << (2 * l);
                qs[32 + ib32 * 4 + l] = best_signs;
            }
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&qs[..64]);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&scales);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Iq2S, bytes, None).unwrap()
}

#[cfg(test)]
#[path = "../../tests/unit/quantize/iq2_s.rs"]
mod tests;
