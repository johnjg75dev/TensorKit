//! IQ1_S quantizer: 256 elements per super-block, 1-bit.
//!
//! On-disk: 2 d (f16), 16 qh (8 Ã— u16), 32 qs = 50 bytes / 256 elements.
//! Each group of 32 elements uses a 3-bit per-group scale, a delta sign,
//! and 4 sub-groups of 8 elements each referencing a shared codebook
//! (IQ1S_GRID with 2048 entries).

use crate::formats::gguf::dequant;
use crate::formats::gguf::dequant::lookup::{IQ1S_GRID, IQ1S_DELTA};
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 50;

fn expand_iq1s_grid() -> [[i8; 8]; 2048] {
    let mut grid = [[0i8; 8]; 2048];
    for i in 0..2048 {
        grid[i] = IQ1S_GRID[i].to_le_bytes().map(|b| b as i8);
    }
    grid
}

fn find_best_grid(target: &[f32; 8], expanded: &[[i8; 8]; 2048]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for (i, entry) in expanded.iter().enumerate() {
        let mut dist = 0.0;
        for j in 0..8 {
            let diff = target[j] - entry[j] as f32;
            dist += diff * diff;
        }
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx
}

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(QK_K));
    let mut out = Vec::with_capacity(src.len() / QK_K * BLOCK_BYTES);
    let expanded = expand_iq1s_grid();

    for blk in src.chunks_exact(QK_K) {
        let max_abs = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = if max_abs == 0.0 { 0.0 } else { max_abs / 1000.0 };

        let mut qh = [0u16; 8];
        let mut qs = [0u8; 32];

        for ib in 0..8 {
            let group_start = ib * 32;

            let mut best_scale = 0u16;
            let mut best_delta_neg = false;
            let mut best_grid = [0usize; 4];
            let mut best_error = f32::MAX;

            for scale_bits in 0..8u16 {
                let sf = 2.0 * scale_bits as f32 + 1.0;
                for &delta_neg in &[false, true] {
                    let delta = if delta_neg { -IQ1S_DELTA } else { IQ1S_DELTA };
                    let mut error = 0.0;
                    let mut grid = [0usize; 4];

                    if d == 0.0 {
                        error = 0.0;
                    } else {
                        for l in 0..4 {
                            let mut target = [0.0f32; 8];
                            for j in 0..8 {
                                let v = blk[group_start + l * 8 + j];
                                target[j] = (v / d / sf) - delta;
                            }
                            let best = find_best_grid(&target, &expanded);
                            grid[l] = best;
                            for j in 0..8 {
                                let recon = d * sf * (expanded[best][j] as f32 + delta);
                                let diff = blk[group_start + l * 8 + j] - recon;
                                error += diff * diff;
                            }
                        }
                    }

                    if error < best_error {
                        best_error = error;
                        best_scale = scale_bits;
                        best_delta_neg = delta_neg;
                        best_grid = grid;
                    }
                }
            }

            qh[ib] = (best_scale << 12) | ((best_delta_neg as u16) << 15);
            for l in 0..4 {
                let g = best_grid[l];
                qs[ib * 4 + l] = (g & 0xff) as u8;
                qh[ib] |= ((g >> 8) as u16 & 7) << (3 * l);
            }
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for &h in &qh {
            out.extend_from_slice(&h.to_le_bytes());
        }
        out.extend_from_slice(&qs);
    }

    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Iq1S, bytes, None).unwrap()
}

#[cfg(test)]
#[path = "../../tests/unit/quantize/iq1_s.rs"]
mod tests;
