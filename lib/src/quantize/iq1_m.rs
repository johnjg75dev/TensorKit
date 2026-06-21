//! IQ1_M quantizer: 256 elements per super-block.
//! On-disk (56 bytes): qs[32] + qh[16] + scales[8].
//! d is packed as a 12-bit FP16 spread across the high bits of 4 u16 in scales.
//! Uses the same IQ1S_GRID as IQ1_S (2048-entry codebook).

use crate::formats::gguf::dequant;
use crate::formats::gguf::dequant::lookup::{IQ1S_GRID, IQ1S_DELTA};
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 56;

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
        let mut qs = [0u8; 32];
        let mut qh = [0u8; 16];
        let mut scales = [0u16; 4];

        for ib in 0..8 {
            let group_start = ib * 32;

            let mut best_scale1 = 0u8;
            let mut best_scale2 = 0u8;
            let mut best_delta = [false; 4];
            let mut best_grid = [0usize; 4];
            let mut best_error = f32::MAX;

            for s1 in 0..8u8 {
                for s2 in 0..8u8 {
                    let sf1 = 2.0 * s1 as f32 + 1.0;
                    let sf2 = 2.0 * s2 as f32 + 1.0;

                    for &d0 in &[false, true] {
                        let delta0 = if d0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                        for &d1 in &[false, true] {
                            let delta1 = if d1 { -IQ1S_DELTA } else { IQ1S_DELTA };
                            for &d2 in &[false, true] {
                                let delta2 = if d2 { -IQ1S_DELTA } else { IQ1S_DELTA };
                                for &d3 in &[false, true] {
                                    let delta3 = if d3 { -IQ1S_DELTA } else { IQ1S_DELTA };

                                    let deltas = [delta0, delta1, delta2, delta3];
                                    let sfs = [sf1, sf2];
                                    let mut error = 0.0;
                                    let mut grid = [0usize; 4];

                                    if d != 0.0 {
                                        for l in 0..4 {
                                            let sf = sfs[l / 2];
                                            let mut target = [0.0f32; 8];
                                            for j in 0..8 {
                                                let v = blk[group_start + l * 8 + j];
                                                target[j] = (v / d / sf) - deltas[l];
                                            }
                                            let best = find_best_grid(&target, &expanded);
                                            grid[l] = best;
                                            for j in 0..8 {
                                                let recon = d * sf * (expanded[best][j] as f32 + deltas[l]);
                                                let diff = blk[group_start + l * 8 + j] - recon;
                                                error += diff * diff;
                                            }
                                        }
                                    }

                                    if error < best_error {
                                        best_error = error;
                                        best_scale1 = s1;
                                        best_scale2 = s2;
                                        best_delta = [d0, d1, d2, d3];
                                        best_grid = grid;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let scale_bits = (best_scale1 as u16) | ((best_scale2 as u16) << 3);
            scales[ib / 2] |= scale_bits << (6 * (ib % 2));

            for l in 0..4 {
                let g = best_grid[l];
                qs[ib * 4 + l] = (g & 0xff) as u8;
                let qh_shift = if l % 2 == 0 { 0 } else { 4 };
                qh[ib * 2 + l / 2] |= ((g >> 8) as u8 & 7) << qh_shift;
                if best_delta[l] {
                    let bit = if l % 2 == 0 { 0x08 } else { 0x80 };
                    qh[ib * 2 + l / 2] |= bit;
                }
            }
        }

        let d_bits = f32_to_f16_bits(d);
        scales[0] |= (d_bits & 0x000f) << 12;
        scales[1] |= (d_bits & 0x00f0) << 4;
        scales[2] |= (d_bits & 0x0f00) >> 4;
        scales[3] |= d_bits & 0xf000;

        out.extend_from_slice(&qs);
        for &h in &qh {
            out.extend_from_slice(&h.to_le_bytes());
        }
        for &s in &scales {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Iq1M, bytes, None).unwrap()
}

#[cfg(test)]
#[path = "../../tests/unit/quantize/iq1_m.rs"]
mod tests;
