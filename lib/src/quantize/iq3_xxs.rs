//! IQ3_XXS quantizer: 256 elements per super-block, 3-bit non-linear codebook.
//!
//! On-disk: 2 d (f16), 64 qs, 32 scs = 98 bytes / 256 elements.
//! 8 groups Ã— 32 elements, each group has 4 sub-groups of 8.
//! Per group: scs[g*4..g*4+4] decoded as u32 aux32.
//!   db = d * (0.5 + (aux32>>28) * 0.5)
//!   Signs from KSIGNS_IQ2XS[(aux32 >> (7*l)) & 0x7F]
//!   Grid indices from qs[ib32*8 + 2*l] (8-bit entry in IQ3XXS_GRID[256])

use crate::formats::gguf::dequant;
use crate::formats::gguf::dequant::lookup::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const BLOCK_LEN: usize = 256;
const GROUP_LEN: usize = 32;
const N_GROUPS: usize = BLOCK_LEN / GROUP_LEN;
const SUB_LEN: usize = 8;
const N_SUB: usize = GROUP_LEN / SUB_LEN;
const BLOCK_BYTES: usize = 98;

const MAX_GRID_VAL: f32 = 10.0;

fn grid_entries() -> Vec<[i8; 4]> {
    IQ3XXS_GRID.iter().map(|&g| {
        let b = g.to_le_bytes();
        [b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8]
    }).collect()
}

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK_LEN));
    let mut out = Vec::with_capacity(src.len() / BLOCK_LEN * BLOCK_BYTES);
    let grid = grid_entries();

    for blk in src.chunks_exact(BLOCK_LEN) {
        let mut group_max = [0.0f32; N_GROUPS];
        let mut block_max_abs = f32::NEG_INFINITY;
        for g in 0..N_GROUPS {
            let (mx, mn) = blk[g * GROUP_LEN..(g + 1) * GROUP_LEN]
                .iter()
                .fold((f32::NEG_INFINITY, f32::INFINITY), |(mx, mn), &v| {
                    (mx.max(v), mn.min(v))
                });
            let max_abs = mx.abs().max(mn.abs());
            group_max[g] = max_abs;
            if max_abs > block_max_abs { block_max_abs = max_abs; }
        }

        let d = if block_max_abs == 0.0 { 0.0 } else { block_max_abs / 64.0 };

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());

        let mut qs = [0u8; 64];
        let mut scs = [0u8; 32];

        for g in 0..N_GROUPS {
            let group = &blk[g * GROUP_LEN..(g + 1) * GROUP_LEN];

            let scale = if d == 0.0 || group_max[g] == 0.0 {
                0u32
            } else {
                let s = ((group_max[g] / (d.abs() * MAX_GRID_VAL) - 0.5) / 0.5).ceil() as i32;
                s.clamp(0, 15) as u32
            };
            let db = d * (0.5 + scale as f32 * 0.5);

            let mut aux32 = scale << 28;

            for l in 0..N_SUB {
                let sub = &group[l * SUB_LEN..(l + 1) * SUB_LEN];
                let mut best_g1 = 0u8;
                let mut best_g2 = 0u8;
                let mut best_si = 0u8;
                let mut best_err = f32::MAX;

                for si in 0..128u8 {
                    let sign_mask = KSIGNS_IQ2XS[si as usize];

                    let mut best_g1_err = f32::MAX;
                    let mut best_g1_idx = 0u8;
                    for g1 in 0..=255u16 {
                        let gv = grid[g1 as usize];
                        let mut err = 0.0f32;
                        for j in 0..4 {
                            let s = if sign_mask & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let diff = sub[2 * j] - db * gv[j] as f32 * s;
                            err += diff * diff;
                        }
                        if err < best_g1_err {
                            best_g1_err = err;
                            best_g1_idx = g1 as u8;
                        }
                    }

                    let mut best_g2_err = f32::MAX;
                    let mut best_g2_idx = 0u8;
                    for g2 in 0..=255u16 {
                        let gv = grid[g2 as usize];
                        let mut err = 0.0f32;
                        for j in 0..4 {
                            let s = if sign_mask & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                            let diff = sub[2 * j + 1] - db * gv[j] as f32 * s;
                            err += diff * diff;
                        }
                        if err < best_g2_err {
                            best_g2_err = err;
                            best_g2_idx = g2 as u8;
                        }
                    }

                    let combined = best_g1_err + best_g2_err;
                    if combined < best_err {
                        best_err = combined;
                        best_g1 = best_g1_idx;
                        best_g2 = best_g2_idx;
                        best_si = si;
                    }
                }

                qs[g * 8 + 2 * l] = best_g1;
                qs[g * 8 + 2 * l + 1] = best_g2;
                aux32 |= (best_si as u32) << (7 * l);
            }

            scs[g * 4..g * 4 + 4].copy_from_slice(&aux32.to_le_bytes());
        }

        out.extend_from_slice(&qs);
        out.extend_from_slice(&scs);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Iq3Xxs, bytes, None).unwrap()
}

#[cfg(test)]
#[path = "../../tests/unit/quantize/iq3_xxs.rs"]
mod tests;
