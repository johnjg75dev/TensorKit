//! IQ3_S quantizer: 256 elements per super-block, 3-bit non-linear codebook.
//!
//! On-disk: 2 d (f16), 64 qs, 8 qh, 32 signs, 4 scales = 110 bytes / 256 elements.
//! 8 groups Ã— 32 elements, processes groups in pairs (ib32 step 2).
//! Per pair: scales[ib32/2] â†’ db1 = d*(1+2*(lo&0xF)), db2 = d*(1+2*(hi>>4))
//! Each sub-group: 2 Ã— 9-bit grid indices from qs (low 8) + qh (high bit).
//!   g1: qs[g*8+2*l] | ((qh[g] << (8-2*l)) & 0x100)
//!   g2: qs[g*8+2*l+1] | ((qh[g] << (7-2*l)) & 0x100)
//! Signs: signs[g*4+l] as 8-bit mask (bits 0-3 for g1, bits 4-7 for g2)

use crate::formats::gguf::dequant;
use crate::formats::gguf::dequant::lookup::{IQ3S_GRID, KMASK_IQ2XS};
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const BLOCK_LEN: usize = 256;
const GROUP_LEN: usize = 32;
const N_GROUPS: usize = BLOCK_LEN / GROUP_LEN;
const SUB_LEN: usize = 8;
const N_SUB: usize = GROUP_LEN / SUB_LEN;
const BLOCK_BYTES: usize = 110;

const MAX_GRID_VAL: f32 = 15.0;

fn grid_entries() -> Vec<[i8; 4]> {
    IQ3S_GRID.iter().map(|&g| {
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
        let mut block_sum = 0.0f32;
        for g in 0..N_GROUPS {
            let (mx, mn, sum) = blk[g * GROUP_LEN..(g + 1) * GROUP_LEN]
                .iter()
                .fold((f32::NEG_INFINITY, f32::INFINITY, 0.0f32), |(mx, mn, s), &v| {
                    (mx.max(v), mn.min(v), s + v)
                });
            let max_abs = mx.abs().max(mn.abs());
            group_max[g] = max_abs;
            if max_abs > block_max_abs { block_max_abs = max_abs; }
            block_sum += sum;
        }

        let d_sign = if block_sum >= 0.0 { 1.0 } else { -1.0 };
        let d = if block_max_abs == 0.0 { 0.0 } else { d_sign * block_max_abs / 372.0 };

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());

        let mut qs = [0u8; 64];
        let mut qh = [0u8; 8];
        let mut signs = [0u8; 32];
        let mut scales = [0u8; 4];

        for ib32 in (0..8).step_by(2) {
            let g_lo = ib32;
            let g_hi = ib32 + 1;

            let group_lo = &blk[g_lo * GROUP_LEN..(g_lo + 1) * GROUP_LEN];
            let group_hi = &blk[g_hi * GROUP_LEN..(g_hi + 1) * GROUP_LEN];

            let scale_lo = if d == 0.0 || group_max[g_lo] == 0.0 {
                0u8
            } else {
                let s = ((group_max[g_lo] / (d.abs() * MAX_GRID_VAL) - 1.0) / 2.0).ceil() as i32;
                s.clamp(0, 15) as u8
            };
            let scale_hi = if d == 0.0 || group_max[g_hi] == 0.0 {
                0u8
            } else {
                let s = ((group_max[g_hi] / (d.abs() * MAX_GRID_VAL) - 1.0) / 2.0).ceil() as i32;
                s.clamp(0, 15) as u8
            };

            scales[ib32 / 2] = scale_lo | (scale_hi << 4);

            let db_lo = d * (1.0 + 2.0 * scale_lo as f32);
            let db_hi = d * (1.0 + 2.0 * scale_hi as f32);

            for group_idx in 0..2 {
                let group = if group_idx == 0 { group_lo } else { group_hi };
                let db = if group_idx == 0 { db_lo } else { db_hi };
                let g_idx = ib32 + group_idx;

                for l in 0..N_SUB {
                    let sub = &group[l * SUB_LEN..(l + 1) * SUB_LEN];
                    let mut best_g1 = 0u16;
                    let mut best_g2 = 0u16;
                    let mut best_err = f32::MAX;
                    let mut best_s_byte = 0u8;

                    for s_byte in 0..=255u16 {
                        let s_byte = s_byte as u8;

                        let mut best_g1_err = f32::MAX;
                        let mut best_g1_idx = 0u16;
                        for gi1 in 0..512u16 {
                            let gv = grid[gi1 as usize];
                            let mut err = 0.0f32;
                            for j in 0..4 {
                                let s = if s_byte & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                                let diff = sub[2 * j] - db * gv[j] as f32 * s;
                                err += diff * diff;
                            }
                            if err < best_g1_err {
                                best_g1_err = err;
                                best_g1_idx = gi1;
                            }
                        }

                        let mut best_g2_err = f32::MAX;
                        let mut best_g2_idx = 0u16;
                        for gi2 in 0..512u16 {
                            let gv = grid[gi2 as usize];
                            let mut err = 0.0f32;
                            for j in 0..4 {
                                let s = if s_byte & KMASK_IQ2XS[j + 4] != 0 { -1.0 } else { 1.0 };
                                let diff = sub[2 * j + 1] - db * gv[j] as f32 * s;
                                err += diff * diff;
                            }
                            if err < best_g2_err {
                                best_g2_err = err;
                                best_g2_idx = gi2;
                            }
                        }

                        let combined = best_g1_err + best_g2_err;
                        if combined < best_err {
                            best_err = combined;
                            best_g1 = best_g1_idx;
                            best_g2 = best_g2_idx;
                            best_s_byte = s_byte;
                        }
                    }

                    let qs_base = g_idx * 8;
                    qs[qs_base + 2 * l] = (best_g1 & 0xFF) as u8;
                    qs[qs_base + 2 * l + 1] = (best_g2 & 0xFF) as u8;

                    if best_g1 & 0x100 != 0 { qh[g_idx] |= 1 << (2 * l); }
                    if best_g2 & 0x100 != 0 { qh[g_idx] |= 1 << (2 * l + 1); }

                    signs[g_idx * 4 + l] = best_s_byte;
                }
            }
        }

        out.extend_from_slice(&qs);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&signs);
        out.extend_from_slice(&scales);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Iq3S, bytes, None).unwrap()
}

#[cfg(test)]
#[path = "../../tests/unit/quantize/iq3_s.rs"]
mod tests;
