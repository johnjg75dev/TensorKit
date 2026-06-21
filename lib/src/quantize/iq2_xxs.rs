//! IQ2_XXS quantizer: 256 elements per super-block, 8 groups of 32,
//! each group has 4 sub-groups of 8. 2-bit non-linear codebook.
//!
//! On-disk: 2 d (f16), 64 qs = 66 bytes / 256 elements.
//! qs is 8 groups Ã— 8 bytes = 64 bytes, each group stores 2 Ã— u32:
//!   a0 = 4 Ã— 8-bit grid indices (low)
//!   a1 = 4 Ã— 7-bit sign indices + 4-bit scale (high nibble)
//!
//! Dequant: db = d * (0.5 + scale * 0.25)
//!          out = db * grid_val * sign
//!
//! The IQ2_XXS grid has 256 entries stored as `[u64; 256]` but is accessed
//! as a flat byte array: an 8-bit grid_off selects an 8-byte window at byte
//! offset `grid_off` in the flat array.

use crate::formats::gguf::dequant;
use crate::formats::gguf::dequant::lookup::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const BLOCK_LEN: usize = 256;
const GROUP_LEN: usize = 32;
const N_GROUPS: usize = BLOCK_LEN / GROUP_LEN;
const SUB_LEN: usize = 8;
const N_SUB: usize = GROUP_LEN / SUB_LEN;
const BLOCK_BYTES: usize = 66;

const MAX_GRID_VAL: f32 = 43.0;

fn grid_flat() -> &'static [u8] {
    unsafe {
        std::slice::from_raw_parts(
            IQ2XXS_GRID.as_ptr() as *const u8,
            IQ2XXS_GRID.len() * 8,
        )
    }
}

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK_LEN));
    let mut out = Vec::with_capacity(src.len() / BLOCK_LEN * BLOCK_BYTES);

    let flat = grid_flat();

    for blk in src.chunks_exact(BLOCK_LEN) {
        let mut group_max = [0.0f32; N_GROUPS];
        let mut block_max_abs = f32::NEG_INFINITY;
        let mut block_max_val = f32::NEG_INFINITY;
        let mut block_min_val = f32::INFINITY;
        for g in 0..N_GROUPS {
            let (mx, mn) = blk[g * GROUP_LEN..(g + 1) * GROUP_LEN]
                .iter()
                .fold((f32::NEG_INFINITY, f32::INFINITY), |(mx, mn), &v| {
                    (mx.max(v), mn.min(v))
                });
            let max_abs = mx.abs().max(mn.abs());
            group_max[g] = max_abs;
            if max_abs > block_max_abs { block_max_abs = max_abs; }
            if mx > block_max_val { block_max_val = mx; }
            if mn < block_min_val { block_min_val = mn; }
        }

        let d_sign = if block_max_val.abs() >= block_min_val.abs() { 1.0 } else { -1.0 };
        let d = if block_max_abs == 0.0 {
            0.0
        } else {
            d_sign * block_max_abs / ((0.5 + 15.0 * 0.25) * MAX_GRID_VAL)
        };

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());

        for g in 0..N_GROUPS {
            let group = &blk[g * GROUP_LEN..(g + 1) * GROUP_LEN];

            let scale = if d == 0.0 || group_max[g] == 0.0 {
                0u32
            } else {
                let s = ((group_max[g] / (d.abs() * MAX_GRID_VAL) - 0.5) / 0.25).ceil();
                (s as u32).min(15)
            };
            let db = d * (0.5 + scale as f32 * 0.25);

            let mut a0: u32 = 0;
            let mut a1: u32 = scale << 28;

            for l in 0..N_SUB {
                let sub = &group[l * SUB_LEN..(l + 1) * SUB_LEN];
                let mut best_off = 0u8;
                let mut best_sign = 0u8;
                let mut best_err = f32::MAX;

                for off in 0..=255u16 {
                    let off = off as usize;
                    let g0 = flat[off] as f32;
                    let g1 = flat[off + 1] as f32;
                    let g2 = flat[off + 2] as f32;
                    let g3 = flat[off + 3] as f32;
                    let g4 = flat[off + 4] as f32;
                    let g5 = flat[off + 5] as f32;
                    let g6 = flat[off + 6] as f32;
                    let g7 = flat[off + 7] as f32;
                    let gvals = [g0, g1, g2, g3, g4, g5, g6, g7];

                    for si in 0..128u8 {
                        let sign_mask = KSIGNS_IQ2XS[si as usize];
                        let mut err = 0.0f32;
                        for j in 0..8 {
                            let sgn = if sign_mask & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 };
                            let recon = db * gvals[j] * sgn;
                            let diff = sub[j] - recon;
                            err += diff * diff;
                        }
                        if err < best_err {
                            best_err = err;
                            best_off = off as u8;
                            best_sign = si;
                        }
                    }
                }

                a0 |= (best_off as u32) << (8 * l);
                a1 |= (best_sign as u32) << (7 * l);
            }

            out.extend_from_slice(&a0.to_le_bytes());
            out.extend_from_slice(&a1.to_le_bytes());
        }
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Iq2Xxs, bytes, None).unwrap()
}

#[cfg(test)]
#[path = "../../tests/unit/quantize/iq2_xxs.rs"]
mod tests;
