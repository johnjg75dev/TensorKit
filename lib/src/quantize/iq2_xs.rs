//! IQ2_XS quantizer: 256 elements per super-block, 8 groups of 32,
//! each group has 4 sub-groups of 8. 2-bit non-linear codebook.
//!
//! On-disk: 2 d (f16), 8 scales, 64 qs = 74 bytes / 256 elements.
//!
//! Group g scales:
//!   db[0] = d * (0.5 + (scales[g] & 0x0F) * 0.25)
//!   db[1] = d * (0.5 + (scales[g] >> 4) * 0.25)
//!
//! Sub-group l in group g:
//!   dl = db[l / 2]
//!   single byte: grid_off (8-bit index into IQ2XS_GRID, signs inferred from d sign)

use crate::formats::gguf::dequant;
use crate::formats::gguf::dequant::lookup::IQ2XS_GRID;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const BLOCK_LEN: usize = 256;
const GROUP_LEN: usize = 32;
const N_GROUPS: usize = BLOCK_LEN / GROUP_LEN;
const SUB_LEN: usize = 8;
const N_SUB: usize = GROUP_LEN / SUB_LEN;
const BLOCK_BYTES: usize = 74;

const MAX_GRID_VAL: f32 = 43.0;

fn grid_flat() -> &'static [u8] {
    unsafe {
        std::slice::from_raw_parts(
            IQ2XS_GRID.as_ptr() as *const u8,
            IQ2XS_GRID.len() * 8,
        )
    }
}

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(BLOCK_LEN));
    let mut out = Vec::with_capacity(src.len() / BLOCK_LEN * BLOCK_BYTES);

    let flat = grid_flat();

    for blk in src.chunks_exact(BLOCK_LEN) {
        let mut group_max_abs = [0.0f32; N_GROUPS];
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
            group_max_abs[g] = max_abs;
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

        let mut scales = [0u8; N_GROUPS];
        let mut qs = [0u8; 64];

        for g in 0..N_GROUPS {
            let group = &blk[g * GROUP_LEN..(g + 1) * GROUP_LEN];

            let mut sub_max_half = [0.0f32; 2];
            for half in 0..2 {
                let mut hmax = f32::NEG_INFINITY;
                for l in half * 2..(half + 1) * 2 {
                    let sub = &group[l * SUB_LEN..(l + 1) * SUB_LEN];
                    let max_abs = sub.iter().map(|&v| v.abs()).fold(f32::NEG_INFINITY, f32::max);
                    if max_abs > hmax { hmax = max_abs; }
                }
                sub_max_half[half] = hmax;
            }

            for half in 0..2 {
                let nibble = if d == 0.0 || sub_max_half[half] == 0.0 {
                    0u32
                } else {
                    let s = ((sub_max_half[half] / (d.abs() * MAX_GRID_VAL) - 0.5) / 0.25).ceil();
                    (s as u32).min(15)
                };
                scales[g] |= (nibble as u8) << (if half == 0 { 0 } else { 4 });
            }

            let db = [
                d * (0.5 + (scales[g] & 0x0F) as f32 * 0.25),
                d * (0.5 + (scales[g] >> 4) as f32 * 0.25),
            ];

            for l in 0..N_SUB {
                let dl = db[l / 2];
                let sub = &group[l * SUB_LEN..(l + 1) * SUB_LEN];

                let mut best_off = 0u8;
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

                    let mut err = 0.0f32;
                    for j in 0..8 {
                        let recon = dl * gvals[j];
                        let diff = sub[j] - recon;
                        err += diff * diff;
                    }
                    if err < best_err {
                        best_err = err;
                        best_off = off as u8;
                    }
                }

                qs[g * 4 + l] = best_off;
            }
        }

        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&scales);
        out.extend_from_slice(&qs);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Iq2Xs, bytes, None).unwrap()
}

#[cfg(test)]
#[path = "../../tests/unit/quantize/iq2_xs.rs"]
mod tests;
