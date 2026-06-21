//! Q3_K quantizer: 256 elements per super-block, 8 sub-blocks of 32, 3-bit.
//!
//! On-disk layout (110 bytes / 256 elements), matching llama.cpp
//! `block_q3_K`:
//!
//!   - 32 bytes: hmask[] — 1-bit high bits (256 bits, packed 8 per byte)
//!   - 64 bytes: qs[] — 2-bit low bits (4× 2-bit per byte)
//!   -  2 bytes: f16 d — super-block scale
//!   - 12 bytes: scales[] — 6-bit sub-block scales in Q5_K-style packing
//!
//! Dequant: `x = d * sc * q` where `q = (ql + h*4) - 16`, ql ∈ [0,3] (2 bits
//! from qs), h ∈ {0,1} (from hmask), giving q ∈ [-16, -9] (8 values = 3 bits).

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::GgmlType;
use crate::quantize::f32_to_f16_bits;

const QK_K: usize = 256;
const SUB: usize = 32;
const N_SUB: usize = QK_K / SUB;
const BLOCK_BYTES: usize = 110;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(QK_K));
    let mut out = Vec::with_capacity(src.len() / QK_K * BLOCK_BYTES);
    for blk in src.chunks_exact(QK_K) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 16.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        let mut sc = [0u8; N_SUB];
        for s in 0..N_SUB {
            let sub = &blk[s * SUB..(s + 1) * SUB];
            let sub_amax = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let sc_f = if d == 0.0 {
                0.0
            } else {
                sub_amax * inv_d / 15.0 * 63.0
            };
            sc[s] = if sub_amax > 0.0 {
                sc_f.round().clamp(1.0, 63.0) as u8
            } else {
                0
            };
        }

        // Pack 12-byte scale table (Q5_K-style: low 6 bits in scales[0..7],
        // high 2 bits packed into scales[8..9], 4 per byte).
        let mut packed = [0u8; 12];
        for s in 0..N_SUB {
            packed[s] = sc[s] & 0x3F;
            packed[8 + (s >> 2)] |= ((sc[s] >> 6) & 3) << ((s & 3) * 2);
        }

        // Quantize each element into 2-bit ql (qs) + 1-bit h (hmask).
        let mut qs = [0u8; 64];
        let mut hmask = [0u8; 32];
        for j in 0..QK_K {
            let s = j / SUB;
            let denom = d * sc[s] as f32;
            let unscaled = if denom == 0.0 {
                0.0
            } else {
                (blk[j] / denom).round()
            };
            let unscaled = unscaled.clamp(-15.999, 0.999);
            let m = ((unscaled + 16.0).round() as i32).clamp(0, 7) as u8;
            let ql = m & 3;
            let h = (m >> 2) & 1;
            qs[j / 4] |= ql << ((j % 4) * 2);
            hmask[j / 8] |= h << (j % 8);
        }

        out.extend_from_slice(&hmask);
        out.extend_from_slice(&qs);
        out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        out.extend_from_slice(&packed);
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    dequant::dequantize(GgmlType::Q3K, bytes, None).unwrap()
}