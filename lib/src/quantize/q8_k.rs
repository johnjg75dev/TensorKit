//! Q8_K quantizer: 256 elements per super-block, symmetric, 8-bit.
//!
//! On-disk: 4 bytes f32 d, 256 bytes i8 qs. Total 292 B / 256 el.
//! Dequant: x = d * q.

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 292;

pub fn quantize(src: &[f32]) -> Vec<u8> {
    debug_assert!(src.len().is_multiple_of(QK_K));
    let mut out = Vec::with_capacity(src.len() / QK_K * BLOCK_BYTES);
    for blk in src.chunks_exact(QK_K) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

        out.extend_from_slice(&d.to_le_bytes());
        for &v in blk {
            let q = (v * inv_d).round();
            let q = q.clamp(-128.0, 127.0) as i8;
            out.push(q as u8);
        }
    }
    out
}

#[doc(hidden)]
pub fn dequant(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / BLOCK_BYTES * QK_K);
    for blk in bytes.chunks_exact(BLOCK_BYTES) {
        let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        for j in 0..QK_K {
            out.push(d * (blk[4 + j] as i8 as f32));
        }
    }
    out
}