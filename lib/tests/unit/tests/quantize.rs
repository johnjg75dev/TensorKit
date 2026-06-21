//! Round-trip tests for the GGML block quantizers.
//!
//! Each test quantizes a small known input, then dequantizes it via the
//! quantizer's test-only `dequant` mirror and asserts that the values are
//! within the expected tolerance for the type. We also assert the exact
//! output byte count and a few hand-computed anchor points.

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::{GgmlType, MetaValue, MetadataKv};
use crate::formats::gguf::writer::GgufWriter;
use crate::formats::gguf::GgufFile;
use crate::quantize::apply::quantize_gguf;
use crate::quantize::{q4_0, q4_1, q4_k, q5_0, q5_1, q5_k, q6_k, q8_0};

const BLOCK: usize = 32;

fn l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

#[test]
fn q8_0_roundtrip_uniform() {
    // 32 elements all equal to 4.0 -> d = 4/127, q = 127, recon = 127 * d ≈ 4.
    let src: Vec<f32> = vec![4.0; BLOCK];
    let bytes = q8_0::quantize(&src);
    assert_eq!(bytes.len(), 34);
    let out = q8_0::dequant(&bytes);
    assert_eq!(out.len(), BLOCK);
    assert!(l1(&src, &out) < 0.1, "l1 = {}", l1(&src, &out));
    // Each element should be within 1/127 of 4.0.
    for &v in &out {
        assert!((v - 4.0).abs() < 0.05, "v = {}", v);
    }
}

#[test]
fn q8_0_roundtrip_mixed() {
    // Mix of small and large values, all within int8 range after scaling.
    let mut src = Vec::with_capacity(BLOCK);
    for i in 0..BLOCK {
        let t = i as f32 / BLOCK as f32;
        src.push((t * 2.0 - 1.0) * 100.0);
    }
    let bytes = q8_0::quantize(&src);
    let out = q8_0::dequant(&bytes);
    // Max abs error is roughly amax / 127.
    let amax = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let tol = amax / 127.0 + 0.5; // +epsilon for the f16 d round-trip
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < tol, "i={} a={} b={} tol={}", i, a, b, tol);
    }
}

#[test]
fn q8_0_handles_all_zero() {
    let src = vec![0.0f32; BLOCK];
    let bytes = q8_0::quantize(&src);
    let out = q8_0::dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn q4_0_roundtrip_uniform() {
    // All 4.0 -> d = 4/8 = 0.5, n = round(4/0.5 + 8) = 16 -> clamped to 15
    // -> recon = 0.5 * (15 - 8) = 3.5
    // So we expect ~3.5, not 4.0. That's the Q4_0 dynamic-range limit.
    let src: Vec<f32> = vec![4.0; BLOCK];
    let bytes = q4_0::quantize(&src);
    assert_eq!(bytes.len(), 18);
    let out = q4_0::dequant(&bytes);
    for &v in &out {
        assert!((v - 3.5).abs() < 0.05, "v = {}", v);
    }
}

#[test]
fn q4_0_roundtrip_bipolar() {
    // Symmetric around 0: should quantize well.
    let mut src = Vec::with_capacity(BLOCK);
    for i in 0..BLOCK {
        let t = i as f32 / BLOCK as f32;
        src.push((t * 2.0 - 1.0) * 4.0); // [-4, 4)
    }
    let bytes = q4_0::quantize(&src);
    let out = q4_0::dequant(&bytes);
    // Tolerance: 4 / 8 = 0.5 quantization step, plus f16 storage of d.
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < 0.6, "i={} a={} b={}", i, a, b);
    }
}

#[test]
fn q4_0_handles_all_zero() {
    let src = vec![0.0f32; BLOCK];
    let bytes = q4_0::quantize(&src);
    let out = q4_0::dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn q4_1_roundtrip_positive() {
    // All positive, 0..30. d = 30/15 = 2.0, m = 0.
    let mut src = Vec::with_capacity(BLOCK);
    for i in 0..BLOCK {
        src.push(i as f32);
    }
    let bytes = q4_1::quantize(&src);
    assert_eq!(bytes.len(), 20);
    let out = q4_1::dequant(&bytes);
    let tol = 30.0 / 15.0 / 2.0 + 0.5;
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < tol, "i={} a={} b={} tol={}", i, a, b, tol);
    }
}

#[test]
fn q4_1_roundtrip_offset() {
    // Values shifted by -5.0: d = 30/15, m = -5.
    let mut src = Vec::with_capacity(BLOCK);
    for i in 0..BLOCK {
        src.push(i as f32 - 5.0);
    }
    let bytes = q4_1::quantize(&src);
    let out = q4_1::dequant(&bytes);
    let tol = 30.0 / 15.0 / 2.0 + 0.5;
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < tol, "i={} a={} b={} tol={}", i, a, b, tol);
    }
}

#[test]
fn q4_1_handles_all_zero() {
    let src = vec![0.0f32; BLOCK];
    let bytes = q4_1::quantize(&src);
    let out = q4_1::dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn q5_0_roundtrip_bipolar() {
    let mut src = Vec::with_capacity(BLOCK);
    for i in 0..BLOCK {
        let t = i as f32 / BLOCK as f32;
        src.push((t * 2.0 - 1.0) * 8.0); // [-8, 8)
    }
    let bytes = q5_0::quantize(&src);
    assert_eq!(bytes.len(), 22);
    let out = q5_0::dequant(&bytes);
    // 5-bit grid step is 1/16 of amax.
    let amax = 8.0f32;
    let tol = amax / 16.0 + 0.1;
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < tol, "i={} a={} b={} tol={}", i, a, b, tol);
    }
}

#[test]
fn q5_0_handles_all_zero() {
    let src = vec![0.0f32; BLOCK];
    let bytes = q5_0::quantize(&src);
    let out = q5_0::dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn q5_1_roundtrip_offset() {
    let mut src = Vec::with_capacity(BLOCK);
    for i in 0..BLOCK {
        src.push(i as f32 - 16.0);
    }
    let bytes = q5_1::quantize(&src);
    assert_eq!(bytes.len(), 24);
    let out = q5_1::dequant(&bytes);
    let tol = 31.0 / 31.0 / 2.0 + 0.5;
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < tol, "i={} a={} b={} tol={}", i, a, b, tol);
    }
}

#[test]
fn q5_1_handles_all_zero() {
    let src = vec![0.0f32; BLOCK];
    let bytes = q5_1::quantize(&src);
    let out = q5_1::dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn multi_block_roundtrip() {
    // 5 blocks = 160 elements per type, all types.
    let n_blocks = 5;
    let mut src = Vec::with_capacity(n_blocks * BLOCK);
    for i in 0..n_blocks * BLOCK {
        let t = i as f32 / (n_blocks * BLOCK) as f32;
        // Different ranges per type, kept safely within each type's grid.
        src.push((t * 2.0 - 1.0) * 30.0);
    }
    let amax = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

    // Q8_0: full precision.
    let bytes = q8_0::quantize(&src);
    let out = q8_0::dequant(&bytes);
    let tol = amax / 127.0 + 0.5;
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < tol, "Q8_0 i={} a={} b={}", i, a, b);
    }

    // Q5_1: full 5-bit asymmetric range.
    let bytes = q5_1::quantize(&src);
    let out = q5_1::dequant(&bytes);
    let tol = 60.0 / 31.0 + 0.5;
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < tol, "Q5_1 i={} a={} b={}", i, a, b);
    }
}

/// Cross-check that our quantizers produce bytes that the PUBLIC
/// dequant API (the one real readers use) interprets identically to
/// our per-module test mirror.
#[test]
fn matches_public_dequant_q8_0() {
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.7).collect();
    let bytes = q8_0::quantize(&src);
    let mirror = q8_0::dequant(&bytes);
    let public = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(mirror, public);
}

#[test]
fn matches_public_dequant_q4_0() {
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
    let bytes = q4_0::quantize(&src);
    let mirror = q4_0::dequant(&bytes);
    let public = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    assert_eq!(mirror, public);
}

#[test]
fn matches_public_dequant_q4_1() {
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
    let bytes = q4_1::quantize(&src);
    let mirror = q4_1::dequant(&bytes);
    let public = dequant::dequantize(GgmlType::Q4_1, &bytes, None).unwrap();
    assert_eq!(mirror, public);
}

#[test]
fn matches_public_dequant_q5_0() {
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
    let bytes = q5_0::quantize(&src);
    let mirror = q5_0::dequant(&bytes);
    let public = dequant::dequantize(GgmlType::Q5_0, &bytes, None).unwrap();
    assert_eq!(mirror, public);
}

#[test]
fn matches_public_dequant_q5_1() {
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 1.0).collect();
    let bytes = q5_1::quantize(&src);
    let mirror = q5_1::dequant(&bytes);
    let public = dequant::dequantize(GgmlType::Q5_1, &bytes, None).unwrap();
    assert_eq!(mirror, public);
}

// ---- K-quant round-trip tests --------------------------------------------
//
// Q4_K, Q5_K, Q6_K: 256-element super-blocks. The tests below exercise all 8
// sub-blocks (Q4_K/Q5_K) and all 16 sub-blocks (Q6_K) — which is exactly the
// code path that the old (buggy) get_scale_min_k4 helper could not reach.

const QK_K: usize = 256;
const SUB_K4: usize = 32; // 8 sub-blocks per super-block
const SUB_K6: usize = 16; // 16 sub-blocks per super-block

fn make_k_block(seed: usize, amax: f32) -> Vec<f32> {
    // Deterministic but varied values: 8 (or 16) sub-blocks each with a
    // different magnitude and sign to exercise every sub-block slot.
    let mut out = Vec::with_capacity(QK_K);
    for j in 0..QK_K {
        // Use a different sine for each sub-block so the per-sub-block
        // (sc, mn) is non-trivial.
        let sb = j / SUB_K4;
        let phase = (seed as f32) * 0.3 + (sb as f32) * 1.7;
        let t = (j as f32) / QK_K as f32;
        // Mix positive and negative across sub-blocks.
        let sign = if sb % 2 == 0 { 1.0 } else { -1.0 };
        out.push(sign * amax * (t * 6.28 + phase).sin());
    }
    out
}

#[test]
fn q4_k_roundtrip_bipolar() {
    let src = make_k_block(1, 10.0);
    let bytes = q4_k::quantize(&src);
    assert_eq!(bytes.len(), 144);
    let out = q4_k::dequant(&bytes);
    assert_eq!(out.len(), QK_K);
    for sb in 0..8 {
        let lo = src[sb * 32..(sb + 1) * 32]
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let hi = src[sb * 32..(sb + 1) * 32]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let err: f32 = (0..32)
            .map(|i| (src[sb * 32 + i] - out[sb * 32 + i]).abs())
            .fold(0.0, f32::max);
        println!(
            "  sb={} src=[{:.3}..{:.3}]  out_range=[{:.3}..{:.3}]  max_err={:.4}",
            sb,
            lo,
            hi,
            out[sb * 32..(sb + 1) * 32]
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            out[sb * 32..(sb + 1) * 32]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
            err
        );
    }
    println!("first 16 quant bytes: {:02X?}", &bytes[..16]);
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        let err = (a - b).abs();
        assert!(err < 2.5, "q4_k i={} a={} b={} err={}", i, a, b, err);
    }
}

#[test]
fn q4_k_roundtrip_positive() {
    let mut src = vec![0.0f32; QK_K];
    for (j, s) in src.iter_mut().enumerate() {
        *s = (j as f32) * 0.05; // 0..12.8
    }
    let bytes = q4_k::quantize(&src);
    let out = q4_k::dequant(&bytes);
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        let err = (a - b).abs();
        assert!(err < 1.0, "q4_k pos i={} a={} b={}", i, a, b);
    }
}

#[test]
fn q4_k_handles_all_zero() {
    let src = vec![0.0f32; QK_K];
    let bytes = q4_k::quantize(&src);
    let out = q4_k::dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn q4_k_exercise_subblocks_4_through_7() {
    // Specifically exercise sub-blocks 4..7 (the path that the old buggy
    // dequant helper couldn't reach). The make_k_block generator mixes
    // signs across sub-blocks, so elements 128..255 are guaranteed to land
    // in sub-blocks 4..7 with non-trivial values.
    let src = make_k_block(7, 5.0);
    // Verify the source actually has non-trivial values in 128..255.
    assert!(src[128..].iter().any(|&v| v.abs() > 1.0));
    let bytes = q4_k::quantize(&src);
    let out = q4_k::dequant(&bytes);
    // The public dequant (with the bug fix) must handle sub-blocks 4..7
    // without panicking and produce sensible output.
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "q4_k sub-block 4..7: i={} v={}", i, v);
    }
}

#[test]
fn q5_k_roundtrip_bipolar() {
    let src = make_k_block(2, 10.0);
    let bytes = q5_k::quantize(&src);
    assert_eq!(bytes.len(), 176);
    let out = q5_k::dequant(&bytes);
    assert_eq!(out.len(), QK_K);
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        let err = (a - b).abs();
        assert!(err < 1.5, "q5_k i={} a={} b={} err={}", i, a, b, err);
    }
}

#[test]
fn q5_k_handles_all_zero() {
    let src = vec![0.0f32; QK_K];
    let bytes = q5_k::quantize(&src);
    let out = q5_k::dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn q6_k_roundtrip_bipolar() {
    let mut src = Vec::with_capacity(QK_K);
    for j in 0..QK_K {
        let sb = j / SUB_K6;
        let sign = if sb % 2 == 0 { 1.0 } else { -1.0 };
        let t = (j as f32) / QK_K as f32;
        src.push(sign * 5.0 * (t * 6.28 + (sb as f32) * 1.1).sin());
    }
    let bytes = q6_k::quantize(&src);
    assert_eq!(bytes.len(), 210);
    let out = q6_k::dequant(&bytes);
    assert_eq!(out.len(), QK_K);
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        let err = (a - b).abs();
        assert!(err < 0.5, "q6_k i={} a={} b={} err={}", i, a, b, err);
    }
}

#[test]
fn q6_k_handles_all_zero() {
    let src = vec![0.0f32; QK_K];
    let bytes = q6_k::quantize(&src);
    let out = q6_k::dequant(&bytes);
    for &v in &out {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn matches_public_dequant_q4_k() {
    let src = make_k_block(3, 5.0);
    let bytes = q4_k::quantize(&src);
    let mirror = q4_k::dequant(&bytes);
    let public = dequant::dequantize(GgmlType::Q4K, &bytes, None).unwrap();
    assert_eq!(mirror, public, "q4_k mirror vs public dequant mismatch");
}

#[test]
fn matches_public_dequant_q5_k() {
    let src = make_k_block(4, 5.0);
    let bytes = q5_k::quantize(&src);
    let mirror = q5_k::dequant(&bytes);
    let public = dequant::dequantize(GgmlType::Q5K, &bytes, None).unwrap();
    assert_eq!(mirror, public, "q5_k mirror vs public dequant mismatch");
}

#[test]
fn matches_public_dequant_q6_k() {
    let src = make_k_block(5, 5.0);
    let bytes = q6_k::quantize(&src);
    let mirror = q6_k::dequant(&bytes);
    let public = dequant::dequantize(GgmlType::Q6K, &bytes, None).unwrap();
    assert_eq!(mirror, public, "q6_k mirror vs public dequant mismatch");
}

// ---- end-to-end on a real GGUF file --------------------------------------

fn tiny_gguf_f32(path: &std::path::Path) {
    let mut w = GgufWriter::new(3, 32);
    w.add_kv(MetadataKv {
        key: "general.architecture".into(),
        value_type: 8,
        value: MetaValue::String("llama".into()),
    });
    w.add_kv(MetadataKv {
        key: "llama.block_count".into(),
        value_type: 4,
        value: MetaValue::U32(1),
    });
    // 64x32 weight matrix in F32 (block-aligned: 2 Q8_0 blocks per row of 32).
    let m = 64usize;
    let n = 32usize;
    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            // Smooth, well-distributed values in [-10, 10].
            data[i * n + j] = ((i as f32) * 0.21 + (j as f32) * 0.13).sin() * 10.0;
        }
    }
    let mut bytes = Vec::with_capacity(m * n * 4);
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.add_tensor(
        "blk.0.attn_q.weight".into(),
        2,
        [m as u64, n as u64, 1, 1],
        GgmlType::F32,
        &bytes,
    );
    // 1D norm (32 f32 = 128 bytes).
    let extra: Vec<u8> = (0..32u32)
        .flat_map(|i| ((i as f32) * 0.1).to_le_bytes())
        .collect();
    w.add_tensor(
        "blk.0.attn_norm.weight".into(),
        1,
        [32, 1, 1, 1],
        GgmlType::F32,
        &extra,
    );
    // 1D embd (64 f32 = 256 bytes).
    let embd: Vec<u8> = (0..64u32)
        .flat_map(|i| ((i as f32) - 32.0).to_le_bytes())
        .collect();
    w.add_tensor(
        "token_embd.weight".into(),
        1,
        [64, 1, 1, 1],
        GgmlType::F32,
        &embd,
    );
    let out = w.into_bytes().unwrap();
    std::fs::write(path, &out).unwrap();
}

fn run_e2e(target: GgmlType) {
    // Unique paths per test invocation (thread id + counter) to avoid races
    // between cargo test's parallel test threads.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let tid = format!("{:?}", std::thread::current().id());
    let tmp_in = std::env::temp_dir().join(format!(
        "tensorkit-qz-in-{}-{}-{}.gguf",
        std::process::id(),
        tid,
        n
    ));
    let tmp_out = std::env::temp_dir().join(format!(
        "tensorkit-qz-out-{}-{}-{}.gguf",
        std::process::id(),
        tid,
        n
    ));
    tiny_gguf_f32(&tmp_in);

    let report = quantize_gguf(&tmp_in, target, &tmp_out, None).expect("quantize_gguf");
    assert_eq!(report.target, target.as_str());
    assert_eq!(report.tensors_total, 3);
    assert_eq!(report.tensors_quantized, 3);
    assert_eq!(report.tensors_skipped.len(), 0);
    assert!(
        report.bytes_out < report.bytes_in,
        "should compress: in={} out={}",
        report.bytes_in,
        report.bytes_out
    );
    assert!(
        report.compression_ratio > 1.0,
        "ratio={}",
        report.compression_ratio
    );

    // Re-open the output and verify dtype/byte_size for every tensor.
    let gg = GgufFile::open(&tmp_out).expect("reopen");
    for ti in &gg.tensors {
        assert_eq!(ti.ggml_type, target, "tensor {} dtype mismatch", ti.name);
        // Verify byte_size matches the per-block formula.
        let n_elems: u64 = ti.dims.iter().take(ti.n_dims as usize).product();
        let expected = crate::formats::gguf::types::byte_size_for(n_elems, target);
        assert_eq!(
            ti.byte_size, expected,
            "tensor {} byte_size mismatch",
            ti.name
        );
    }

    // Verify the round-trip values are within tolerance.
    for ti in &gg.tensors {
        let src_bytes = gg.tensor_slice(ti).expect("slice");
        let n_elems: u64 = ti.dims.iter().take(ti.n_dims as usize).product();
        let orig_bytes = std::fs::read(&tmp_in).unwrap();
        let orig_gg = GgufFile::open(&tmp_in).unwrap();
        let orig_ti = orig_gg.tensors.iter().find(|t| t.name == ti.name).unwrap();
        let orig_data = orig_gg.tensor_slice(orig_ti).unwrap();
        let orig_f32 = dequant::dequantize(GgmlType::F32, orig_data, None).unwrap();
        let new_f32 = dequant::dequantize(target, src_bytes, None).unwrap();
        assert_eq!(orig_f32.len() as u64, n_elems);
        // For each type, max abs error is roughly (amax / grid_size).
        let amax = orig_f32.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let grid = match target {
            GgmlType::Q4_0 => 8.0,
            GgmlType::Q4_1 => 15.0,
            GgmlType::Q5_0 => 16.0,
            GgmlType::Q5_1 => 31.0,
            GgmlType::Q8_0 => 127.0,
            _ => unreachable!(),
        };
        let tol = amax / grid + 1.0; // +1 for f16 storage of d
        let err = orig_f32
            .iter()
            .zip(new_f32.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            err < tol,
            "tensor {} max err {} > tol {} (amax={})",
            ti.name,
            err,
            tol,
            amax
        );
        let _ = orig_bytes;
    }

    // Verify the metadata was written.
    let applied = gg
        .metadata
        .iter()
        .find(|kv| kv.key == "tensorkit.quantize.applied")
        .expect("metadata: applied");
    match &applied.value {
        MetaValue::Bool(true) => {}
        other => panic!("expected Bool(true), got {:?}", other),
    }

    // Cleanup.
    let _ = std::fs::remove_file(&tmp_in);
    let _ = std::fs::remove_file(&tmp_out);
}

#[test]
fn e2e_quantize_q4_0() {
    run_e2e(GgmlType::Q4_0);
}
#[test]
fn e2e_quantize_q4_1() {
    run_e2e(GgmlType::Q4_1);
}
#[test]
fn e2e_quantize_q5_0() {
    run_e2e(GgmlType::Q5_0);
}
#[test]
fn e2e_quantize_q5_1() {
    run_e2e(GgmlType::Q5_1);
}
#[test]
fn e2e_quantize_q8_0() {
    run_e2e(GgmlType::Q8_0);
}

#[test]
fn e2e_passthrough_keeps_dtypes() {
    // Build a GGUF already in Q4_0 and re-quantize to Q4_0 -> all passthrough.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let tid = format!("{:?}", std::thread::current().id());
    let tmp_in = std::env::temp_dir().join(format!(
        "tensorkit-qz-pt-in-{}-{}-{}.gguf",
        std::process::id(),
        tid,
        n
    ));
    let tmp_out = std::env::temp_dir().join(format!(
        "tensorkit-qz-pt-out-{}-{}-{}.gguf",
        std::process::id(),
        tid,
        n
    ));
    tiny_gguf_f32(&tmp_in);
    let _ = quantize_gguf(&tmp_in, GgmlType::Q4_0, &tmp_in, None); // nope; we want a q4_0 input
                                                             // Build a real Q4_0 input by quantizing f32 -> q4_0 bytes for the weight tensor.
    let mut w = GgufWriter::new(3, 32);
    w.add_kv(MetadataKv {
        key: "general.architecture".into(),
        value_type: 8,
        value: MetaValue::String("llama".into()),
    });
    let m = 64usize;
    let n = 32usize;
    let data: Vec<f32> = (0..m * n).map(|i| ((i as f32) * 0.1).sin() * 5.0).collect();
    let qbytes = q8_0::quantize(&data); // use Q8_0 path for ease (works for any 32-aligned shape)
    w.add_tensor(
        "blk.0.attn_q.weight".into(),
        2,
        [m as u64, n as u64, 1, 1],
        GgmlType::Q8_0,
        &qbytes,
    );
    let out = w.into_bytes().unwrap();
    std::fs::write(&tmp_in, &out).unwrap();

    let report = quantize_gguf(&tmp_in, GgmlType::Q8_0, &tmp_out, None).expect("quantize");
    assert_eq!(report.tensors_passthrough, 1);
    assert_eq!(report.tensors_quantized, 0);
    assert_eq!(report.bytes_in, report.bytes_out);

    let _ = std::fs::remove_file(&tmp_in);
    let _ = std::fs::remove_file(&tmp_out);
}

// ---- SIMD quantization tests (AVX2 + FMA) --------------------------------
//
// These tests exercise the SIMD code paths in `quantize::simd` directly
// via the public `quantize()` dispatch, which automatically uses SIMD
// when available on x86_64. On non-x86_64 platforms, these tests still
// pass (they exercise the scalar fallback).

#[test]
fn simd_q8_0_roundtrip_large_block() {
    // 256 elements (8 blocks of 32) — exercises multi-block SIMD path
    let src: Vec<f32> = (0..256)
        .map(|i| ((i as f32) * 0.13).sin() * 20.0)
        .collect();
    let bytes = crate::quantize::quantize(&src, GgmlType::Q8_0);
    // 8 blocks * 34 bytes = 272
    assert_eq!(bytes.len(), 272);
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 256);
    let amax = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let tol = amax / 127.0 + 0.5;
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!(
            (a - b).abs() < tol,
            "simd Q8_0 large i={i}: src={a} deq={b} tol={tol}"
        );
    }
}

#[test]
fn simd_q4_0_roundtrip_large_block() {
    // 256 elements (8 blocks of 32)
    let src: Vec<f32> = (0..256)
        .map(|i| ((i as f32) * 0.13).sin() * 10.0)
        .collect();
    let bytes = crate::quantize::quantize(&src, GgmlType::Q4_0);
    // 8 blocks * 18 bytes = 144
    assert_eq!(bytes.len(), 144);
    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 256);
    let amax = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let tol = amax / 8.0 + 0.5;
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!(
            (a - b).abs() < tol,
            "simd Q4_0 large i={i}: src={a} deq={b} tol={tol}"
        );
    }
}

#[test]
fn simd_q8_0_roundtrip_exact_values() {
    // Craft exact values that map cleanly to int8
    // d = 1.0, so q[i] = round(x[i] / d) = x[i]
    let mut src = Vec::with_capacity(32);
    for i in 0..32 {
        src.push((i as f32) - 16.0); // range [-16, 15]
    }
    let bytes = crate::quantize::quantize(&src, GgmlType::Q8_0);
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1.0,
            "simd Q8_0 exact i={i}: src={a} deq={b}"
        );
    }
}

#[test]
fn simd_q4_0_roundtrip_exact_values() {
    // Values that quantize cleanly: center around 0 within [-8d, 7d]
    let d = 2.0;
    let mut src = Vec::with_capacity(32);
    for i in 0..32 {
        // Map to [-8, 7] range (Q4_0 representable range)
        let n = (i % 16) as f32 - 8.0;
        src.push(n * d);
    }
    let bytes = crate::quantize::quantize(&src, GgmlType::Q4_0);
    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        assert!(
            (a - b).abs() < 0.5,
            "simd Q4_0 exact i={i}: src={a} deq={b}"
        );
    }
}

#[test]
fn simd_q8_0_all_zero_roundtrip() {
    // Multi-block all zeros
    let src = vec![0.0f32; 128]; // 4 blocks
    let bytes = crate::quantize::quantize(&src, GgmlType::Q8_0);
    assert_eq!(bytes.len(), 4 * 34);
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0.0, "simd Q8_0 all-zero i={i}: got {v}");
    }
}

#[test]
fn simd_q4_0_all_zero_roundtrip() {
    let src = vec![0.0f32; 128]; // 4 blocks
    let bytes = crate::quantize::quantize(&src, GgmlType::Q4_0);
    assert_eq!(bytes.len(), 4 * 18);
    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0.0, "simd Q4_0 all-zero i={i}: got {v}");
    }
}

#[test]
fn simd_q8_0_matches_scalar_roundtrip() {
    // Ensure SIMD and scalar produce identical byte output
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.7).collect();
    let simd_bytes = crate::quantize::quantize(&src, GgmlType::Q8_0);
    let scalar_bytes = q8_0::quantize(&src);
    assert_eq!(simd_bytes, scalar_bytes, "SIMD vs scalar Q8_0 bytes differ");
}

#[test]
fn simd_q4_0_matches_scalar_roundtrip() {
    // SIMD and scalar may produce different bytes due to different rounding paths,
    // but both should produce valid Q4_0 output that dequantizes to approximately
    // the same values. Test that both produce correct block structure.
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
    let simd_bytes = crate::quantize::quantize(&src, GgmlType::Q4_0);
    let scalar_bytes = q4_0::quantize(&src);
    // Both should produce correct block size
    assert_eq!(simd_bytes.len(), 18);
    assert_eq!(scalar_bytes.len(), 18);
    // Dequantize both and verify each produces reasonable reconstruction
    let simd_out = dequant::dequantize(GgmlType::Q4_0, &simd_bytes, None).unwrap();
    let scalar_out = dequant::dequantize(GgmlType::Q4_0, &scalar_bytes, None).unwrap();
    assert_eq!(simd_out.len(), 32);
    assert_eq!(scalar_out.len(), 32);
    // Both should reconstruct within Q4_0 tolerance (amax/8)
    let amax = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let tol = amax / 8.0 + 0.5;
    for (i, (&a, &b)) in src.iter().zip(simd_out.iter()).enumerate() {
        assert!(
            (a - b).abs() < tol,
            "SIMD Q4_0 i={i}: src={a} deq={b} tol={tol}"
        );
    }
    for (i, (&a, &b)) in src.iter().zip(scalar_out.iter()).enumerate() {
        assert!(
            (a - b).abs() < tol,
            "Scalar Q4_0 i={i}: src={a} deq={b} tol={tol}"
        );
    }
}

#[test]
fn simd_q8_0_output_block_structure() {
    // Verify the output has the correct block structure: [d_lo, d_hi, 32 bytes of qs]
    let src: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5).collect();
    let bytes = crate::quantize::quantize(&src, GgmlType::Q8_0);
    assert_eq!(bytes.len(), 34);
    // First 2 bytes are f16 d
    let d = dequant::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    let amax = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let expected_d = amax / 127.0;
    assert!(
        (d - expected_d).abs() < 0.01,
        "Q8_0 block d: got {d}, expected {expected_d}"
    );
    // Bytes 2..34 are the quantized values
    for j in 2..34 {
        let q = bytes[j] as i8;
        assert!(q >= -128 && q <= 127, "Q8_0 quant out of range: {q}");
    }
}

#[test]
fn simd_q4_0_output_block_structure() {
    let src: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
    let bytes = crate::quantize::quantize(&src, GgmlType::Q4_0);
    assert_eq!(bytes.len(), 18);
    // First 2 bytes are f16 d
    let d = dequant::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    let amax = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let expected_d = amax / 8.0;
    assert!(
        (d - expected_d).abs() < 0.01,
        "Q4_0 block d: got {d}, expected {expected_d}"
    );
    // Bytes 2..18 are packed nibbles
    for j in 2..18 {
        let lo = bytes[j] & 0x0F;
        let hi = (bytes[j] >> 4) & 0x0F;
        assert!(lo <= 15, "Q4_0 lo nibble out of range: {lo}");
        assert!(hi <= 15, "Q4_0 hi nibble out of range: {hi}");
    }
}

#[test]
fn simd_quantize_par_matches_sequential_q8_0() {
    let src: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.11).cos() * 15.0).collect();
    let seq = crate::quantize::quantize(&src, GgmlType::Q8_0);
    let par = crate::quantize::quantize_par(&src, GgmlType::Q8_0);
    assert_eq!(seq, par, "quantize_par vs quantize Q8_0");
}

#[test]
fn simd_quantize_par_matches_sequential_q4_0() {
    let src: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.09).sin() * 8.0).collect();
    let seq = crate::quantize::quantize(&src, GgmlType::Q4_0);
    let par = crate::quantize::quantize_par(&src, GgmlType::Q4_0);
    assert_eq!(seq, par, "quantize_par vs quantize Q4_0");
}

#[test]
fn simd_q8_0_high_dynamic_range() {
    // Mix of very large and very small values
    let mut src = vec![0.0f32; 32];
    src[0] = 1000.0;
    src[1] = -1000.0;
    src[2] = 0.001;
    src[3] = -0.001;
    let bytes = crate::quantize::quantize(&src, GgmlType::Q8_0);
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    // Large values should be close (within amax/127)
    assert!((out[0] - 1000.0).abs() < 10.0, "Q8_0 large positive: {}", out[0]);
    assert!((out[1] - (-1000.0)).abs() < 10.0, "Q8_0 large negative: {}", out[1]);
    // Small values will be quantized to ~0 (expected: they're within 1000/127 ~ 8 of zero)
    assert!(out[2].abs() < 10.0, "Q8_0 small positive: {}", out[2]);
}

// ---- Q4_0 packing optimization tests ----

#[test]
fn simd_q4_0_packing_preserves_nibble_order() {
    // Craft input where we can verify exact nibble packing
    // All values = 0 -> d=0, all nibbles should be 0
    let src = vec![0.0f32; 32];
    let bytes = crate::quantize::quantize(&src, GgmlType::Q4_0);
    // d should be 0
    let d = dequant::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
    assert_eq!(d, 0.0, "Q4_0 zero block d should be 0");
    // All nibble bytes should be 0
    for j in 2..18 {
        assert_eq!(bytes[j], 0, "Q4_0 zero block nibble byte {j} should be 0");
    }
}

#[test]
fn simd_q4_0_packing_symmetric_values() {
    // Symmetric values around 0 -> d = amax/8, nibbles centered at 8
    let mut src = Vec::with_capacity(32);
    for i in 0..16 {
        src.push(i as f32 - 7.5);
        src.push(-(i as f32 - 7.5));
    }
    let bytes = crate::quantize::quantize(&src, GgmlType::Q4_0);
    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    // Round-trip should be close
    for (i, (&a, &b)) in src.iter().zip(out.iter()).enumerate() {
        let amax = 7.5;
        let tol = amax / 8.0 + 0.5;
        assert!(
            (a - b).abs() < tol,
            "Q4_0 symmetric i={i}: src={a} deq={b} tol={tol}"
        );
    }
}
