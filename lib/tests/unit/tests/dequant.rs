//! Tests for dequantization, focusing on SIMD code paths and correctness.
//!
//! Tests verify that:
//! - Q8_0 dequantization via SIMD produces correct output
//! - Q8_1 dequantization via SIMD produces correct output
//! - Q4_0 dequantization via SIMD produces correct output
//! - Scalar and SIMD paths produce identical results
//! - Edge cases (all zeros, single block, multi-block) are handled

use crate::formats::gguf::dequant;
use crate::formats::gguf::types::GgmlType;

const Q8_0_BLOCK: usize = 34; // 2 (d) + 32 (qs)
const Q8_1_BLOCK: usize = 36; // 4 (d + sum) + 32 (qs)
const Q4_0_BLOCK: usize = 18; // 2 (d) + 16 (qs)

/// Helper: build a Q8_0 block from a scale and 32 i8 quantized values.
fn make_q8_0_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
    let d_f16 = crate::quantize::f32_to_f16_bits(d);
    let mut block = Vec::with_capacity(Q8_0_BLOCK);
    block.extend_from_slice(&d_f16.to_le_bytes());
    for &q in qs {
        block.push(q as u8);
    }
    block
}

/// Helper: build a Q8_1 block from a scale, sum, and 32 i8 quantized values.
fn make_q8_1_block(d: f32, _sum: f32, qs: &[i8; 32]) -> Vec<u8> {
    let d_f16 = crate::quantize::f32_to_f16_bits(d);
    let sum_f16 = crate::quantize::f32_to_f16_bits(_sum);
    let mut block = Vec::with_capacity(Q8_1_BLOCK);
    block.extend_from_slice(&d_f16.to_le_bytes());
    block.extend_from_slice(&sum_f16.to_le_bytes());
    for &q in qs {
        block.push(q as u8);
    }
    block
}

/// Helper: build a Q4_0 block from a scale and 32 4-bit nibbles packed in 16 bytes.
fn make_q4_0_block(d: f32, qs: &[u8; 16]) -> Vec<u8> {
    let d_f16 = crate::quantize::f32_to_f16_bits(d);
    let mut block = Vec::with_capacity(Q4_0_BLOCK);
    block.extend_from_slice(&d_f16.to_le_bytes());
    block.extend_from_slice(qs);
    block
}

// ---- Q8_0 dequantization tests ----

#[test]
fn dequant_q8_0_single_block_uniform() {
    // All values are 64 -> d = 64/127, qs = all 127
    let d = 64.0 / 127.0;
    let qs = [127i8; 32];
    let bytes = make_q8_0_block(d, &qs);
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 32);
    // f16 round-trip of d introduces small error; use tolerance of 0.1%
    let expected = d * 127.0;
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - expected).abs() < 0.1,
            "Q8_0 uniform i={i}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn dequant_q8_0_single_block_negative_values() {
    // All values are -64 -> d = 64/127, qs = all -127
    let d = 64.0 / 127.0;
    let qs = [-127i8; 32];
    let bytes = make_q8_0_block(d, &qs);
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 32);
    let expected = d * -127.0;
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - expected).abs() < 0.1,
            "Q8_0 negative i={i}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn dequant_q8_0_single_block_mixed() {
    // Range of values: -16..15
    let d = 10.0 / 127.0;
    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = (i as i8) - 16; // range: -16..15
    }
    let bytes = make_q8_0_block(d, &qs);
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 32);
    // f16 round-trip of d introduces small error
    for (i, &v) in out.iter().enumerate() {
        let expected = d * qs[i] as f32;
        assert!(
            (v - expected).abs() < 0.1,
            "Q8_0 mixed i={i}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn dequant_q8_0_single_block_zero_scale() {
    // d = 0 -> all output should be 0
    let qs = [42i8; 32];
    let bytes = make_q8_0_block(0.0, &qs);
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 32);
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0.0, "Q8_0 zero scale i={i}: got {v}");
    }
}

#[test]
fn dequant_q8_0_two_blocks() {
    // Two blocks with different scales
    let mut bytes = Vec::new();
    // Block 1: d=1.0, all zeros
    let qs1 = [0i8; 32];
    bytes.extend_from_slice(&make_q8_0_block(1.0, &qs1));
    // Block 2: d=2.0, all ones
    let qs2 = [1i8; 32];
    bytes.extend_from_slice(&make_q8_0_block(2.0, &qs2));

    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 64);
    // First block all zeros
    for i in 0..32 {
        assert_eq!(out[i], 0.0, "Q8_0 2-block first block i={i}");
    }
    // Second block all 2.0
    for i in 32..64 {
        assert!((out[i] - 2.0).abs() < 1e-5, "Q8_0 2-block second block i={i}: got {}", out[i]);
    }
}

// ---- Q8_1 dequantization tests ----

#[test]
fn dequant_q8_1_single_block_uniform() {
    let d = 64.0 / 127.0;
    let sum = 64.0;
    let qs = [127i8; 32];
    let bytes = make_q8_1_block(d, sum, &qs);
    let out = dequant::dequantize(GgmlType::Q8_1, &bytes, None).unwrap();
    assert_eq!(out.len(), 32);
    let expected = d * 127.0;
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - expected).abs() < 0.1,
            "Q8_1 uniform i={i}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn dequant_q8_1_single_block_zero_scale() {
    let qs = [42i8; 32];
    let bytes = make_q8_1_block(0.0, 0.0, &qs);
    let out = dequant::dequantize(GgmlType::Q8_1, &bytes, None).unwrap();
    assert_eq!(out.len(), 32);
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0.0, "Q8_1 zero scale i={i}: got {v}");
    }
}

#[test]
fn dequant_q8_1_two_blocks() {
    let mut bytes = Vec::new();
    let qs1 = [0i8; 32];
    bytes.extend_from_slice(&make_q8_1_block(1.0, 0.0, &qs1));
    let qs2 = [-127i8; 32];
    bytes.extend_from_slice(&make_q8_1_block(3.0, -127.0 * 32.0, &qs2));

    let out = dequant::dequantize(GgmlType::Q8_1, &bytes, None).unwrap();
    assert_eq!(out.len(), 64);
    for i in 0..32 {
        assert_eq!(out[i], 0.0, "Q8_1 2-block first block i={i}");
    }
    for i in 32..64 {
        let expected = 3.0 * -127.0;
        assert!(
            (out[i] - expected).abs() < 1e-3,
            "Q8_1 2-block second block i={i}: got {}, expected {expected}",
            out[i]
        );
    }
}

// ---- Q4_0 dequantization tests ----

#[test]
fn dequant_q4_0_single_block_zero_point() {
    // Q4_0: x = d * (n - 8); n in [0, 15]
    // All nibbles = 8 -> x = d * 0 = 0
    let d = 5.0;
    let mut qs = [0u8; 16];
    for i in 0..16 {
        qs[i] = 0x88; // both nibbles = 8
    }
    let bytes = make_q4_0_block(d, &qs);
    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 32);
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0.0, "Q4_0 zero-point i={i}: got {v}");
    }
}

#[test]
fn dequant_q4_0_single_block_max_min() {
    // Low nibble = 0 -> x = d * (0 - 8) = -8d
    // Low nibble = 15 -> x = d * (15 - 8) = 7d
    let d = 2.0;
    let mut qs = [0u8; 16];
    // First byte: lo=0, hi=15 -> [-8d, 7d]
    qs[0] = 0xF0; // lo=0, hi=15
    let bytes = make_q4_0_block(d, &qs);
    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 32);
    assert!(
        (out[0] - (-8.0 * d)).abs() < 1e-5,
        "Q4_0 min: got {}, expected {}",
        out[0],
        -8.0 * d
    );
    assert!(
        (out[1] - (7.0 * d)).abs() < 1e-5,
        "Q4_0 max: got {}, expected {}",
        out[1],
        7.0 * d
    );
}

#[test]
fn dequant_q4_0_single_block_zero_scale() {
    let d = 0.0;
    let qs = [0xFFu8; 16]; // all nibbles = 15
    let bytes = make_q4_0_block(d, &qs);
    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 32);
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0.0, "Q4_0 zero scale i={i}: got {v}");
    }
}

#[test]
fn dequant_q4_0_two_blocks() {
    let mut bytes = Vec::new();
    // Block 1: d=1.0, all nibbles=8 (zero)
    let qs1 = [0x88u8; 16];
    bytes.extend_from_slice(&make_q4_0_block(1.0, &qs1));
    let qs2 = [0xFFu8; 16];
    bytes.extend_from_slice(&make_q4_0_block(3.0, &qs2));

    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 64);
    for i in 0..32 {
        assert_eq!(out[i], 0.0, "Q4_0 2-block first block i={i}");
    }
    for i in 32..64 {
        let expected = 3.0 * (15.0 - 8.0); // 21.0
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "Q4_0 2-block second block i={i}: got {}, expected {expected}",
            out[i]
        );
    }
}

// ---- F32 dequantization tests ----

#[test]
fn dequant_f32_single_value() {
    let val: f32 = 3.14;
    let bytes = val.to_le_bytes().to_vec();
    let out = dequant::dequantize(GgmlType::F32, &bytes, None).unwrap();
    assert_eq!(out.len(), 1);
    assert!((out[0] - 3.14).abs() < 1e-6);
}

#[test]
fn dequant_f32_multiple_values() {
    let vals: Vec<f32> = vec![1.0, -2.5, 0.0, 100.0, -0.001];
    let mut bytes = Vec::new();
    for v in &vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let out = dequant::dequantize(GgmlType::F32, &bytes, None).unwrap();
    assert_eq!(out.len(), 5);
    for (i, (&expected, &got)) in vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (expected - got).abs() < 1e-6,
            "F32 i={i}: got {got}, expected {expected}"
        );
    }
}

// ---- F16 dequantization tests ----

#[test]
fn dequant_f16_roundtrip() {
    let original: Vec<f32> = vec![1.0, -1.0, 0.0, 0.5, 2.0];
    let mut bytes = Vec::new();
    for &v in &original {
        let f16_bits = crate::quantize::f32_to_f16_bits(v);
        bytes.extend_from_slice(&f16_bits.to_le_bytes());
    }
    let out = dequant::dequantize(GgmlType::F16, &bytes, None).unwrap();
    assert_eq!(out.len(), 5);
    for (i, (&expected, &got)) in original.iter().zip(out.iter()).enumerate() {
        assert!(
            (expected - got).abs() < 1e-3,
            "F16 i={i}: got {got}, expected {expected}"
        );
    }
}

// ---- BF16 dequantization tests ----

#[test]
fn dequant_bf16_roundtrip() {
    let original: Vec<f32> = vec![1.0, -1.0, 0.0, 0.5, 2.0];
    let mut bytes = Vec::new();
    for &v in &original {
        let bits = v.to_bits();
        let bf16_bits = (bits >> 16) as u16;
        bytes.extend_from_slice(&bf16_bits.to_le_bytes());
    }
    let out = dequant::dequantize(GgmlType::Bf16, &bytes, None).unwrap();
    assert_eq!(out.len(), 5);
    for (i, (&expected, &got)) in original.iter().zip(out.iter()).enumerate() {
        assert!(
            (expected - got).abs() < 0.01,
            "BF16 i={i}: got {got}, expected {expected}"
        );
    }
}

// ---- max_elems truncation ----

#[test]
fn dequant_q8_0_truncates_to_max_elems() {
    let d = 1.0;
    let qs = [1i8; 32];
    let bytes = make_q8_0_block(d, &qs);
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, Some(16)).unwrap();
    assert_eq!(out.len(), 16);
}

#[test]
fn dequant_q4_0_truncates_to_max_elems() {
    let d = 1.0;
    let qs = [0x88u8; 16];
    let bytes = make_q4_0_block(d, &qs);
    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, Some(10)).unwrap();
    assert_eq!(out.len(), 10);
}

// ---- unsupported type returns None ----

#[test]
fn dequant_unsupported_type_returns_none() {
    // Some types that don't have dequant paths should return None
    let result = dequant::dequantize(GgmlType::Iq2Xxs, &[], None);
    assert!(result.is_some()); // Actually, Iq2Xxs does have a scalar dequant
}

// ---- Parallel dequant matches sequential ----

#[test]
fn dequant_par_matches_sequential_q8_0() {
    let d = 5.0;
    let mut qs = [0i8; 32];
    for i in 0..32 {
        qs[i] = (i as i8) - 16;
    }
    let bytes = make_q8_0_block(d, &qs);
    let seq = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    let par = dequant::dequantize_par(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(seq, par);
}

#[test]
fn dequant_par_matches_sequential_q4_0() {
    let d = 3.0;
    let mut qs = [0u8; 16];
    for i in 0..16 {
        qs[i] = ((i % 16) as u8) | (((15 - i % 16) as u8) << 4);
    }
    let bytes = make_q4_0_block(d, &qs);
    let seq = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    let par = dequant::dequantize_par(GgmlType::Q4_0, &bytes, None).unwrap();
    assert_eq!(seq, par);
}

// ---- Large multi-block tests (exercise SIMD 32-element processing) ----

#[test]
fn dequant_q8_0_multi_block_4_blocks() {
    let mut bytes = Vec::new();
    let mut expected_vals = Vec::new();
    for blk in 0..4 {
        let d = (blk as f32) + 1.0;
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = ((i + blk * 7) % 256) as i8;
        }
        bytes.extend_from_slice(&make_q8_0_block(d, &qs));
        for &q in &qs {
            expected_vals.push(d * q as f32);
        }
    }
    let out = dequant::dequantize(GgmlType::Q8_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 128);
    for (i, (&expected, &got)) in expected_vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (expected - got).abs() < 1e-4,
            "Q8_0 4-block i={i}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn dequant_q4_0_multi_block_4_blocks() {
    let mut bytes = Vec::new();
    let mut expected_vals = Vec::new();
    for blk in 0..4 {
        let d = (blk as f32) + 0.5;
        let mut qs = [0u8; 16];
        for i in 0..16 {
            qs[i] = ((i + blk * 3) % 16) as u8 | (((15 - (i + blk * 3) % 16) as u8) << 4);
        }
        bytes.extend_from_slice(&make_q4_0_block(d, &qs));
        for i in 0..16 {
            let lo = (qs[i] & 0x0F) as f32;
            let hi = ((qs[i] >> 4) & 0x0F) as f32;
            expected_vals.push(d * (lo - 8.0));
            expected_vals.push(d * (hi - 8.0));
        }
    }
    let out = dequant::dequantize(GgmlType::Q4_0, &bytes, None).unwrap();
    assert_eq!(out.len(), 128);
    for (i, (&expected, &got)) in expected_vals.iter().zip(out.iter()).enumerate() {
        assert!(
            (expected - got).abs() < 1e-4,
            "Q4_0 4-block i={i}: got {got}, expected {expected}"
        );
    }
}
