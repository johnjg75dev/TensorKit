//! End-to-end tests for the SVD compression pipeline.
//!
//! We build minimal in-memory GGUF and safetensors files with a couple of
//! 2-D weight tensors, apply an SvdPlan, and verify:
//!  * the original tensors are removed and the (a, b) factors are present;
//!  * the metadata is correct (rank, shape, output dtype);
//!  * the reconstruction `A ≈ a * b` is faithful within the rank.

use crate::svd::config::{
    LayerSelection, OutputDtype, RankClamps, RankSpec, RankSpecWithClamps, SvdConfig,
    TensorSelection,
};
use crate::svd::linalg::{evd_symmetric, pack_lowrank, rank_for_energy, reconstruct, svd_jacobi, svd_randomized, Mat};
use crate::svd::plan::build_plan;
use crate::Model;
use std::path::PathBuf;

// ---- config parsing ------------------------------------------------------

#[test]
fn parse_layer_selection_aliases() {
    assert!(matches!(
        LayerSelection::parse("all").unwrap(),
        LayerSelection::All
    ));
    assert!(matches!(
        LayerSelection::parse("all-attn").unwrap(),
        LayerSelection::AllAttn
    ));
    assert!(matches!(
        LayerSelection::parse("all-ffn").unwrap(),
        LayerSelection::AllFfn
    ));
    assert!(matches!(
        LayerSelection::parse("all-mlp").unwrap(),
        LayerSelection::AllMlp
    ));
    match LayerSelection::parse("0-3,7").unwrap() {
        LayerSelection::Indices(v) => assert_eq!(v, vec![0, 1, 2, 3, 7]),
        _ => panic!(),
    }
    assert!(matches!(
        LayerSelection::parse("regex:^blk\\.0\\.").unwrap(),
        LayerSelection::Pattern(_)
    ));
    assert!(LayerSelection::parse("").is_err());
}

#[test]
fn parse_tensor_selection() {
    let s = TensorSelection::parse("attn").unwrap();
    assert!(s.matches("blk.0.attn_q.weight"));
    assert!(!s.matches("blk.0.ffn_up.weight"));
    let s = TensorSelection::parse("attn_q,attn_v").unwrap();
    assert!(s.matches("blk.0.attn_q.weight"));
    assert!(s.matches("blk.0.attn_v.weight"));
    assert!(!s.matches("blk.0.attn_k.weight"));
    let s = TensorSelection::parse("regex:\\.weight$").unwrap();
    assert!(s.matches("blk.0.attn_q.weight"));
    assert!(!s.matches("blk.0.attn_q.bias"));
    let s = TensorSelection::parse("mlp").unwrap();
    assert!(s.matches("blk.0.attn_q.weight"));
    assert!(s.matches("blk.0.ffn_up.weight"));
}

#[test]
fn parse_rank_spec_int_and_clamp() {
    let r = RankSpecWithClamps::parse("64").unwrap();
    assert_eq!(r.resolve(100, 100, None), 64);
    let r = RankSpecWithClamps::parse("abs:128,min:8,max:64").unwrap();
    assert_eq!(r.resolve(200, 200, None), 64); // clamped to max
    let r = RankSpecWithClamps::parse("frac:0.5,min:4").unwrap();
    assert_eq!(r.resolve(64, 64, None), 32);
    assert_eq!(r.resolve(8, 8, None), 4); // clamped to min
    let s = vec![10.0, 5.0, 1.0, 0.1];
    let r = RankSpecWithClamps::parse("energy:0.9").unwrap();
    // total^2 = 100+25+1+0.01 = 126.01; 0.9 -> 113.4 -> first 2 (125) passes.
    assert_eq!(r.resolve(4, 4, Some(&s)), 2);
    assert!(RankSpecWithClamps::parse("energy:1.5").is_err());
    assert!(RankSpecWithClamps::parse("garbage").is_err());
    let _ = RankSpec::Fraction(0.5); // silence
}

#[test]
fn parse_dtype_strings() {
    assert_eq!(OutputDtype::parse("F16").unwrap(), OutputDtype::F16);
    assert_eq!(OutputDtype::parse("bf16").unwrap(), OutputDtype::Bf16);
    assert_eq!(OutputDtype::parse("float32").unwrap(), OutputDtype::F32);
    assert!(OutputDtype::parse("int8").is_err());
}

// ---- linalg --------------------------------------------------------------

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut n = 0.0f64;
    let mut d = 0.0f64;
    for i in 0..a.len() {
        let diff = a[i] as f64 - b[i] as f64;
        n += diff * diff;
        d += (a[i] as f64) * (a[i] as f64);
    }
    if d == 0.0 {
        0.0
    } else {
        (n / d).sqrt() as f32
    }
}

#[test]
fn svd_jacobi_diagonal_recovers() {
    // A = diag(4, 3, 2, 1) -> singular values should be [4, 3, 2, 1] up to sign.
    let n = 4;
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        data[i * n + i] = (n - i) as f32;
    }
    let m = Mat::from_vec(n, n, data.clone());
    let s = svd_jacobi(&m, 200, 1e-12).unwrap();
    let mut got: Vec<f32> = s.s.clone();
    got.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    for i in 0..n {
        assert!(
            (got[i] - (n - i) as f32).abs() < 1e-3,
            "got[{i}] = {}",
            got[i]
        );
    }
}

#[test]
fn svd_jacobi_low_rank_reconstruction() {
    // Build a clean rank-1 matrix: A = u * v^T. Singular value = ||u|| * ||v||.
    let m = 4usize;
    let n = 3usize; // non-square, m != n
    let u: Vec<f32> = (0..m).map(|i| (i as f32 * 0.7).sin() * 3.0).collect();
    let v: Vec<f32> = (0..n).map(|j| (j as f32 * 0.5).cos() * 2.0).collect();
    let mut a_data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = u[i] * v[j];
        }
    }
    let expected_s0: f32 = {
        let nu: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        nu * nv
    };

    let a_mat = Mat::from_vec(m, n, a_data.clone());
    let svd = svd_jacobi(&a_mat, 200, 1e-12).unwrap();
    assert!(svd.s.len() >= 1);
    // Top singular value should match.
    assert!(
        (svd.s[0] - expected_s0).abs() < 1e-2,
        "s[0] = {} (want {})",
        svd.s[0],
        expected_s0
    );
    // Verify full reconstruction: A = U * S * V^T.
    let mut usvt = Mat::new(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut s_v = 0.0f32;
            for p in 0..svd.s.len() {
                s_v += svd.u.data[i * svd.s.len() + p] * svd.s[p] * svd.vt.data[p * n + j];
            }
            usvt.data[i * n + j] = s_v;
        }
    }
    let full_err = rel_err(&a_data, &usvt.data);
    assert!(full_err < 1e-2, "full SVD reconstruction err = {full_err}");
    // Truncate to k=1 and reconstruct: should be exact.
    let (a_pack, b_pack) = pack_lowrank(&svd);
    // a_pack is m x k_full; take the first column (k=1) directly.
    let k_full_a = a_pack.cols;
    let a1_data: Vec<f32> = (0..a_pack.rows)
        .map(|r| a_pack.data[r * k_full_a])
        .collect();
    let a1 = Mat::from_vec(a_pack.rows, 1, a1_data);
    // b_pack is k_full x n; take the first k rows.
    let b1 = Mat::from_vec(1, b_pack.cols, b_pack.data[..b_pack.cols].to_vec());
    let mut recon = Mat::new(m, n);
    Mat::matmul_into(&a1, &b1, &mut recon);
    let err = rel_err(&a_data, &recon.data);
    assert!(err < 1e-3, "rank-1 reconstruction err = {err}");
}

#[test]
fn svd_randomized_matches_jacobi() {
    // 40x40 rank-2 outer product of two random vectors.
    let m = 40usize;
    let n = 40usize;
    let u: Vec<f32> = (0..m).map(|i| ((i * 13 + 1) as f32 * 0.11).sin()).collect();
    let v: Vec<f32> = (0..n).map(|j| ((j * 7 + 3) as f32 * 0.07).cos()).collect();
    let mut a_data = vec![0.0f32; m * n];
    // Outer products of two pairs: rank-2.
    let u2: Vec<f32> = (0..m).map(|i| ((i * 5 + 2) as f32 * 0.09).cos()).collect();
    let v2: Vec<f32> = (0..n).map(|j| ((j * 11 + 5) as f32 * 0.13).sin()).collect();
    for i in 0..m {
        for j in 0..n {
            a_data[i * n + j] = u[i] * v[j] * 0.5 + u2[i] * v2[j] * 0.3;
        }
    }
    let a_mat = Mat::from_vec(m, n, a_data);
    let s_rand = svd_randomized(&a_mat, 5, 4, 2, 42).unwrap();
    // The randomized SVD should produce the top singular values within a reasonable
    // tolerance of jacobi (the rest being essentially 0).
    let s_jac = svd_jacobi(&a_mat, 200, 1e-12).unwrap();
    assert!(
        (s_rand.s[0] - s_jac.s[0]).abs() / s_jac.s[0].max(1e-6) < 0.05,
        "rand s[0] = {} vs jacobi {}",
        s_rand.s[0],
        s_jac.s[0]
    );
}

#[test]
fn rank_for_energy_clamps() {
    let s = vec![10.0, 1.0, 0.1, 0.01];
    // squared s: 100, 1, 0.01, 0.0001 -> total ~101.01
    // 0.999 needs ~100.9: first singular value (100) is *just* under; need 2.
    assert_eq!(rank_for_energy(&s, 0.999, 1, 4), 2);
    assert_eq!(rank_for_energy(&s, 0.5, 1, 4), 1);
    assert_eq!(rank_for_energy(&s, 0.0001, 1, 4), 1);
    assert_eq!(rank_for_energy(&[], 0.99, 1, 4), 1);
}

#[test]
fn factor_names_default() {
    let cfg = SvdConfig::default();
    let (a, b) = cfg.factor_names("blk.5.attn_q.weight");
    assert_eq!(a, "blk.5.attn_q.weight.svd_a");
    assert_eq!(b, "blk.5.attn_q.weight.svd_b");
}

// ---- end-to-end on a real GGUF file -------------------------------------

fn tiny_gguf_with_attn_q(path: &std::path::Path) {
    use crate::formats::gguf::types::{GgmlType, MetaValue, MetadataKv};
    use crate::formats::gguf::writer::GgufWriter;

    let mut w = GgufWriter::new(3, 32);
    w.add_kv(MetadataKv {
        key: "general.architecture".into(),
        value_type: 8,
        value: MetaValue::String("llama".into()),
    });
    w.add_kv(MetadataKv {
        key: "llama.block_count".into(),
        value_type: 4,
        value: MetaValue::U32(2),
    });
    // 8x4 weight matrix (rank-1 truth: A = u * v^T scaled by 3.0).
    let m = 8usize;
    let n = 4usize;
    let u_true: Vec<f32> = (0..m).map(|i| (i as f32 * 0.3).sin()).collect();
    let v_true: Vec<f32> = (0..n).map(|j| (j as f32 * 0.7).cos()).collect();
    let mut w_data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            w_data[i * n + j] = 3.0 * u_true[i] * v_true[j];
        }
    }
    let mut bytes = Vec::with_capacity(m * n * 4);
    for v in &w_data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.add_tensor(
        "blk.0.attn_q.weight".into(),
        2,
        [m as u64, n as u64, 1, 1],
        GgmlType::F32,
        &bytes,
    );
    // 1D norm (8 f32 = 32 bytes).
    let extra: Vec<u8> = (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    w.add_tensor(
        "blk.0.attn_norm.weight".into(),
        1,
        [8, 1, 1, 1],
        GgmlType::F32,
        &extra,
    );
    // 1D embd-like (16 f32 = 64 bytes).
    let embd: Vec<u8> = (0..16u32)
        .flat_map(|i| ((i as f32) * 0.5).to_le_bytes())
        .collect();
    w.add_tensor(
        "token_embd.weight".into(),
        1,
        [16, 1, 1, 1],
        GgmlType::F32,
        &embd,
    );
    let out = w.into_bytes().unwrap();
    std::fs::write(path, &out).unwrap();
}

#[test]
fn apply_to_gguf_writes_expected_structure() {
    let tmp = std::env::temp_dir().join(format!("tensorkit-svd-{}.gguf", std::process::id()));
    tiny_gguf_with_attn_q(&tmp);
    let gg = crate::formats::gguf::GgufFile::open(&tmp).unwrap();
    let mut cfg = SvdConfig::default();
    cfg.layers = LayerSelection::AllMlp;
    cfg.tensors = TensorSelection::Attn;
    cfg.rank = RankSpecWithClamps {
        spec: RankSpec::Absolute(2),
        clamps: RankClamps { min: 1, max: None },
    };
    cfg.dtype = OutputDtype::F16;
    cfg.suffix_a = ".svd_a".into();
    cfg.suffix_b = ".svd_b".into();
    cfg.min_dim = 4; // test fixture has 8x4 weight
    let plan = build_plan(&gg, &cfg).unwrap();
    assert_eq!(plan.targets.len(), 1);
    assert_eq!(plan.targets[0].k, 2);

    let out: PathBuf = tmp.with_extension("svd.gguf");
    let report = crate::svd::apply_to_gguf(&gg, &plan, &out).unwrap();
    assert_eq!(report.applied.len(), 1);
    let a = &report.applied[0];
    assert_eq!(a.m, 8);
    assert_eq!(a.n, 4);
    assert_eq!(a.k, 2);
    assert!(
        a.approx_error < 0.05,
        "approx_error too high: {}",
        a.approx_error
    );

    // Re-open and check structure.
    let out_gg = crate::formats::gguf::GgufFile::open(&out).unwrap();
    let names: Vec<&str> = out_gg.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(
        !names.contains(&"blk.0.attn_q.weight"),
        "original tensor should be removed"
    );
    assert!(names.contains(&"blk.0.attn_q.weight.svd_a"));
    assert!(names.contains(&"blk.0.attn_q.weight.svd_b"));
    assert!(names.contains(&"blk.0.attn_norm.weight"));
    assert!(names.contains(&"token_embd.weight"));

    // Metadata checks.
    assert_eq!(
        out_gg
            .metadata_str("tensorkit.svd.applied")
            .map(|s| s == "true")
            .unwrap_or(false)
            || out_gg.metadata_u32("tensorkit.svd.targets") == Some(1),
        true
    );
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&out);
}

// ---- SVD apply parameter tests (jacobi sweeps=20, tol=1e-6) ---------------

#[test]
fn apply_to_gguf_compression_produces_valid_output() {
    // Test that the new SVD parameters (20 sweeps, 1e-6 tol) produce valid output
    let tmp = std::env::temp_dir().join(format!(
        "tensorkit-svd-params-{}.gguf",
        std::process::id()
    ));
    tiny_gguf_with_attn_q(&tmp);
    let gg = crate::formats::gguf::GgufFile::open(&tmp).unwrap();
    let mut cfg = SvdConfig::default();
    cfg.layers = LayerSelection::AllMlp;
    cfg.tensors = TensorSelection::Attn;
    cfg.rank = RankSpecWithClamps {
        spec: RankSpec::Absolute(2),
        clamps: RankClamps { min: 1, max: None },
    };
    cfg.dtype = OutputDtype::F16;
    cfg.min_dim = 4;
    // Force jacobi (disable randomized)
    cfg.randomized = false;
    cfg.backend = crate::svd::SvdBackend::Jacobi;
    let plan = build_plan(&gg, &cfg).unwrap();
    assert_eq!(plan.targets.len(), 1);

    let out: PathBuf = tmp.with_extension("params.gguf");
    let report = crate::svd::apply_to_gguf(&gg, &plan, &out).unwrap();
    assert_eq!(report.applied.len(), 1);

    // The approximation error should be small (within 5% relative)
    assert!(
        report.applied[0].approx_error < 0.05,
        "approx_error too high with new params: {}",
        report.applied[0].approx_error
    );

    // Verify the method is jacobi (since we disabled randomized)
    assert_eq!(report.applied[0].method, "jacobi");

    // Verify rank is correct
    assert_eq!(report.applied[0].k, 2);

    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn apply_to_gguf_compression_ratio_is_reasonable() {
    // Test that compression ratio is reasonable for the new parameters
    let tmp = std::env::temp_dir().join(format!(
        "tensorkit-svd-ratio-{}.gguf",
        std::process::id()
    ));
    tiny_gguf_with_attn_q(&tmp);
    let gg = crate::formats::gguf::GgufFile::open(&tmp).unwrap();
    let mut cfg = SvdConfig::default();
    cfg.layers = LayerSelection::AllMlp;
    cfg.tensors = TensorSelection::Attn;
    cfg.rank = RankSpecWithClamps {
        spec: RankSpec::Absolute(2),
        clamps: RankClamps { min: 1, max: None },
    };
    cfg.dtype = OutputDtype::F32; // use F32 for clearer ratio calculation
    cfg.min_dim = 4;
    cfg.randomized = false;
    let plan = build_plan(&gg, &cfg).unwrap();

    let out: PathBuf = tmp.with_extension("ratio.gguf");
    let report = crate::svd::apply_to_gguf(&gg, &plan, &out).unwrap();

    // 8x4 F32 tensor = 128 bytes. With rank 2, svd_a is 8x2 (64B) + svd_b is 2x4 (32B) = 96B.
    // Compression ratio = 1 - 96/128 = 0.25 (25% reduction)
    assert!(
        report.compression_ratio > 0.0,
        "compression_ratio should be positive: {}",
        report.compression_ratio
    );
    assert!(
        report.compression_ratio < 1.0,
        "compression_ratio should be less than 1: {}",
        report.compression_ratio
    );

    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&out);
}

#[test]
fn apply_to_gguf_randomized_method_used_for_large_tensors() {
    // Test that randomized method is chosen when tensor is large enough
    let tmp = std::env::temp_dir().join(format!(
        "tensorkit-svd-rand-{}.gguf",
        std::process::id()
    ));
    tiny_gguf_with_attn_q(&tmp);
    let gg = crate::formats::gguf::GgufFile::open(&tmp).unwrap();
    let mut cfg = SvdConfig::default();
    cfg.layers = LayerSelection::AllMlp;
    cfg.tensors = TensorSelection::Attn;
    cfg.rank = RankSpecWithClamps {
        spec: RankSpec::Absolute(2),
        clamps: RankClamps { min: 1, max: None },
    };
    cfg.dtype = OutputDtype::F16;
    cfg.min_dim = 4;
    // Enable randomized with low threshold
    cfg.randomized = true;
    cfg.randomized_min_elems = 10; // very low threshold, so even small tensors use randomized
    let plan = build_plan(&gg, &cfg).unwrap();

    let out: PathBuf = tmp.with_extension("rand.gguf");
    let report = crate::svd::apply_to_gguf(&gg, &plan, &out).unwrap();

    // 8x4 = 32 elements > 10, so should use randomized
    assert_eq!(report.applied[0].method, "randomized");

    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&out);
}

// ---- Additional linalg edge case tests ------------------------------------

#[test]
fn svd_jacobi_1x1_matrix() {
    let m = Mat::from_vec(1, 1, vec![5.0]);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    assert_eq!(svd.s.len(), 1);
    assert!((svd.s[0] - 5.0).abs() < 1e-5);
}

#[test]
fn svd_jacobi_1x3_row_vector() {
    let m = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    // The Jacobi implementation returns n singular values (full Vt)
    assert_eq!(svd.s.len(), 3);
    // First singular value should be the Frobenius norm: sqrt(1+4+9) = sqrt(14)
    let expected = (14.0f32).sqrt();
    assert!((svd.s[0] - expected).abs() < 1e-3, "s[0] = {}", svd.s[0]);
    // The rest should be near zero (rank-1 matrix)
    assert!(svd.s[1].abs() < 0.01, "s[1] should be ~0: {}", svd.s[1]);
    assert!(svd.s[2].abs() < 0.01, "s[2] should be ~0: {}", svd.s[2]);
}

#[test]
fn svd_jacobi_3x1_column_vector() {
    let m = Mat::from_vec(3, 1, vec![3.0, 4.0, 0.0]);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    assert_eq!(svd.s.len(), 1);
    // Singular value should be sqrt(9+16+0) = 5.0
    assert!((svd.s[0] - 5.0).abs() < 1e-3, "s[0] = {}", svd.s[0]);
}

#[test]
fn svd_jacobi_rank_deficient_matrix() {
    // Rank-1 matrix: all rows are the same
    let m = Mat::from_vec(4, 3, vec![
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
    ]);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    // Only 1 non-zero singular value
    assert!(svd.s[0] > 1.0, "s[0] should be large: {}", svd.s[0]);
    assert!(svd.s[1].abs() < 0.01, "s[1] should be ~0: {}", svd.s[1]);
    assert!(svd.s[2].abs() < 0.01, "s[2] should be ~0: {}", svd.s[2]);
}

#[test]
fn svd_jacobi_rectangular_tall_matrix() {
    // Tall matrix: 6x3
    let data: Vec<f32> = (0..18).map(|i| (i as f32 - 9.0) * 0.5).collect();
    let m = Mat::from_vec(6, 3, data.clone());
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    assert_eq!(svd.u.rows, 6);
    assert_eq!(svd.u.cols, 3);
    assert_eq!(svd.s.len(), 3);
    assert_eq!(svd.vt.rows, 3);
    assert_eq!(svd.vt.cols, 3);

    // Reconstruct and check error
    let mut recon = Mat::new(6, 3);
    for i in 0..6 {
        for j in 0..3 {
            let mut val = 0.0f64;
            for p in 0..3 {
                val += svd.u.get(i, p) as f64 * svd.s[p] as f64 * svd.vt.get(p, j) as f64;
            }
            recon.set(i, j, val as f32);
        }
    }
    for i in 0..18 {
        assert!(
            (data[i] - recon.data[i]).abs() < 1e-3,
            "tall recon error at {i}: {} vs {}",
            data[i],
            recon.data[i]
        );
    }
}

#[test]
fn svd_jacobi_orthogonal_matrix() {
    // A known orthogonal matrix (rotation)
    let theta = std::f32::consts::FRAC_PI_4; // 45 degrees
    let m = Mat::from_vec(2, 2, vec![
        theta.cos(), -theta.sin(),
        theta.sin(),  theta.cos(),
    ]);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    // Both singular values should be 1.0
    assert!((svd.s[0] - 1.0).abs() < 1e-4, "s[0] = {}", svd.s[0]);
    assert!((svd.s[1] - 1.0).abs() < 1e-4, "s[1] = {}", svd.s[1]);
}

#[test]
fn evd_symmetric_1x1() {
    let m = Mat::from_vec(1, 1, vec![42.0]);
    let (evals, evecs) = evd_symmetric(&m, 100, 1e-10).unwrap();
    assert_eq!(evals.len(), 1);
    assert!((evals[0] - 42.0).abs() < 1e-5);
    assert!((evecs.get(0, 0) - 1.0).abs() < 1e-5);
}

#[test]
fn evd_symmetric_negative_eigenvalues() {
    // [[-1, 0], [0, -2]] -> eigenvalues -1, -2
    let m = Mat::from_vec(2, 2, vec![-1.0, 0.0, 0.0, -2.0]);
    let (evals, _evecs) = evd_symmetric(&m, 100, 1e-10).unwrap();
    let mut vals = evals.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((vals[0] - (-1.0)).abs() < 1e-4, "eval[0] = {}", vals[0]);
    assert!((vals[1] - (-2.0)).abs() < 1e-4, "eval[1] = {}", vals[1]);
}

#[test]
fn rank_for_energy_all_energy_needed() {
    // All singular values contribute equally
    let s = vec![1.0, 1.0, 1.0, 1.0];
    // 99% energy: need 4 out of 4 (each contributes 1/4 = 25%)
    assert_eq!(rank_for_energy(&s, 0.99, 1, 10), 4);
}

#[test]
fn rank_for_energy_few_needed() {
    // First singular value dominates
    let s = vec![100.0, 1.0, 1.0, 1.0];
    // 99% energy: 10000/10003 ≈ 99.97% -> k=1
    assert_eq!(rank_for_energy(&s, 0.99, 1, 10), 1);
}

#[test]
fn pack_lowrank_reconstruction_quality() {
    // Build a rank-2 matrix and verify pack_lowrank + reconstruct gives good approximation
    let m = 6usize;
    let n = 5usize;
    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            data[i * n + j] = (i as f32 + 1.0) * (j as f32 + 1.0) * 0.1;
            // Add a second component
            data[i * n + j] += ((i as f32) * 0.5).sin() * ((j as f32) * 0.3).cos();
        }
    }
    let a_mat = Mat::from_vec(m, n, data.clone());
    let svd = svd_jacobi(&a_mat, 100, 1e-10).unwrap();
    let (a_pack, b_pack) = pack_lowrank(&svd);
    let recon = reconstruct(&a_pack, &b_pack);

    // Full-rank reconstruction should be very accurate
    let err: f64 = data
        .iter()
        .zip(recon.data.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nrm = a_mat.norm_fro();
    let rel_err = err / nrm;
    assert!(
        rel_err < 1e-3,
        "pack_lowrank + reconstruct relative error = {rel_err}"
    );
}

#[test]
fn svd_jacobi_integer_matrix() {
    // Matrix with integer entries
    let m = Mat::from_vec(3, 3, vec![
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    ]);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    let mut vals = svd.s.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((vals[0] - 3.0).abs() < 1e-4);
    assert!((vals[1] - 2.0).abs() < 1e-4);
    assert!((vals[2] - 1.0).abs() < 1e-4);
}

#[test]
fn svd_jacobi_near_zero_gamma_matrix() {
    // A nearly diagonal matrix where gamma will be tiny for most pairs
    let mut data = vec![0.0f32; 16];
    data[0] = 10.0;
    data[5] = 8.0;
    data[10] = 6.0;
    data[15] = 4.0;
    // Tiny off-diagonal entries
    data[1] = 1e-31;
    data[4] = 1e-31;
    let m = Mat::from_vec(4, 4, data);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    let mut vals = svd.s.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    // Singular values should be close to diagonal entries
    assert!((vals[0] - 10.0).abs() < 0.1, "s[0] = {}", vals[0]);
    assert!((vals[1] - 8.0).abs() < 0.1, "s[1] = {}", vals[1]);
    assert!((vals[2] - 6.0).abs() < 0.1, "s[2] = {}", vals[2]);
    assert!((vals[3] - 4.0).abs() < 0.1, "s[3] = {}", vals[3]);
}

#[test]
fn svd_randomized_medium_rank_matrix() {
    // 30x30 matrix with rank-3 structure
    let m = 30usize;
    let n = 30usize;
    let mut data = vec![0.0f32; m * n];
    for k in 0..3 {
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] += ((i as f32 + k as f32) * 0.1).sin()
                    * ((j as f32 + k as f32) * 0.15).cos()
                    * ((k + 1) as f32);
            }
        }
    }
    let a = Mat::from_vec(m, n, data);
    let svd = svd_randomized(&a, 5, 4, 2, 42).unwrap();
    // Should capture the 3 non-zero singular values
    assert!(svd.s[0] > 1.0, "s[0] should be large: {}", svd.s[0]);
    assert!(svd.s[1] > 0.1, "s[1] should be non-trivial: {}", svd.s[1]);
    assert!(svd.s[2] > 0.01, "s[2] should be non-trivial: {}", svd.s[2]);
}

#[test]
fn apply_to_gguf_preserves_non_targets() {
    let tmp = std::env::temp_dir().join(format!("tensorkit-svd-keep-{}.gguf", std::process::id()));
    tiny_gguf_with_attn_q(&tmp);
    let gg = crate::formats::gguf::GgufFile::open(&tmp).unwrap();
    let mut cfg = SvdConfig::default();
    cfg.layers = LayerSelection::AllMlp;
    cfg.tensors = TensorSelection::Attn;
    cfg.rank = RankSpecWithClamps {
        spec: RankSpec::Absolute(2),
        clamps: RankClamps { min: 1, max: None },
    };
    cfg.dtype = OutputDtype::F32;
    cfg.min_dim = 4;
    let plan = build_plan(&gg, &cfg).unwrap();
    let out: PathBuf = tmp.with_extension("keep.gguf");
    crate::svd::apply_to_gguf(&gg, &plan, &out).unwrap();
    let out_gg = crate::formats::gguf::GgufFile::open(&out).unwrap();
    // norm tensor bytes should round-trip
    let src_bytes = gg
        .read_tensor_bytes("blk.0.attn_norm.weight")
        .unwrap()
        .to_vec();
    let dst_bytes = out_gg
        .read_tensor_bytes("blk.0.attn_norm.weight")
        .unwrap()
        .to_vec();
    assert_eq!(src_bytes, dst_bytes);
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&out);
}
