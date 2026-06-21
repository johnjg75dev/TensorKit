use tensorkit::svd::linalg::*;

// ---- AlignedVec tests ----

#[test]
fn aligned_vec_creation_and_len() {
    let v: AlignedVec<f32> = AlignedVec::with_capacity(16);
    assert_eq!(v.len(), 0);
    assert_eq!(v.as_ptr() as usize % 32, 0, "pointer must be 32-byte aligned");
}

#[test]
fn aligned_vec_push_and_deref() {
    let mut v: AlignedVec<f32> = AlignedVec::with_capacity(4);
    v.push(1.0);
    v.push(2.0);
    v.push(3.0);
    assert_eq!(v.len(), 3);
    assert_eq!(&*v, &[1.0, 2.0, 3.0]);
}

#[test]
fn aligned_vec_deref_mut() {
    let mut v: AlignedVec<f32> = AlignedVec::with_capacity(4);
    v.push(10.0);
    v.push(20.0);
    v[0] = 99.0;
    assert_eq!(v[0], 99.0);
    assert_eq!(v[1], 20.0);
}

#[test]
fn aligned_vec_clone() {
    let mut v: AlignedVec<f32> = AlignedVec::with_capacity(4);
    v.push(5.0);
    v.push(10.0);
    let cloned = v.clone();
    assert_eq!(&*cloned, &[5.0, 10.0]);
    assert_eq!(cloned.len(), 2);
    assert_eq!(cloned.as_ptr() as usize != v.as_ptr() as usize, true);
}

#[test]
fn aligned_vec_into_iter() {
    let mut v: AlignedVec<f32> = AlignedVec::with_capacity(4);
    v.push(1.0);
    v.push(2.0);
    v.push(3.0);
    let sum: f32 = v.iter().sum();
    assert!((sum - 6.0).abs() < 1e-10);
}

#[test]
fn aligned_vec_set_len() {
    let mut v: AlignedVec<f32> = AlignedVec::with_capacity(8);
    for i in 0..8 {
        v.push(i as f32);
    }
    v.set_len(4);
    assert_eq!(v.len(), 4);
}

#[test]
#[should_panic]
fn aligned_vec_set_len_exceeds_capacity() {
    let mut v: AlignedVec<f32> = AlignedVec::with_capacity(4);
    v.set_len(5);
}

#[test]
fn aligned_vec_debug() {
    let mut v: AlignedVec<f32> = AlignedVec::with_capacity(4);
    v.push(1.0);
    let debug = format!("{:?}", v);
    assert!(debug.contains("1.0"));
}

// ---- Mat tests ----

#[test]
fn mat_new_creates_zeroed_aligned() {
    let m = Mat::new(3, 4);
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 4);
    assert_eq!(m.data.len(), 12);
    for &v in &*m.data {
        assert_eq!(v, 0.0);
    }
    assert_eq!(m.data.as_ptr() as usize % 32, 0);
}

#[test]
fn mat_from_vec_creates_aligned() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = Mat::from_vec(2, 3, data);
    assert_eq!(m.rows, 2);
    assert_eq!(m.cols, 3);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(0, 2), 3.0);
    assert_eq!(m.get(1, 0), 4.0);
    assert_eq!(m.get(1, 2), 6.0);
    assert_eq!(m.data.as_ptr() as usize % 32, 0);
}

#[test]
fn mat_get_set() {
    let mut m = Mat::new(2, 3);
    m.set(0, 0, 7.0);
    m.set(1, 2, 13.0);
    assert_eq!(m.get(0, 0), 7.0);
    assert_eq!(m.get(1, 2), 13.0);
    assert_eq!(m.get(0, 1), 0.0);
}

#[test]
fn mat_norm_fro() {
    let m = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let expected = 30.0_f64.sqrt();
    let nrm = m.norm_fro();
    assert!((nrm - expected).abs() < 1e-5, "norm_fro = {nrm}, expected {expected}");
}

#[test]
fn mat_norm_fro_zero() {
    let m = Mat::new(3, 3);
    assert_eq!(m.norm_fro(), 0.0);
}

// ---- matmul tests ----

#[test]
fn matmul_identity() {
    let a = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Mat::new(3, 3);
    let mut b = b;
    for i in 0..3 {
        b.set(i, i, 1.0);
    }
    let mut out = Mat::new(2, 3);
    Mat::matmul_into(&a, &b, &mut out);
    for i in 0..6 {
        assert!((a.data[i] - out.data[i]).abs() < 1e-5);
    }
}

#[test]
fn matmul_2x2_times_2x2() {
    let a = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Mat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
    let mut out = Mat::new(2, 2);
    Mat::matmul_into(&a, &b, &mut out);
    assert!((out.get(0, 0) - 19.0).abs() < 1e-5);
    assert!((out.get(0, 1) - 22.0).abs() < 1e-5);
    assert!((out.get(1, 0) - 43.0).abs() < 1e-5);
    assert!((out.get(1, 1) - 50.0).abs() < 1e-5);
}

#[test]
fn matmul_rectangular() {
    let a = Mat::from_vec(2, 3, vec![1.0, 0.0, 2.0, 0.0, 3.0, 1.0]);
    let b = Mat::from_vec(3, 2, vec![4.0, 1.0, 2.0, 0.0, 0.0, 5.0]);
    let mut out = Mat::new(2, 2);
    Mat::matmul_into(&a, &b, &mut out);
    assert!((out.get(0, 0) - 4.0).abs() < 1e-5);
    assert!((out.get(0, 1) - 11.0).abs() < 1e-5);
    assert!((out.get(1, 0) - 6.0).abs() < 1e-5);
    assert!((out.get(1, 1) - 5.0).abs() < 1e-5);
}

#[test]
#[should_panic(expected = "inner dims must match")]
fn matmul_dimension_mismatch_panics() {
    let a = Mat::new(2, 3);
    let b = Mat::new(4, 2);
    let mut out = Mat::new(2, 2);
    Mat::matmul_into(&a, &b, &mut out);
}

// ---- jacobi_2x2 tests ----

#[test]
fn jacobi_2x2_already_diagonal() {
    let (c, s) = jacobi_2x2(5.0, 3.0, 0.0);
    assert!((c - 1.0).abs() < 1e-15);
    assert!(s.abs() < 1e-15);
}

#[test]
fn jacobi_2x2_symmetric_offdiag() {
    let (c, s) = jacobi_2x2(5.0, 3.0, 3.0);
    let a_f = 5.0f64;
    let b_f = 3.0f64;
    let g_f = 3.0f64;
    let g_prime = s * c * (b_f - a_f) + (c * c - s * s) * g_f;
    assert!(g_prime.abs() < 1e-10, "off-diagonal should be ~0, got {g_prime}");
}

#[test]
fn jacobi_2x2_negative_offdiag() {
    let (c, s) = jacobi_2x2(4.0, 2.0, -3.0);
    let a_f = 4.0f64;
    let b_f = 2.0f64;
    let g_f = -3.0f64;
    let g_prime = s * c * (b_f - a_f) + (c * c - s * s) * g_f;
    assert!(g_prime.abs() < 1e-10, "off-diagonal should be ~0, got {g_prime}");
}

#[test]
fn jacobi_2x2_large_condition_number() {
    let (c, s) = jacobi_2x2(1000.0, 1.0, 10.0);
    let a_f = 1000.0f64;
    let b_f = 1.0f64;
    let g_f = 10.0f64;
    let g_prime = s * c * (b_f - a_f) + (c * c - s * s) * g_f;
    assert!(g_prime.abs() < 1e-6, "off-diagonal should be ~0, got {g_prime}");
}

// ---- svd_jacobi tests ----

#[test]
fn svd_jacobi_empty_matrix_returns_error() {
    let m = Mat::new(0, 5);
    assert!(svd_jacobi(&m, 100, 1e-6).is_err());
    let m = Mat::new(5, 0);
    assert!(svd_jacobi(&m, 100, 1e-6).is_err());
}

#[test]
fn svd_jacobi_2x2_identity() {
    let m = Mat::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    assert_eq!(svd.s.len(), 2);
    assert!((svd.s[0] - 1.0).abs() < 1e-4);
    assert!((svd.s[1] - 1.0).abs() < 1e-4);
}

#[test]
fn svd_jacobi_2x2_diagonal() {
    let m = Mat::from_vec(2, 2, vec![3.0, 0.0, 0.0, 1.0]);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    let mut vals = svd.s.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((vals[0] - 3.0).abs() < 1e-4);
    assert!((vals[1] - 1.0).abs() < 1e-4);
}

#[test]
fn svd_jacobi_3x3_random_reconstruction() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0];
    let m = Mat::from_vec(3, 3, data.clone());
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();

    let mut recon = Mat::new(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            let mut val = 0.0f64;
            for p in 0..svd.s.len() {
                val += svd.u.get(i, p) as f64 * svd.s[p] as f64 * svd.vt.get(p, j) as f64;
            }
            recon.set(i, j, val as f32);
        }
    }

    for i in 0..9 {
        assert!(
            (data[i] - recon.data[i]).abs() < 1e-3,
            "recon error at {i}: {} vs {}",
            data[i],
            recon.data[i]
        );
    }
}

#[test]
fn svd_jacobi_reduced_sweeps_converges() {
    let data: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.3).sin() * 2.0).collect();
    let m = Mat::from_vec(4, 4, data);
    let svd = svd_jacobi(&m, 20, 1e-6).unwrap();

    let mut recon = Mat::new(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            let mut val = 0.0f64;
            for p in 0..svd.s.len() {
                val += svd.u.get(i, p) as f64 * svd.s[p] as f64 * svd.vt.get(p, j) as f64;
            }
            recon.set(i, j, val as f32);
        }
    }
    let err: f64 = (0..16)
        .map(|i| (m.data[i] as f64 - recon.data[i] as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let nrm = m.norm_fro();
    let rel_err = err / nrm;
    assert!(rel_err < 1e-3, "relative reconstruction error = {rel_err}");
}

#[test]
fn svd_jacobi_2x2_symmetric_matrix() {
    let data = vec![2.0, 1.0, 1.0, 2.0];
    let m = Mat::from_vec(2, 2, data);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    let mut vals = svd.s.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((vals[0] - 3.0).abs() < 1e-3, "s[0] = {}", vals[0]);
    assert!((vals[1] - 1.0).abs() < 1e-3, "s[1] = {}", vals[1]);
}

// ---- evd_symmetric tests ----

#[test]
fn evd_symmetric_requires_square() {
    let m = Mat::new(3, 4);
    assert!(evd_symmetric(&m, 100, 1e-10).is_err());
}

#[test]
fn evd_symmetric_identity() {
    let m = Mat::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let (evals, evecs) = evd_symmetric(&m, 100, 1e-10).unwrap();
    let mut vals = evals.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((vals[0] - 1.0).abs() < 1e-4);
    assert!((vals[1] - 1.0).abs() < 1e-4);
    assert!((vals[2] - 1.0).abs() < 1e-4);
    let mut eye = Mat::new(3, 3);
    Mat::matmul_into(&evecs, &transpose(&evecs), &mut eye);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (eye.get(i, j) - expected).abs() < 1e-4,
                "V^T V [{i},{j}] = {}",
                eye.get(i, j)
            );
        }
    }
}

#[test]
fn evd_symmetric_diagonal_matrix() {
    let m = Mat::from_vec(3, 3, vec![5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0]);
    let (evals, _evecs) = evd_symmetric(&m, 100, 1e-10).unwrap();
    let mut vals = evals.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((vals[0] - 5.0).abs() < 1e-4);
    assert!((vals[1] - 3.0).abs() < 1e-4);
    assert!((vals[2] - 1.0).abs() < 1e-4);
}

#[test]
fn evd_symmetric_2x2_known_eigenvalues() {
    let m = Mat::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]);
    let (evals, _evecs) = evd_symmetric(&m, 100, 1e-10).unwrap();
    let mut vals = evals.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let expected_0 = (7.0 + 17.0_f64.sqrt()) / 2.0;
    let expected_1 = (7.0 - 17.0_f64.sqrt()) / 2.0;
    assert!((vals[0] as f64 - expected_0).abs() < 1e-3, "eval[0] = {}", vals[0]);
    assert!((vals[1] as f64 - expected_1).abs() < 1e-3, "eval[1] = {}", vals[1]);
}

#[test]
fn evd_symmetric_eigenvector_orthogonality() {
    let m = Mat::from_vec(3, 3, vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
    let (_evals, evecs) = evd_symmetric(&m, 200, 1e-12).unwrap();
    let mut vt_v = Mat::new(3, 3);
    Mat::matmul_into(&transpose(&evecs), &evecs, &mut vt_v);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (vt_v.get(i, j) - expected).abs() < 1e-4,
                "V^T V [{i},{j}] = {}",
                vt_v.get(i, j)
            );
        }
    }
}

// ---- svd_randomized tests ----

#[test]
fn svd_randomized_empty_returns_error() {
    let m = Mat::new(0, 5);
    assert!(svd_randomized(&m, 2, 4, 2, 42).is_err());
}

#[test]
fn svd_randomized_rank_zero_returns_error() {
    let m = Mat::from_vec(5, 5, (0..25).map(|i| i as f32).collect());
    assert!(svd_randomized(&m, 0, 4, 2, 42).is_err());
}

#[test]
fn svd_randomized_small_rank1_matrix() {
    let m = 8usize;
    let n = 6usize;
    let u: Vec<f32> = (0..m).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let v: Vec<f32> = (0..n).map(|j| (j as f32 + 1.0) * 0.3).collect();
    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            data[i * n + j] = u[i] * v[j];
        }
    }
    let a = Mat::from_vec(m, n, data);
    let svd = svd_randomized(&a, 3, 4, 2, 42).unwrap();
    assert!(svd.s.len() >= 1);
    assert!(svd.s[0] > svd.s[1] * 10.0);
}

#[test]
fn svd_randomized_matches_jacobi_on_small_matrix() {
    let m = 20usize;
    let n = 15usize;
    let data: Vec<f32> = (0..m * n)
        .map(|i| (i as f32 * 0.1).sin() * 3.0 + 1.0)
        .collect();
    let a = Mat::from_vec(m, n, data);
    let s_rand = svd_randomized(&a, 5, 4, 2, 42).unwrap();
    let s_jac = svd_jacobi(&a, 100, 1e-10).unwrap();
    assert!(!s_rand.s.is_empty() && !s_jac.s.is_empty());
    let ratio = s_rand.s[0] / s_jac.s[0];
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "s_rand[0]={} vs s_jac[0]={}",
        s_rand.s[0],
        s_jac.s[0]
    );
    let norm_rand: f32 = s_rand.s.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_jac: f32 = s_jac.s.iter().take(s_rand.s.len()).map(|x| x * x).sum::<f32>().sqrt();
    let norm_ratio = norm_rand / norm_jac;
    assert!(
        (norm_ratio - 1.0).abs() < 0.1,
        "frobenius norm ratio: {}",
        norm_ratio
    );
}

// ---- rank_for_energy tests ----

#[test]
fn rank_for_energy_empty_s() {
    assert_eq!(rank_for_energy(&[], 0.99, 2, 10), 2);
}

#[test]
fn rank_for_energy_all_zero() {
    let s = vec![0.0, 0.0, 0.0];
    assert_eq!(rank_for_energy(&s, 0.99, 1, 10), 1);
}

#[test]
fn rank_for_energy_single_dominant() {
    let s = vec![100.0, 0.1, 0.01];
    assert_eq!(rank_for_energy(&s, 0.99, 1, 10), 1);
}

#[test]
fn rank_for_energy_needs_two() {
    let s = vec![10.0, 10.0, 0.1, 0.01];
    assert_eq!(rank_for_energy(&s, 0.99, 1, 10), 2);
}

#[test]
fn rank_for_energy_clamped_to_max() {
    let s = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    assert_eq!(rank_for_energy(&s, 0.99, 1, 3), 3);
}

#[test]
fn rank_for_energy_clamped_to_min() {
    let s = vec![100.0];
    assert_eq!(rank_for_energy(&s, 0.99, 4, 10), 4);
}

// ---- pack_lowrank and reconstruct tests ----

#[test]
fn pack_lowrank_preserves_rank1() {
    let data = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0];
    let m = Mat::from_vec(2, 3, data.clone());
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    let (a, b) = pack_lowrank(&svd);
    assert_eq!(a.rows, 2);
    assert_eq!(a.cols, svd.s.len());
    assert_eq!(b.rows, svd.s.len());
    assert_eq!(b.cols, 3);

    let mut recon = Mat::new(2, 3);
    Mat::matmul_into(&a, &b, &mut recon);
    for i in 0..6 {
        assert!(
            (data[i] - recon.data[i]).abs() < 1e-3,
            "recon[{i}] = {} vs {}",
            recon.data[i],
            data[i]
        );
    }
}

#[test]
fn reconstruct_matches_matmul() {
    let a = Mat::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Mat::from_vec(2, 3, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let out = reconstruct(&a, &b);
    let mut expected = Mat::new(3, 3);
    Mat::matmul_into(&a, &b, &mut expected);
    for i in 0..9 {
        assert!((out.data[i] - expected.data[i]).abs() < 1e-10);
    }
}

// ---- slice_cols and slice_rows tests ----

#[test]
fn slice_cols_extracts_correct_columns() {
    let m = Mat::from_vec(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let s = slice_cols(&m, 1, 2);
    assert_eq!(s.rows, 2);
    assert_eq!(s.cols, 2);
    assert_eq!(s.get(0, 0), 2.0);
    assert_eq!(s.get(0, 1), 3.0);
    assert_eq!(s.get(1, 0), 6.0);
    assert_eq!(s.get(1, 1), 7.0);
}

#[test]
fn slice_rows_extracts_correct_rows() {
    let m = Mat::from_vec(4, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let s = slice_rows(&m, 1, 2);
    assert_eq!(s.rows, 2);
    assert_eq!(s.cols, 3);
    assert_eq!(s.get(0, 0), 4.0);
    assert_eq!(s.get(0, 2), 6.0);
    assert_eq!(s.get(1, 0), 7.0);
    assert_eq!(s.get(1, 2), 9.0);
}

// ---- transpose tests ----

#[test]
fn transpose_2x3_to_3x2() {
    let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = transpose(&m);
    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 2);
    assert_eq!(t.get(0, 0), 1.0);
    assert_eq!(t.get(0, 1), 4.0);
    assert_eq!(t.get(2, 0), 3.0);
    assert_eq!(t.get(2, 1), 6.0);
}

// ---- orthonormalize_cols tests ----

#[test]
fn orthonormalize_cols_produces_orthonormal() {
    let m = Mat::from_vec(4, 3, vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let q = orthonormalize_cols(&m);
    for j in 0..3 {
        let mut nrm = 0.0f64;
        for i in 0..4 {
            let v = q.get(i, j) as f64;
            nrm += v * v;
        }
        assert!((nrm - 1.0).abs() < 1e-4, "column {j} norm = {nrm}");
    }
    for j1 in 0..3 {
        for j2 in (j1 + 1)..3 {
            let mut dot = 0.0f64;
            for i in 0..4 {
                dot += q.get(i, j1) as f64 * q.get(i, j2) as f64;
            }
            assert!(dot.abs() < 1e-4, "columns {j1},{j2} not orthogonal: dot={dot}");
        }
    }
}

// ---- SVD structure tests ----

#[test]
fn svd_jacobi_output_shapes() {
    let m = Mat::from_vec(5, 3, (0..15).map(|i| i as f32).collect());
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    let rank = 5.min(3);
    assert_eq!(svd.u.rows, 5);
    assert_eq!(svd.u.cols, rank);
    assert_eq!(svd.s.len(), rank);
    assert_eq!(svd.vt.rows, rank);
    assert_eq!(svd.vt.cols, 3);
}

#[test]
fn svd_jacobi_singular_values_non_negative() {
    let data: Vec<f32> = (0..20).map(|i| ((i as f32) - 10.0) * 0.5).collect();
    let m = Mat::from_vec(4, 5, data);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    for &s in &svd.s {
        assert!(s >= 0.0, "singular value should be non-negative: {s}");
    }
}

#[test]
fn svd_jacobi_singular_values_sorted_descending() {
    let data: Vec<f32> = (0..20).map(|i| ((i as f32) * 0.3).sin() * 5.0).collect();
    let m = Mat::from_vec(4, 5, data);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    for i in 0..svd.s.len() - 1 {
        assert!(
            svd.s[i] >= svd.s[i + 1] - 1e-5,
            "singular values not sorted: s[{}]={} < s[{}]={}",
            i,
            svd.s[i],
            i + 1,
            svd.s[i + 1]
        );
    }
}

#[test]
fn svd_jacobi_near_zero_gamma() {
    let mut data = vec![0.0f32; 9];
    data[0] = 100.0;
    data[4] = 100.0;
    data[8] = 100.0;
    data[1] = 1e-31;
    data[3] = 1e-31;
    let m = Mat::from_vec(3, 3, data);
    let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
    let mut vals = svd.s.clone();
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!((vals[0] - 100.0).abs() < 0.1);
}
