//! LAPACK-quality SVD backend using the `faer` crate.
//!
//! Enabled with the `faer-svd` feature. Provides `svd_faer()` which uses
//! faer's divide-and-conquer SVD for O(m n²) cost with much better constants
//! than the pure-Rust Jacobi implementation.

use crate::error::{Error, Result};
use crate::svd::linalg::{Mat, Svd};

/// Compute full SVD of an m×n row-major matrix using faer.
///
/// Returns (U, S, Vt) where A = U * diag(S) * Vt.
/// U is m×n (thin), S has n entries (sorted descending), Vt is n×n.
pub fn svd_faer(a: &Mat) -> Result<Svd> {
    let m = a.rows;
    let n = a.cols;
    if m == 0 || n == 0 {
        return Err(Error::Svd("empty matrix".into()));
    }

    // faer uses column-major; transpose from our row-major layout.
    let mut col_major = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            col_major[j * m + i] = a.data[i * n + j] as f64;
        }
    }

    let mat = faer::Mat::from_fn(m, n, |i, j| col_major[j * m + i]);

    let svd_result = mat.svd().map_err(|e| Error::Svd(format!("faer SVD failed: {e:?}")))?;
    let s_vec = svd_result.S();
    let u_mat = svd_result.U();
    let vt_mat = svd_result.V();

    let k = m.min(n);

    // Collect singular values (already sorted descending by faer).
    let s: Vec<f32> = (0..k).map(|i| s_vec[i] as f32).collect();

    // Extract thin U (m × k).
    let mut u = Mat::new(m, k);
    for j in 0..k {
        for i in 0..m {
            u.set(i, j, u_mat[(i, j)] as f32);
        }
    }

    // Extract Vt (k × n).
    let mut vt = Mat::new(k, n);
    for i in 0..k {
        for j in 0..n {
            vt.set(i, j, vt_mat[(i, j)] as f32);
        }
    }

    Ok(Svd { u, s, vt })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faer_svd_identity() {
        let m = Mat::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let svd = svd_faer(&m).unwrap();
        assert_eq!(svd.s.len(), 3);
        for i in 0..3 {
            assert!((svd.s[i] - 1.0).abs() < 1e-5, "s[{}] = {}", i, svd.s[i]);
        }
    }

    #[test]
    fn faer_svd_row_vector() {
        let m = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let svd = svd_faer(&m).unwrap();
        // Thin SVD of 1×3: k=min(1,3)=1, s has 1 entry
        assert_eq!(svd.s.len(), 1);
        let expected = (14.0f32).sqrt();
        assert!((svd.s[0] - expected).abs() < 1e-3, "s[0] = {}", svd.s[0]);
    }

    #[test]
    fn faer_svd_2x2_known() {
        let m = Mat::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]);
        let svd = svd_faer(&m).unwrap();
        // Singular values of [[4,2],[2,3]] are (7+sqrt(17))/2 ≈ 5.562 and (7-sqrt(17))/2 ≈ 1.438
        assert!((svd.s[0] - 5.562).abs() < 0.01, "s[0] = {}", svd.s[0]);
        assert!((svd.s[1] - 1.438).abs() < 0.01, "s[1] = {}", svd.s[1]);
    }

    #[test]
    fn faer_svd_reconstruction() {
        let m = Mat::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let svd = svd_faer(&m).unwrap();
        // Reconstruct: U * diag(S) * Vt should equal original matrix
        let k = svd.s.len();
        for i in 0..m.rows {
            for j in 0..m.cols {
                let mut val = 0.0f32;
                for l in 0..k {
                    val += svd.u.get(i, l) * svd.s[l] * svd.vt.get(l, j);
                }
                let orig = m.get(i, j);
                assert!(
                    (val - orig).abs() < 1e-4,
                    "reconstruction[{i},{j}] = {val}, original = {orig}"
                );
            }
        }
    }
}
