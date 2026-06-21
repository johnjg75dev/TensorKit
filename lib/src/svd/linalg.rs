//! Pure-Rust linear-algebra primitives used by the SVD compressor.
//!
//! * **One-sided Jacobi SVD** — accurate, simple, fast for matrices up to a
//!   few thousand rows/columns (the size of typical transformer attention /
//!   FFN projections).
//! * **Randomized SVD** (Halko et al., 2011) — used for matrices where Jacobi
//!   would be too slow. Trades some accuracy for an O(m n log k) cost.
//!
//! All matrices are stored in row-major `Vec<f32>` with `n_rows` and `n_cols`
//! passed alongside. Element `(i, j)` is at index `i * n_cols + j`.

use crate::error::{Error, Result};
use rayon::prelude::*;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// A simple 32-byte aligned vector for SIMD operations.
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

impl<T> AlignedVec<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<T>(),
            32,
        ).expect("Invalid layout");
        let ptr = unsafe { alloc(layout) as *mut T };
        Self {
            ptr: NonNull::new(ptr).expect("Allocation failed"),
            len: 0,
            cap: capacity,
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.cap);
        self.len = len;
    }

    pub fn push(&mut self, val: T) {
        assert!(self.len < self.cap);
        unsafe {
            self.ptr.as_ptr().add(self.len).write(val);
        }
        self.len += 1;
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(
            self.cap * std::mem::size_of::<T>(),
            32,
        ).expect("Invalid layout");
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, layout);
        }
    }
}

impl<T> std::ops::Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> std::ops::DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<'a, T> IntoIterator for &'a AlignedVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new = Self::with_capacity(self.len);
        for i in 0..self.len {
            new.push(self[i].clone());
        }
        new
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Row-major dense matrix in `f32`.
#[derive(Debug, Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: AlignedVec<f32>,
}

impl Mat {
    #[inline]
    pub fn new(rows: usize, cols: usize) -> Self {
        let mut data = AlignedVec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(0.0);
        }
        data.set_len(rows * cols);
        Self {
            rows,
            cols,
            data,
        }
    }
    #[inline]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        let mut aligned = AlignedVec::with_capacity(data.len());
        for x in data {
            aligned.push(x);
        }
        Self { rows, cols, data: aligned }
    }
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }

    /// Frobenius norm.
    pub fn norm_fro(&self) -> f64 {
        let mut s = 0.0f64;
        for &x in &self.data {
            s += (x as f64) * (x as f64);
        }
        s.sqrt()
    }

    /// `out = a * b`. `a` is m x k, `b` is k x n, `out` is m x n.
    /// Dispatches to cache-blocked SIMD (AVX2+FMA) when available,
    /// tiled scalar otherwise. Tiny matrices use a reordered naive path.
    pub fn matmul_into(a: &Mat, b: &Mat, out: &mut Mat) {
        assert_eq!(a.cols, b.rows, "matmul: inner dims must match");
        assert_eq!(out.rows, a.rows, "matmul: out rows must match a rows");
        assert_eq!(out.cols, b.cols, "matmul: out cols must match b cols");
        matmul_dispatch(a, b, out);
    }
}

// ---- matmul internals --------------------------------------------------

/// Skip tiling overhead for matrices this small.
const MATMUL_SMALL_MN: usize = 256;
const MATMUL_MIN_K:  usize = 32;

fn matmul_dispatch(a: &Mat, b: &Mat, out: &mut Mat) {
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    if m * n <= MATMUL_SMALL_MN || k < MATMUL_MIN_K {
        matmul_naive(a, b, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                matmul_tiled_avx2(a, b, out);
            }
            return;
        }
    }

    matmul_tiled_scalar(a, b, out);
}

/// Reordered naive triple-loop for tiny matrices.
/// Inner loops: i → p → j, so B is accessed contiguously.
fn matmul_naive(a: &Mat, b: &Mat, out: &mut Mat) {
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    out.data[..m * n].fill(0.0);
    for i in 0..m {
        let out_row = &mut out.data[i * n..];
        for p in 0..k {
            let a_val = a.data[i * k + p];
            let b_row = &b.data[p * n..];
            for j in 0..n {
                out_row[j] += a_val * b_row[j];
            }
        }
    }
}

/// Cache-blocked scalar matmul. Tiles the output 64×64 so the working set
/// of `b` stays resident in L1/L2. Uses an outer-product inner loop for
/// good row-major spatial locality on A, B, and C.
fn matmul_tiled_scalar(a: &Mat, b: &Mat, out: &mut Mat) {
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    const BM: usize = 64;
    const BN: usize = 64;
    const BK: usize = 64;

    out.data[..m * n].fill(0.0);

    for ib in (0..m).step_by(BM) {
        let ib_end = (ib + BM).min(m);
        for pb in (0..k).step_by(BK) {
            let pb_end = (pb + BK).min(k);
            for jb in (0..n).step_by(BN) {
                let jb_end = (jb + BN).min(n);

                for i in ib..ib_end {
                    let a_row = i * k;
                    let c_row = i * n;
                    for pp in pb..pb_end {
                        let a_val = a.data[a_row + pp];
                        let b_row = pp * n;

                        let mut j = jb;
                        // Unrolled ×4 for ILP.
                        while j + 4 <= jb_end {
                            let b0 = b.data[b_row + j];
                            let b1 = b.data[b_row + j + 1];
                            let b2 = b.data[b_row + j + 2];
                            let b3 = b.data[b_row + j + 3];
                            out.data[c_row + j]     += a_val * b0;
                            out.data[c_row + j + 1] += a_val * b1;
                            out.data[c_row + j + 2] += a_val * b2;
                            out.data[c_row + j + 3] += a_val * b3;
                            j += 4;
                        }
                        while j < jb_end {
                            out.data[c_row + j] += a_val * b.data[b_row + j];
                            j += 1;
                        }
                    }
                }
            }
        }
    }
}

// ---- AVX2+FMA matmul (x86_64 only) -------------------------------------

#[cfg(target_arch = "x86_64")]
const BM_AVX2: usize = 128;
#[cfg(target_arch = "x86_64")]
const BN_AVX2: usize = 128;
#[cfg(target_arch = "x86_64")]
const BK_AVX2: usize = 64;

/// Cache-blocked outer-product matmul using `f32x8` AVX2 FMA intrinsics.
/// Processes 8 columns of B / C per `_mm256_fmadd_ps` instruction.
///
/// # Safety
/// Caller must verify AVX2 + FMA are available at runtime
/// (`is_x86_feature_detected!` at the dispatch site).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_tiled_avx2(a: &Mat, b: &Mat, out: &mut Mat) { unsafe {
    use std::arch::x86_64::*;

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    out.data.fill(0.0);

    for ib in (0..m).step_by(BM_AVX2) {
        let ib_end = (ib + BM_AVX2).min(m);
        for pb in (0..k).step_by(BK_AVX2) {
            let pb_end = (pb + BK_AVX2).min(k);
            for jb in (0..n).step_by(BN_AVX2) {
                let jb_end = (jb + BN_AVX2).min(n);

                for i in ib..ib_end {
                    let a_off = i * k;
                    let c_off = i * n;
                    for pp in pb..pb_end {
                        let a_val = _mm256_set1_ps(*a.data.get_unchecked(a_off + pp));

                        let mut j = jb;
                        // 8-wide FMA: C[i][j..j+7] += A[i][pp] * B[pp][j..j+7]
                        while j + 8 <= jb_end {
                            let b_ptr = b.data.as_ptr().add(pp * n + j);
                            let c_ptr = out.data.as_ptr().add(c_off + j);
                            let b_vec = _mm256_loadu_ps(b_ptr);
                            let c_vec = _mm256_loadu_ps(c_ptr);
                            _mm256_storeu_ps(
                                out.data.as_mut_ptr().add(c_off + j),
                                _mm256_fmadd_ps(a_val, b_vec, c_vec),
                            );
                            j += 8;
                        }
                        // Tail: 0–7 columns.
                        while j < jb_end {
                            *out.data.get_unchecked_mut(c_off + j) +=
                                a.data.get_unchecked(a_off + pp) *
                                    b.data.get_unchecked(pp * n + j);
                            j += 1;
                        }
                    }
                }
            }
        }
    }
}}

/// One-sided Jacobi SVD result. `s.len() == u.cols == vt.rows`.
///
/// `A == U * diag(s) * Vt` (within numerical tolerance). The decomposition is
/// economy-size: `u.rows == a.rows`, `u.cols == vt.rows == s.len() == rank`
/// where `rank = min(a.rows, a.cols)`.
#[derive(Debug, Clone)]
pub struct Svd {
    pub u: Mat,      // m x k
    pub s: Vec<f32>, // k
    pub vt: Mat,     // k x n
}

/// Compute the eigenvalue decomposition of a symmetric matrix `a` (n x n).
/// Returns `(eigenvalues, eigenvectors)`.
pub fn evd_symmetric(a: &Mat, max_sweeps: usize, tol: f64) -> Result<(Vec<f32>, Mat)> {
    let n = a.rows;
    if n != a.cols {
        return Err(Error::Svd("EVD requires a square matrix".into()));
    }

    // Use f64 internally for precision.
    let mut work_f64 = vec![0.0f64; n * n];
    for i in 0..n * n {
        work_f64[i] = a.data[i] as f64;
    }

    let mut v_f64 = vec![0.0f64; n * n];
    for i in 0..n {
        v_f64[i * n + i] = 1.0;
    }

    let mut sweep = 0;
    let target_off = (a.norm_fro() * a.norm_fro()) * tol * tol;

    while sweep < max_sweeps {
        sweep += 1;
        let mut off_sq = 0.0f64;

        for i in 0..n {
            for j in (i + 1)..n {
                let a_ii = work_f64[i * n + i];
                let a_jj = work_f64[j * n + j];
                let a_ij = work_f64[i * n + j];

                off_sq += 2.0 * a_ij * a_ij;

                if a_ij.abs() < 1e-30 {
                    continue;
                }

                let (c_rot, s_rot) = jacobi_2x2(a_ii, a_jj, a_ij);

                // Update work matrix: W = R^T * W * R
                for k in 0..n {
                    let wik = work_f64[i * n + k];
                    let wjk = work_f64[j * n + k];
                    work_f64[i * n + k] = c_rot * wik + s_rot * wjk;
                    work_f64[j * n + k] = -s_rot * wik + c_rot * wjk;
                }
                for k in 0..n {
                    let wki = work_f64[k * n + i];
                    let wkj = work_f64[k * n + j];
                    work_f64[k * n + i] = c_rot * wki + s_rot * wkj;
                    work_f64[k * n + j] = -s_rot * wki + c_rot * wkj;
                }
                // Update eigenvectors
                for k in 0..n {
                    let vki = v_f64[k * n + i];
                    let vkj = v_f64[k * n + j];
                    v_f64[k * n + i] = c_rot * vki + s_rot * vkj;
                    v_f64[k * n + j] = -s_rot * vki + c_rot * vkj;
                }
            }
        }

        if off_sq <= target_off {
            break;
        }
    }

    let mut s = Vec::with_capacity(n);
    for i in 0..n {
        s.push(work_f64[i * n + i] as f32);
    }

    // Sort eigenvalues descending
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
    
    let s_sorted: Vec<f32> = order.iter().map(|&i| s[i]).collect();
    let mut v_sorted = Mat::new(n, n);
    for (new_c, &old_c) in order.iter().enumerate() {
        for r in 0..n {
            v_sorted.set(r, new_c, v_f64[r * n + old_c] as f32);
        }
    }

    Ok((s_sorted, v_sorted))
}

/// Run one-sided Jacobi SVD on a row-major matrix `a` (m x n).
///
/// `max_sweeps` caps the number of full sweeps (one sweep = n*(n-1)/2 pair
/// rotations). `tol` is the convergence threshold on the off-diagonal
/// Frobenius norm of `A^T A`, relative to the diagonal norm.
pub fn svd_jacobi(a: &Mat, max_sweeps: usize, tol: f64) -> Result<Svd> {
    let m = a.rows;
    let n = a.cols;
    if m == 0 || n == 0 {
        return Err(Error::Svd("empty matrix".into()));
    }

    // Use f64 internally for precision and to avoid repeated casting.
    let mut work_f64 = vec![0.0f64; m * n];
    for i in 0..m * n {
        work_f64[i] = a.data[i] as f64;
    }

    let mut v_f64 = vec![0.0f64; n * n];
    for i in 0..n {
        v_f64[i * n + i] = 1.0;
    }

    let mut sweep = 0;
    let target_off = (a.norm_fro() * a.norm_fro()) * tol * tol;

    // Build independent pair batches using greedy graph coloring.
    // Pairs sharing no column can rotate in parallel.
    let all_pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect();
    let mut batches: Vec<Vec<(usize, usize)>> = Vec::new();

    for &(i, j) in &all_pairs {
        let mut placed = None;
        for b in 0..batches.len() {
            let mut ok = true;
            for &(pi, pj) in &batches[b] {
                if pi == i || pi == j || pj == i || pj == j {
                    ok = false;
                    break;
                }
            }
            if ok {
                placed = Some(b);
                break;
            }
        }
        let b = placed.unwrap_or_else(|| {
            batches.push(Vec::new());
            batches.len() - 1
        });
        batches[b].push((i, j));
    }

    while sweep < max_sweeps {
        sweep += 1;
        let mut off_sq = 0.0f64;

        for batch in &batches {
            // Compute rotations for all independent pairs in parallel.
            let rotations: Vec<(usize, usize, f64, f64, f64)> = batch
                .par_iter()
                .map(|&(i, j)| {
                    let mut alpha = 0.0f64;
                    let mut beta  = 0.0f64;
                    let mut gamma = 0.0f64;

                    let mut r = 0;
                    while r + 4 <= m {
                        let x0 = work_f64[r * n + i];
                        let y0 = work_f64[r * n + j];
                        let x1 = work_f64[(r + 1) * n + i];
                        let y1 = work_f64[(r + 1) * n + j];
                        let x2 = work_f64[(r + 2) * n + i];
                        let y2 = work_f64[(r + 2) * n + j];
                        let x3 = work_f64[(r + 3) * n + i];
                        let y3 = work_f64[(r + 3) * n + j];

                        alpha += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
                        beta  += y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
                        gamma += x0 * y0 + x1 * y1 + x2 * y2 + x3 * y3;

                        r += 4;
                    }
                    for rr in r..m {
                        let x = work_f64[rr * n + i];
                        let y = work_f64[rr * n + j];
                        alpha += x * x;
                        beta  += y * y;
                        gamma += x * y;
                    }

                    (i, j, alpha, beta, gamma)
                })
                .collect();

            // Accumulate off-norm and apply rotations sequentially (columns shared across batches).
            for &(i, j, alpha, beta, gamma) in &rotations {
                off_sq += gamma * gamma;

                if gamma.abs() < 1e-30 {
                    continue;
                }

                let (c_rot, s_rot) = jacobi_2x2(alpha, beta, gamma);
                if s_rot.abs() < 1e-30 {
                    continue;
                }

                for r in 0..m {
                    let xi = work_f64[r * n + i];
                    let xj = work_f64[r * n + j];
                    work_f64[r * n + i] = c_rot * xi + s_rot * xj;
                    work_f64[r * n + j] = -s_rot * xi + c_rot * xj;
                }
                for r in 0..n {
                    let vi = v_f64[r * n + i];
                    let vj = v_f64[r * n + j];
                    v_f64[r * n + i] = c_rot * vi + s_rot * vj;
                    v_f64[r * n + j] = -s_rot * vi + c_rot * vj;
                }
            }
        }

        if off_sq <= target_off {
            break;
        }
    }

    let mut s = Vec::with_capacity(n);
    let mut u = Mat::new(m, n);
    for c in 0..n {
        let mut nrm = 0.0f64;
        for r in 0..m {
            let x = work_f64[r * n + c];
            nrm += x * x;
        }
        let nrm = nrm.sqrt();
        s.push(nrm as f32);
        let inv = if nrm > 0.0 { 1.0 / nrm } else { 0.0 };
        for r in 0..m {
            u.set(r, c, (work_f64[r * n + c] * inv) as f32);
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
    let s_sorted: Vec<f32> = order.iter().map(|&i| s[i]).collect();
    let mut u_sorted = Mat::new(m, n);
    for (new_c, &old_c) in order.iter().enumerate() {
        for r in 0..m {
            u_sorted.set(r, new_c, u.get(r, old_c));
        }
    }
    let mut v_perm = Mat::new(n, n);
    for (new_c, &old_c) in order.iter().enumerate() {
        for r in 0..n {
            v_perm.set(r, new_c, v_f64[r * n + old_c] as f32);
        }
    }
    let mut vt = Mat::new(n, n);
    for i in 0..n {
        for j in 0..n {
            vt.set(j, i, v_perm.get(i, j));
        }
    }

    Ok(Svd {
        u: u_sorted,
        s: s_sorted,
        vt,
    })
}

/// Closed-form Jacobi rotation for the symmetric 2x2 `[[a, g], [g, b]]`.
/// Returns `(cos, sin)` such that the rotation `R = [[c, s], [-s, c]]`
/// satisfies `R^T * M * R = diag(a', b')` with `a' >= b'`.
#[inline]
pub fn jacobi_2x2(a: f64, b: f64, g: f64) -> (f64, f64) {
    if g.abs() < 1e-30 {
        return (1.0, 0.0);
    }
    // tan(2θ) = 2g / (a - b); pick θ so the larger eigenvalue lands at (0, 0).
    let tau = (a - b) / (2.0 * g);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let c = 1.0 / (1.0 + t * t).sqrt();
    let s = t * c;
    (c, s)
}

/// Compute the rank `k` truncated SVD using the randomized algorithm
/// (Halko, Martinsson, Tropp, 2011, "Finding structure with randomness").
///
/// 1. Draw an n x (k + p) Gaussian test matrix Omega.
/// 2. Form Y = (A A^T)^q A Omega via q power iterations.
/// 3. Orthonormalize Y's columns to obtain an approximate column-space basis Q.
/// 4. Form B = Q^T A and compute its small SVD; lift back to U of A.
pub fn svd_randomized(
    a: &Mat,
    target_rank: usize,
    oversample: usize,
    power_iters: usize,
    seed: u64,
) -> Result<Svd> {
    let m = a.rows;
    let n = a.cols;
    if m == 0 || n == 0 {
        return Err(Error::Svd("empty matrix".into()));
    }
    let k = target_rank.min(m).min(n);
    if k == 0 {
        return Err(Error::Svd("target rank must be > 0".into()));
    }
    let l = (k + oversample).min(n).min(m);

    // 1) Draw Omega: n x l Gaussian (deterministic via xorshift).
    let mut rng = XorShift::new(seed);
    let mut omega = Mat::new(n, l);
    for j in 0..l {
        for i in 0..n {
            omega.data[i * l + j] = rng.gauss();
        }
    }

    // 2) Y = A * Omega  (m x l)
    let mut y = Mat::new(m, l);
    Mat::matmul_into(a, &omega, &mut y);

    // Power iterations: Y <- (A A^T)^q * Y
    let at = transpose(a);
    for _ in 0..power_iters {
        // Z = A^T * Y  (n x l)
        let mut z = Mat::new(n, l);
        Mat::matmul_into(&at, &y, &mut z);
        // Y = A * Z
        y = Mat::new(m, l);
        Mat::matmul_into(a, &z, &mut y);
    }

    // 3) Orthonormalize Y's columns (modified Gram-Schmidt).
    let q = orthonormalize_cols(&y);

    // 4) B = Q^T * A  (l x n), then SVD of B.
    let qt = transpose(&q);
    let mut b = Mat::new(l, n);
    Mat::matmul_into(&qt, a, &mut b);

    // To avoid O(n^2 l) Jacobi SVD on B, we use the Gramian G = B*B^T (l x l).
    // G = U_tilde * Sigma^2 * U_tilde^T.
    let mut g = Mat::new(l, l);
    Mat::matmul_into(&b, &transpose(&b), &mut g);

    let (eigvals, u_tilde) = evd_symmetric(&g, 100, 1e-12)?;

    // Singular values are sqrt of eigenvalues.
    let mut s_full = Vec::with_capacity(l);
    for &ev in &eigvals {
        s_full.push(ev.max(0.0).sqrt());
    }

    // Truncate to k.
    let ks = k.min(s_full.len());
    let u_tilde_k = slice_cols(&u_tilde, 0, ks);
    let s_k: Vec<f32> = s_full.iter().take(ks).copied().collect();

    // Compute Vt = Sigma^-1 * U_tilde^T * B.
    let mut vt = Mat::new(ks, n);
    for i in 0..ks {
        let inv_s = if s_k[i] > 0.0 { 1.0 / s_k[i] } else { 0.0 };
        for j in 0..n {
            let mut dot = 0.0f32;
            for r in 0..l {
                dot += u_tilde.get(r, i) * b.get(r, j);
            }
            vt.set(i, j, dot * inv_s);
        }
    }

    // U = Q * U_tilde (m x ks)
    let mut u = Mat::new(m, ks);
    Mat::matmul_into(&q, &u_tilde_k, &mut u);
    Ok(Svd { u, s: s_k, vt })
}

pub fn transpose(a: &Mat) -> Mat {
    let mut t = Mat::new(a.cols, a.rows);
    for i in 0..a.rows {
        for j in 0..a.cols {
            t.data[j * a.rows + i] = a.data[i * a.cols + j];
        }
    }
    t
}

pub fn slice_cols(a: &Mat, start: usize, count: usize) -> Mat {
    let mut out = Mat::new(a.rows, count);
    for r in 0..a.rows {
        for c in 0..count {
            out.data[r * count + c] = a.data[r * a.cols + start + c];
        }
    }
    out
}

pub fn slice_rows(a: &Mat, start: usize, count: usize) -> Mat {
    let mut out = Mat::new(count, a.cols);
    for r in 0..count {
        for c in 0..a.cols {
            out.data[r * a.cols + c] = a.data[(start + r) * a.cols + c];
        }
    }
    out
}

pub fn orthonormalize_cols(a: &Mat) -> Mat {
    let m = a.rows;
    let n = a.cols;
    // Use classical Gram-Schmidt (CGS) with two passes of re-orthogonalization.
    // The first pass uses the original input for the inner product; the second
    // pass uses the already-orthogonalized q for the inner product, which
    // stabilizes the result for near-degenerate inputs (a common situation
    // for the randomized SVD's `Y` matrix, whose rank can be much smaller
    // than its column count).
    let mut q = a.clone();
    for _pass in 0..2 {
        for j in 0..n {
            for k in 0..j {
                let mut dot = 0.0f64;
                let src: &[f32] = if _pass == 0 { &a.data } else { &q.data };
                for r in 0..m {
                    dot += (q.data[r * n + k] as f64) * (src[r * n + j] as f64);
                }
                for r in 0..m {
                    q.data[r * n + j] -= (dot as f32) * q.data[r * n + k];
                }
            }
            let mut nrm = 0.0f64;
            for r in 0..m {
                let x = q.data[r * n + j] as f64;
                nrm += x * x;
            }
            let nrm = nrm.sqrt();
            if nrm > 0.0 {
                let inv = 1.0 / nrm as f32;
                for r in 0..m {
                    q.data[r * n + j] *= inv;
                }
            } else {
                // Column is in the span of previous ones; zero it.
                for r in 0..m {
                    q.data[r * n + j] = 0.0;
                }
            }
        }
    }
    q
}

/// Tiny deterministic xorshift64* PRNG.
struct XorShift(u64);
impl XorShift {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Box-Muller standard normal.
    fn gauss(&mut self) -> f32 {
        loop {
            let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64);
            let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return (r * theta.cos()) as f32;
            }
        }
    }
}

/// Pick the smallest rank `k` such that `sum_{i<k} s_i^2 >= energy * total`.
/// Returns at least 1 (or `min_k` if provided), and at most `max_k`.
pub fn rank_for_energy(s: &[f32], energy: f64, min_k: usize, max_k: usize) -> usize {
    if s.is_empty() {
        return min_k.max(1);
    }
    let total: f64 = s.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    if total <= 0.0 {
        return min_k.max(1);
    }
    let mut acc = 0.0f64;
    for (i, &v) in s.iter().enumerate() {
        acc += (v as f64) * (v as f64);
        if acc / total >= energy {
            return (i + 1).clamp(min_k.max(1), max_k.max(1));
        }
    }
    s.len().clamp(min_k.max(1), max_k.max(1))
}

/// Compose two factors back into a single matrix. `a` is m x k, `b` is k x n.
pub fn reconstruct(a: &Mat, b: &Mat) -> Mat {
    let mut out = Mat::new(a.rows, b.cols);
    Mat::matmul_into(a, b, &mut out);
    out
}

/// Pack an SVD into a low-rank pair `(a, b)` such that `a * b ~ U * diag(s) * V^T`.
/// `a` is m x k, `b` is k x n. Uses the symmetric "square-root" scaling so that
/// `||a||_2 ~ ||b||_2 ~ sqrt(max singular value)`.
pub fn pack_lowrank(svd: &Svd) -> (Mat, Mat) {
    let m = svd.u.rows;
    let n = svd.vt.cols;
    let k = svd.s.len();
    let mut a = Mat::new(m, k);
    let mut b = Mat::new(k, n);
    for j in 0..k {
        let s = svd.s[j].max(0.0).sqrt();
        for i in 0..m {
            a.data[i * k + j] = svd.u.data[i * k + j] * s;
        }
        for jj in 0..n {
            b.data[j * n + jj] = svd.vt.data[j * n + jj] * s;
        }
    }
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Ensure it's a deep copy
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
        // sqrt(1+4+9+16) = sqrt(30)
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
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
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
        // [1*4+0*2+2*0, 1*1+0*0+2*5] = [4, 11]
        // [0*4+3*2+1*0, 0*1+3*0+1*5] = [6, 5]
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
        // Matrix [[5, 3], [3, 3]] -> eigenvalues approx 7.6056 and 0.3944
        let (c, s) = jacobi_2x2(5.0, 3.0, 3.0);
        // Verify the rotation diagonalizes the matrix
        let a_f = 5.0f64;
        let b_f = 3.0f64;
        let g_f = 3.0f64;
        let a_prime = c * c * a_f + 2.0 * s * c * g_f + s * s * b_f;
        let b_prime = s * s * a_f - 2.0 * s * c * g_f + c * c * b_f;
        let g_prime = s * c * (b_f - a_f) + (c * c - s * s) * g_f;
        assert!(g_prime.abs() < 1e-10, "off-diagonal should be ~0, got {g_prime}");
        assert!(a_prime > b_prime, "larger eigenvalue should be at position 0");
        // Sum should be preserved (trace)
        assert!((a_prime + b_prime - (a_f + b_f)).abs() < 1e-10);
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
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 0.0,
        ];
        let m = Mat::from_vec(3, 3, data.clone());
        let svd = svd_jacobi(&m, 100, 1e-10).unwrap();

        // Reconstruct: A ≈ U * diag(s) * V^T
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
        // Even with reduced sweeps=20 and tol=1e-6, a well-conditioned matrix
        // should converge.
        let data: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.3).sin() * 2.0).collect();
        let m = Mat::from_vec(4, 4, data);
        let svd = svd_jacobi(&m, 20, 1e-6).unwrap();

        // Reconstruct and check error
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
        // eigenvalues of [[2,1],[1,2]] are 3 and 1
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
        // Check eigenvectors are orthonormal
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
        // [[4, 2], [2, 3]] -> eigenvalues (7 ± √17)/2 ≈ 5.562 and 1.438
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
        let m = Mat::from_vec(3, 3, vec![
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ]);
        let (_evals, evecs) = evd_symmetric(&m, 200, 1e-12).unwrap();
        // V^T * V should be ~ identity
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
        // Rank-1 matrix: outer product
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
        // Top singular value should dominate
        assert!(svd.s[0] > svd.s[1] * 10.0);
    }

    #[test]
    fn svd_randomized_matches_jacobi_on_small_matrix() {
        // Use a well-conditioned matrix where all singular values are
        // significantly above zero. The Gramian approach (evd on B*B^T)
        // squares eigenvalues, so smaller singular values are less accurate.
        // Test the top singular value which is the most reliable.
        let m = 20usize;
        let n = 15usize;
        let data: Vec<f32> = (0..m * n)
            .map(|i| (i as f32 * 0.1).sin() * 3.0 + 1.0)
            .collect();
        let a = Mat::from_vec(m, n, data);
        let s_rand = svd_randomized(&a, 5, 4, 2, 42).unwrap();
        let s_jac = svd_jacobi(&a, 100, 1e-10).unwrap();
        // Top singular value should match within 5%. The Gramian approach
        // is most accurate for the dominant eigenvalue.
        assert!(!s_rand.s.is_empty() && !s_jac.s.is_empty());
        let ratio = s_rand.s[0] / s_jac.s[0];
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "s_rand[0]={} vs s_jac[0]={}",
            s_rand.s[0],
            s_jac.s[0]
        );
        // Frobenius norms of the singular values should be close
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
        // 100^2 = 10000, total ≈ 10000.01, 99% = 9900.01 -> k=1
        assert_eq!(rank_for_energy(&s, 0.99, 1, 10), 1);
    }

    #[test]
    fn rank_for_energy_needs_two() {
        let s = vec![10.0, 10.0, 0.1, 0.01];
        // 100+100+0.01+0.0001 = 200.0101, 99% = 198.01 -> need both 10s
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
        let data = vec![
            1.0, 2.0, 3.0,
            2.0, 4.0, 6.0,
        ];
        let m = Mat::from_vec(2, 3, data.clone());
        let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
        let (a, b) = pack_lowrank(&svd);
        assert_eq!(a.rows, 2);
        assert_eq!(a.cols, svd.s.len());
        assert_eq!(b.rows, svd.s.len());
        assert_eq!(b.cols, 3);

        // Reconstruct
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
        let m = Mat::from_vec(4, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]);
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
        let m = Mat::from_vec(4, 3, vec![
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let q = orthonormalize_cols(&m);
        // Check each column has unit norm
        for j in 0..3 {
            let mut nrm = 0.0f64;
            for i in 0..4 {
                let v = q.get(i, j) as f64;
                nrm += v * v;
            }
            assert!(
                (nrm - 1.0).abs() < 1e-4,
                "column {j} norm = {nrm}"
            );
        }
        // Check orthogonality
        for j1 in 0..3 {
            for j2 in (j1 + 1)..3 {
                let mut dot = 0.0f64;
                for i in 0..4 {
                    dot += q.get(i, j1) as f64 * q.get(i, j2) as f64;
                }
                assert!(
                    dot.abs() < 1e-4,
                    "columns {j1},{j2} not orthogonal: dot={dot}"
                );
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

    // ---- Gamma epsilon check (gamma.abs() < 1e-30) ----

    #[test]
    fn svd_jacobi_near_zero_gamma() {
        // Construct a matrix where gamma for one pair will be extremely small
        // (diagonal dominant with tiny off-diagonal).
        let mut data = vec![0.0f32; 9];
        data[0] = 100.0;
        data[4] = 100.0;
        data[8] = 100.0;
        data[1] = 1e-31; // tiny off-diagonal
        data[3] = 1e-31;
        let m = Mat::from_vec(3, 3, data);
        let svd = svd_jacobi(&m, 100, 1e-10).unwrap();
        // Should converge quickly for nearly-diagonal matrix
        let mut vals = svd.s.clone();
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((vals[0] - 100.0).abs() < 0.1);
    }
}
