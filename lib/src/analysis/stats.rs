//! Single-pass statistics for f32 weight arrays.
//!
//! Hot loop is `Accum::push`. Marked `#[inline(always)]` and branchless where
//! possible. The Welford mean/variance update has a hard data dependency
//! (each step depends on the running mean), so it stays scalar; the
//! surrounding L2/abs/bucket updates are the easy SIMD wins (see
//! `push_block`).
//!
//! Percentiles use a reservoir sample (\u{2264} 65 536 floats) and are computed by
//! sorting only that sample.
//!
//! Higher central moments (M3, M4) use P\u{00e9}bay's one-pass recurrence so the
//! streaming `Accum` can report skewness and kurtosis without a second pass.
//! When the tensor was sampled, finalize falls back to computing those
//! moments from the reservoir directly (small set, exact).

use crate::analysis::score::BlockAnalysis;
use crate::model::Tensor;

/// Per-channel (per-row) statistics for 2-D weight tensors.
///
/// `channel_means.len() == channel_stds.len() == n_channels`, where
/// `n_channels` is the first axis length of the tensor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PerChannelStats {
    pub n_channels: usize,
    pub channel_means: Vec<f32>,
    pub channel_stds: Vec<f32>,
}

/// All statistics we collect for one tensor (or one block).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorStats {
    pub n: u64,
    pub n_sampled: bool,
    pub mean: f64,
    pub variance: f64,
    pub std: f64,
    pub abs_mean: f64,
    pub abs_max: f64,
    pub l2: f64,
    pub l1: f64,
    pub min: f32,
    pub max: f32,
    pub sparsity_abs: f64,
    pub outlier_ratio: f64,
    pub unique_bucket_count: u32,
    pub entropy_bits: f64,
    pub p01: f32,
    pub p50: f32,
    pub p99: f32,
    pub p999: f32,
    pub skewness: f64,
    pub kurtosis: f64,
    pub per_channel: Option<PerChannelStats>,
}

pub const RESERVOIR_SIZE: usize = 65_536;
const SPARSITY_EPS_ABS: f32 = 1e-3;
const SPARSITY_EPS_REL: f32 = 1e-3;
const OUTLIER_K: f64 = 4.0;
const N_BUCKETS_LOG2: u32 = 12; // 4096 log-buckets

/// Top-level analysis result.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Analysis {
    pub blocks: Vec<BlockAnalysis>,
    pub recommendation: Vec<i32>,
    pub recommendation_count: usize,
    pub estimated_bytes_after_prune: u64,
    pub sample_per_tensor: usize,
    pub total_tensors: usize,
    pub total_bytes: u64,

    // Model metadata (carried through from the source file)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_size: Option<u64>,
}

#[inline]
pub fn empty_stats() -> TensorStats {
    TensorStats {
        n: 0,
        n_sampled: false,
        mean: 0.0,
        variance: 0.0,
        std: 0.0,
        abs_mean: 0.0,
        abs_max: 0.0,
        l2: 0.0,
        l1: 0.0,
        min: 0.0,
        max: 0.0,
        sparsity_abs: 0.0,
        outlier_ratio: 0.0,
        unique_bucket_count: 0,
        entropy_bits: 0.0,
        p01: 0.0,
        p50: 0.0,
        p99: 0.0,
        p999: 0.0,
        skewness: 0.0,
        kurtosis: 0.0,
        per_channel: None,
    }
}

/// Single-pass Welford statistics. Call `push` repeatedly, then `finalize`.
///
/// Higher central moments (M3, M4) are tracked via P\u{00e9}bay's one-pass
/// recurrence. For 2-D tensors constructed with `new_2d`, per-row sums and
/// sum-of-squares are accumulated so the finalized `TensorStats` can carry
/// `per_channel` mean / std.
#[derive(Default)]
pub struct Accum {
    n: u64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    abs_sum: f64,
    abs_max: f64,
    l2: f64,
    l1: f64,
    min: f32,
    max: f32,
    bucket_counts: Vec<u32>,
    reservoir: Vec<f32>,
    rng_state: u64,

    // Per-channel (per-row) state for 2-D tensors.
    is_2d: bool,
    n_rows: usize,
    n_cols: usize,
    current_row: usize,
    current_col: usize,
    row_sums: Vec<f64>,
    row_sumsq: Vec<f64>,
}

impl Accum {
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            abs_sum: 0.0,
            abs_max: 0.0,
            l2: 0.0,
            l1: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            bucket_counts: vec![0u32; 1usize << N_BUCKETS_LOG2],
            reservoir: Vec::with_capacity(RESERVOIR_SIZE),
            rng_state: 0x9E37_79B9_7F4A_7C15,
            is_2d: false,
            n_rows: 0,
            n_cols: 0,
            current_row: 0,
            current_col: 0,
            row_sums: Vec::new(),
            row_sumsq: Vec::new(),
        }
    }

    /// Construct an accumulator that also tracks per-row sums/sum-of-squares.
    /// Caller is expected to push `rows * cols` values in row-major order.
    pub fn new_2d(rows: usize, cols: usize) -> Self {
        let mut s = Self::new();
        s.is_2d = true;
        s.n_rows = rows;
        s.n_cols = cols;
        s.row_sums = vec![0.0; rows];
        s.row_sumsq = vec![0.0; rows];
        s
    }

    #[inline]
    fn rng_next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Hot loop. `#[inline(always)]` so the caller (analyzer) folds in the
    /// per-element state into the dequant inner loop where possible.
    #[inline(always)]
    pub fn push(&mut self, x: f32) {
        self.n += 1;
        let xf = x as f64;
        let delta = xf - self.mean;

        // P\u{00e9}bay 2008 one-pass higher-moments update. All four accumulators
        // (mean, M2, M3, M4) are kept in sync using only O(1) state per push.
        let n_f = self.n as f64;
        let delta_n = delta / n_f;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n_f - 1.0);

        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n_f * n_f - 3.0 * n_f + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n_f - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;

        let ax = x.abs() as f64;
        self.abs_sum += ax;
        if ax > self.abs_max {
            self.abs_max = ax;
        }
        self.l2 += ax * ax;
        self.l1 += ax;

        // Branchless min/max (uses cmovs/csel on x86_64/AArch64).
        self.min = self.min.min(x);
        self.max = self.max.max(x);

        // Reservoir sample.
        if self.reservoir.len() < RESERVOIR_SIZE {
            self.reservoir.push(x);
        } else {
            let j = (self.rng_next() % self.n) as usize;
            if j < RESERVOIR_SIZE {
                self.reservoir[j] = x;
            }
        }

        let key = bucket_key(x);
        self.bucket_counts[key] = self.bucket_counts[key].saturating_add(1);

        // Per-channel (per-row) update. The analyzer pushes values in
        // row-major order; we just count columns to know when a row ends.
        if self.is_2d && self.current_row < self.n_rows {
            let r = self.current_row;
            self.row_sums[r] += xf;
            self.row_sumsq[r] += xf * xf;
            self.current_col += 1;
            if self.current_col >= self.n_cols {
                self.current_col = 0;
                self.current_row += 1;
            }
        }
    }

    pub fn finalize(&mut self, sparsity_eps: f32, sampled: bool) -> TensorStats {
        let n = self.n.max(1);
        let variance = self.m2 / n as f64;
        let std = variance.sqrt();
        let abs_mean = self.abs_sum / n as f64;
        let l2 = self.l2.sqrt();
        let l1 = self.l1;

        // Skewness / kurtosis: sampled tensors use the reservoir (small
        // exact computation); unsampled tensors use the streaming Welford
        // values that `push` has been updating.
        let (skewness, kurtosis) = if sampled {
            moments_from_reservoir(&self.reservoir)
        } else {
            moments_from_welford(self.m2, self.m3, self.m4, n)
        };

        // Per-channel (per-row) stats, only for 2-D accumulators with at
        // least one element pushed per row.
        let per_channel = if self.is_2d && self.n_rows > 0 {
            let mut channel_means = Vec::with_capacity(self.n_rows);
            let mut channel_stds = Vec::with_capacity(self.n_rows);
            let denom = (self.n_cols as f64).max(1.0);
            for r in 0..self.n_rows {
                let m = self.row_sums[r] / denom;
                let v = (self.row_sumsq[r] / denom - m * m).max(0.0);
                channel_means.push(m as f32);
                channel_stds.push(v.sqrt() as f32);
            }
            Some(PerChannelStats {
                n_channels: self.n_rows,
                channel_means,
                channel_stds,
            })
        } else {
            None
        };

        if !self.reservoir.is_empty() {
            let rlen = self.reservoir.len() as f64;
            let mut nz = 0u64;
            let mut fo = 0u64;
            for &v in &self.reservoir {
                if (v as f64).abs() < sparsity_eps as f64 {
                    nz += 1;
                }
                if (v as f64).abs() > abs_mean + OUTLIER_K * std {
                    fo += 1;
                }
            }
            let sparsity_abs = nz as f64 / rlen;
            let outlier_ratio = fo as f64 / rlen;

            self.reservoir
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p01 = pct(&self.reservoir, 0.001);
            let p50 = pct(&self.reservoir, 0.50);
            let p99 = pct(&self.reservoir, 0.99);
            let p999 = pct(&self.reservoir, 0.999);

            let mut unique = 0u32;
            let mut entropy = 0.0f64;
            let total: u64 = self.bucket_counts.iter().map(|&b| b as u64).sum();
            if total > 0 {
                for &b in &self.bucket_counts {
                    if b == 0 {
                        continue;
                    }
                    unique += 1;
                    let p = b as f64 / total as f64;
                    entropy -= p * p.log2();
                }
            }

            return TensorStats {
                n,
                n_sampled: sampled,
                mean: self.mean,
                variance,
                std,
                abs_mean,
                abs_max: self.abs_max,
                l2,
                l1,
                min: if self.min == f32::INFINITY {
                    0.0
                } else {
                    self.min
                },
                max: if self.max == f32::NEG_INFINITY {
                    0.0
                } else {
                    self.max
                },
                sparsity_abs,
                outlier_ratio,
                unique_bucket_count: unique,
                entropy_bits: entropy,
                p01,
                p50,
                p99,
                p999,
                skewness,
                kurtosis,
                per_channel,
            };
        }
        empty_stats()
    }
}

/// Compute skewness and (excess) kurtosis from Welford's running sums.
pub fn moments_from_welford(m2: f64, m3: f64, m4: f64, n: u64) -> (f64, f64) {
    if n < 2 || m2 <= 0.0 {
        return (0.0, 0.0);
    }
    let n_f = n as f64;
    let m = m2 / n_f;
    if m <= 0.0 {
        return (0.0, 0.0);
    }
    let m3_n = m3 / n_f;
    let m4_n = m4 / n_f;
    let skewness = m3_n / m.powf(1.5);
    let kurtosis = m4_n / (m * m) - 3.0;
    (skewness, kurtosis)
}

/// Compute skewness and (excess) kurtosis directly from a (small) sample of
/// values. Population moments (divided by n) are used for both.
pub fn moments_from_reservoir(values: &[f32]) -> (f64, f64) {
    if values.len() < 2 {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mut mean = 0.0f64;
    for &v in values {
        mean += v as f64;
    }
    mean /= n;
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    let mut m4 = 0.0f64;
    for &v in values {
        let d = (v as f64) - mean;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    if m2 <= 0.0 {
        return (0.0, 0.0);
    }
    (m3 / m2.powf(1.5), m4 / (m2 * m2) - 3.0)
}

#[inline]
pub fn pct(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q) as usize;
    sorted[idx]
}

#[inline]
pub fn bucket_key(x: f32) -> usize {
    if x == 0.0 {
        return 0;
    }
    if !x.is_finite() {
        return (1usize << N_BUCKETS_LOG2) - 1;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    (exp + 1024).clamp(0, (1 << N_BUCKETS_LOG2) - 2) as usize
}

/// Per-tensor sparsity threshold (used by the analyzer).
#[inline]
pub fn sparsity_eps_for(amax: f32) -> f32 {
    (amax * SPARSITY_EPS_REL).max(SPARSITY_EPS_ABS)
}

/// Just a re-export so `analysis::*` has a `Tensor` symbol; the analyzer
/// operates on `&[Tensor]` from the Model trait.
#[allow(dead_code)]
pub type _TensorAlias = Tensor;

#[cfg(test)]
#[path = "../../tests/unit/analysis/stats.rs"]
mod tests;
