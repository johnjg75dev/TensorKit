//! Task-vector arithmetic: compute, apply, compose, and TIES-merge.
//!
//! Based on:
//! - Ilharco et al. 2022 (arXiv 2212.04089) — task vectors
//! - Yadav et al. 2023 (arXiv 2306.01708) — TIES-Merging
//!
//! All operations are pure Rust, no I/O, operating on `&[f32]` slices.

use crate::error::{Error, Result};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Task vectors
// ---------------------------------------------------------------------------

/// Compute a single task vector: τ = θ_finetuned − θ_base.
///
/// # Errors
/// Returns [`Error::TaskArith`] if `base.len() != finetuned.len()`.
pub fn compute_task_vector(base: &[f32], finetuned: &[f32]) -> Result<Vec<f32>> {
    if base.len() != finetuned.len() {
        return Err(Error::TaskArith(format!(
            "length mismatch: base has {} elements, finetuned has {}",
            base.len(),
            finetuned.len()
        )));
    }
    let tau: Vec<f32> = base
        .iter()
        .zip(finetuned.iter())
        .map(|(b, f)| f - b)
        .collect();

    // Scan for NaN/Inf — warn but still return the vector.
    let (nan_count, inf_count) = scan_nan_inf(&tau);
    if nan_count > 0 || inf_count > 0 {
        eprintln!(
            "[warn] task vector contains {} NaN and {} Inf values (of {} total)",
            nan_count,
            inf_count,
            tau.len()
        );
    }

    Ok(tau)
}

/// Apply a task vector into `out`: `out[i] = base[i] + α · tau[i]`.
///
/// # Panics
/// Panics if `out.len() != base.len()` or `base.len() != tau.len()`.
pub fn apply_task_vector(out: &mut [f32], base: &[f32], tau: &[f32], alpha: f32) {
    assert_eq!(
        out.len(),
        base.len(),
        "apply_task_vector: out length {} != base length {}",
        out.len(),
        base.len()
    );
    assert_eq!(
        base.len(),
        tau.len(),
        "apply_task_vector: base length {} != tau length {}",
        base.len(),
        tau.len()
    );
    for i in 0..base.len() {
        out[i] = base[i] + alpha * tau[i];
    }
}

/// Apply a task vector, returning a new `Vec`.
pub fn apply_task_vector_owned(base: &[f32], tau: &[f32], alpha: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; base.len()];
    apply_task_vector(&mut out, base, tau, alpha);
    out
}

// ---------------------------------------------------------------------------
// TIES-Merging
// ---------------------------------------------------------------------------

/// Configuration for the TIES trim → elect-sign → merge pipeline.
#[derive(Debug, Clone)]
pub struct TiesConfig {
    /// Fraction of coordinates to keep per task vector (e.g. `0.2` = top 20%).
    /// Must be in `(0.0, 1.0]`.
    pub density: f64,
    /// If `true`, resolve sign conflicts by majority vote before merging.
    pub elect_sign: bool,
}

/// Trim a task vector: keep only the top-`density` fraction by magnitude,
/// zero out the rest.
///
/// # Errors
/// Returns [`Error::TaskArith`] if `density` is not in `(0.0, 1.0]`.
pub fn trim_task_vector(tau: &[f32], density: f64) -> Result<Vec<f32>> {
    if !(0.0 < density && density <= 1.0) {
        return Err(Error::TaskArith(format!(
            "TIES density must be in (0, 1], got {density}"
        )));
    }

    let dim = tau.len();
    let keep_count = (density * dim as f64).ceil() as usize;
    let keep_count = keep_count.min(dim);

    if keep_count == dim {
        return Ok(tau.to_vec());
    }

    // Build index list sorted by |τ| descending.
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_by(|&a, &b| {
        tau[b]
            .abs()
            .partial_cmp(&tau[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = vec![0.0f32; dim];
    for &idx in &indices[..keep_count] {
        out[idx] = tau[idx];
    }
    Ok(out)
}

/// Elect sign per coordinate across N task vectors.
///
/// Returns a `Vec` of signs (`+1.0` or `-1.0`) of length equal to the
/// task vectors. Ties (equal positive/negative counts) resolve to `+1.0`.
/// All-zero coordinates also resolve to `+1.0`.
pub fn elect_sign(task_vectors: &[&[f32]]) -> Vec<f32> {
    if task_vectors.is_empty() {
        return Vec::new();
    }
    let dim = task_vectors[0].len();
    let mut signs = Vec::with_capacity(dim);

    for i in 0..dim {
        let mut pos = 0i32;
        let mut neg = 0i32;
        for tv in task_vectors {
            match tv[i].partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Greater => pos += 1,
                std::cmp::Ordering::Less => neg += 1,
                std::cmp::Ordering::Equal => {} // zero — no vote
            }
        }
        signs.push(if pos >= neg { 1.0 } else { -1.0 });
    }
    signs
}

/// Merge N task vectors using the TIES algorithm.
///
/// 1. **Trim** each vector to keep top-`density` fraction by magnitude.
/// 2. **Elect sign** per coordinate (if `config.elect_sign` is `true`).
/// 3. **Merge**: average only values whose sign agrees with the elected sign.
///
/// Returns the merged task vector of length `dim`.
///
/// # Errors
/// Returns [`Error::TaskArith`] if density is out of range or no vectors provided.
pub fn ties_merge(task_vectors: &[&[f32]], config: &TiesConfig) -> Result<Vec<f32>> {
    if task_vectors.is_empty() {
        return Err(Error::TaskArith(
            "ties_merge requires at least one task vector".into(),
        ));
    }

    if !(0.0 < config.density && config.density <= 1.0) {
        return Err(Error::TaskArith(format!(
            "TIES density must be in (0, 1], got {}",
            config.density
        )));
    }

    let dim = task_vectors[0].len();
    let n = task_vectors.len();

    // Step 1: Trim each task vector.
    let trimmed: Vec<Vec<f32>> = task_vectors
        .iter()
        .map(|tv| trim_task_vector(tv, config.density))
        .collect::<Result<Vec<_>>>()?;

    // Step 2: Elect sign.
    let trimmed_refs: Vec<&[f32]> = trimmed.iter().map(|v| v.as_slice()).collect();
    let signs = if config.elect_sign {
        elect_sign(&trimmed_refs)
    } else {
        // No sign election — default to positive (keep all agreeing positive values,
        // which is equivalent to naive trimmed average when signs are ignored).
        vec![1.0f32; dim]
    };

    // Step 3: Merge — for each coordinate, average the values whose sign
    // matches the elected sign.
    let mut merged = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for tv in &trimmed {
            let val = tv[i];
            if val == 0.0 {
                continue; // already trimmed — skip
            }
            let val_sign = if val > 0.0 { 1.0 } else { -1.0 };
            if (val_sign - signs[i]).abs() < f36::EPSILON as f32 {
                sum += val;
                count += 1;
            }
        }
        if count > 0 {
            merged[i] = sum / count as f32;
        }
    }

    Ok(merged)
}

// ---------------------------------------------------------------------------
// Multiplier Hub
// ---------------------------------------------------------------------------

/// A named task vector with a scalar multiplier.
#[derive(Debug, Clone)]
pub struct TaskEntry {
    pub name: String,
    pub vector: Vec<f32>,
    pub alpha: f32,
}

/// The multiplier hub: a collection of named task vectors that can be
/// composited onto a base model in one pass.
///
/// ```text
/// θ_out = θ_base + Σ (α_k · τ_k)
/// ```
#[derive(Debug, Clone, Default)]
pub struct MultiplierHub {
    entries: Vec<TaskEntry>,
}

impl MultiplierHub {
    /// Create an empty hub.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a task vector with a given multiplier.
    pub fn add(&mut self, name: impl Into<String>, vector: Vec<f32>, alpha: f32) {
        self.entries.push(TaskEntry {
            name: name.into(),
            vector,
            alpha,
        });
    }

    /// Remove a registered task vector by name. Returns `true` if found.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.name != name);
        self.entries.len() < before
    }

    /// Update the multiplier for an existing task vector. Returns `true` if found.
    pub fn set_alpha(&mut self, name: &str, alpha: f32) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == name) {
            entry.alpha = alpha;
            true
        } else {
            false
        }
    }

    /// List all registered task vectors and their multipliers.
    pub fn list(&self) -> Vec<(&str, f32)> {
        self.entries.iter().map(|e| (e.name.as_str(), e.alpha)).collect()
    }

    /// Apply all registered task vectors to a base tensor.
    ///
    /// ```text
    /// out[i] = base[i] + Σ (entry.alpha * entry.vector[i])
    /// ```
    ///
    /// If the hub is empty, `out` is a copy of `base`.
    ///
    /// # Panics
    /// Panics if `out.len() != base.len()`, or if any entry's vector length
    /// differs from `base.len()`.
    pub fn apply(&self, out: &mut [f32], base: &[f32]) {
        assert_eq!(
            out.len(),
            base.len(),
            "MultiplierHub::apply: out length {} != base length {}",
            out.len(),
            base.len()
        );
        out.copy_from_slice(base);
        for entry in &self.entries {
            assert_eq!(
                entry.vector.len(),
                base.len(),
                "MultiplierHub::apply: entry '{}' vector length {} != base length {}",
                entry.name,
                entry.vector.len(),
                base.len()
            );
            for i in 0..base.len() {
                out[i] += entry.alpha * entry.vector[i];
            }
        }
    }

    /// Apply all registered task vectors, returning a new `Vec`.
    pub fn apply_owned(&self, base: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; base.len()];
        self.apply(&mut out, base);
        out
    }

    /// Composite: first TIES-merge all registered vectors, then apply.
    ///
    /// Equivalent to:
    /// ```text
    /// θ_out = θ_base + ties_merge({α_k · τ_k})
    /// ```
    pub fn apply_ties(&self, base: &[f32], config: &TiesConfig) -> Result<Vec<f32>> {
        if self.entries.is_empty() {
            return Ok(base.to_vec());
        }

        // Scale each vector by its alpha before merging.
        let scaled: Vec<Vec<f32>> = self
            .entries
            .iter()
            .map(|e| {
                e.vector
                    .iter()
                    .map(|&v| e.alpha * v)
                    .collect()
            })
            .collect();

        let refs: Vec<&[f32]> = scaled.iter().map(|v| v.as_slice()).collect();
        let merged = ties_merge(&refs, config)?;

        Ok(apply_task_vector_owned(base, &merged, 1.0))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Scan a vector for NaN and Inf values. Returns `(nan_count, inf_count)`.
fn scan_nan_inf(v: &[f32]) -> (usize, usize) {
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let inf = v.iter().filter(|x| x.is_infinite()).count();
    (nan, inf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/merge/task_arith.rs"]
mod tests;
