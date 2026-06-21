//! Streaming analyzer: read each tensor's bytes (zero-copy via mmap), dequant
//! up to `sample_per_tensor` elements, accumulate stats in a single pass.
//!
//! The per-tensor dequant + accumulate step is parallelized across tensors
//! via rayon (each tensor's bytes can be read + dequantized independently).
//! The score / recommend passes are sequential.

use crate::analysis::score::{classify, per_block_scores, score_tensor, BlockAnalysis};
use crate::analysis::spectrum::tensor_spectrum;
use crate::analysis::stats::{sparsity_eps_for, Accum, Analysis, TensorStats};
use crate::error::Result;
use crate::formats::gguf::dequant as gguf_dequant;
use crate::formats::gguf::types::GgmlType;
use crate::model::{Model, Tensor, TensorDtype};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Hard cap on the number of spectra stored in an `Analysis`. Keeps the
/// output report and the serialized JSON bounded regardless of model size.
const MAX_SPECTRA_PER_ANALYSIS: usize = 12;
const SPECTRA_PRUNABLE: usize = 8;
const SPECTRA_NON_PRUNABLE: usize = 4;
const SPECTRA_SVD_MAX_ELEMS: usize = 64;

pub struct Analyzer {
    sample_per_tensor: usize,
    keep: HashSet<String>,
    parallel: bool,
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Analyzer {
    pub fn new() -> Self {
        Self {
            sample_per_tensor: 200_000,
            keep: HashSet::new(),
            parallel: true,
        }
    }

    pub fn with_sample_per_tensor(n: usize) -> Self {
        Self {
            sample_per_tensor: n,
            keep: HashSet::new(),
            parallel: true,
        }
    }

    pub fn keep(mut self, name: impl Into<String>) -> Self {
        self.keep.insert(name.into());
        self
    }

    pub fn sample_per_tensor(&self) -> usize {
        self.sample_per_tensor
    }

    /// Enable or disable rayon-based parallelism for the per-tensor
    /// dequant + stats pass. Default is `true`.
    pub fn parallel(mut self, yes: bool) -> Self {
        self.parallel = yes;
        self
    }

    pub fn is_parallel(&self) -> bool {
        self.parallel
    }

    pub fn analyze<M: Model + ?Sized>(&self, model: &M) -> Result<Analysis> {
        let tensors = model.tensors();
        let total_bytes: u64 = tensors.iter().map(|t| t.byte_size).sum();
        let sample = self.sample_per_tensor;

        // Per-tensor result: the Tensor itself, the stats, and the
        // dequantized sample (kept around so we can compute a few
        // spectra without re-reading + re-dequantizing the file).
        let per_tensor: Vec<(Tensor, TensorStats, Vec<f32>)> =
            if self.parallel && tensors.len() > 1 {
                tensors
                    .par_iter()
                    .map(|t| analyze_tensor(model, t, sample))
                    .filter_map(|r| r.ok().flatten())
                    .collect()
            } else {
                tensors
                    .iter()
                    .map(|t| analyze_tensor(model, t, sample))
                    .filter_map(|r| r.ok().flatten())
                    .collect()
            };

        // `per_block_scores` only needs the Tensor + stats, so build a
        // thin view for it.
        let per_tensor_view: Vec<(Tensor, TensorStats)> = per_tensor
            .iter()
            .map(|(t, s, _)| (t.clone(), s.clone()))
            .collect();
        let mut blocks = per_block_scores(&per_tensor_view);

        // Populate spectra for a small selection of high-signal tensors.
        populate_spectra(&mut blocks, &per_tensor);

        let (recommendation, recommendation_count, estimated_bytes_after_prune) =
            recommend(&blocks, total_bytes);

        Ok(Analysis {
            blocks,
            recommendation,
            recommendation_count,
            estimated_bytes_after_prune,
            sample_per_tensor: sample,
            total_tensors: tensors.len(),
            total_bytes,
            model_name: model.name().map(|s| s.to_string()),
            architecture: model.architecture().map(|s| s.to_string()),
            file_size: Some(total_bytes),
        })
    }
}

fn analyze_tensor<M: Model + ?Sized>(
    model: &M,
    t: &Tensor,
    sample: usize,
) -> Result<Option<(Tensor, TensorStats, Vec<f32>)>> {
    // Sample limit: if we don't need to look at the whole tensor, only
    // dequant `sample` elements. This is the key streaming optimization.
    let max_elems = if t.shape.iter().product::<u64>() <= sample as u64 {
        None
    } else {
        Some(sample)
    };

    let bytes = model.read_tensor_bytes(&t.name)?;
    let dequantized = dequant_for_dtype(t.dtype, &bytes, max_elems);

    let Some(values) = dequantized else {
        return Ok(None);
    };

    // Compute the adaptive sparsity threshold from the values' amax.
    let amax = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let eps = sparsity_eps_for(amax);

    // 2-D tensors carry per-row stats (axis 0 reduction).
    let is_2d = t.shape.len() == 2;
    let mut acc = if is_2d {
        let rows = t.shape[0] as usize;
        let cols = if t.shape.len() >= 2 {
            t.shape[1] as usize
        } else {
            0
        };
        Accum::new_2d(rows.max(1), cols.max(1))
    } else {
        Accum::new()
    };
    for &v in &values {
        acc.push(v);
    }
    let sampled = max_elems.is_some();
    let stats = acc.finalize(eps, sampled);
    Ok(Some((t.clone(), stats, values)))
}

fn dequant_for_dtype(
    dtype: TensorDtype,
    bytes: &[u8],
    max_elems: Option<usize>,
) -> Option<Vec<f32>> {
    let gg_ty = match dtype {
        TensorDtype::F32 => GgmlType::F32,
        TensorDtype::F16 => GgmlType::F16,
        TensorDtype::Bf16 => GgmlType::Bf16,
        TensorDtype::F64 => GgmlType::F64,
        TensorDtype::I8 => GgmlType::I8,
        TensorDtype::I16 => GgmlType::I16,
        TensorDtype::I32 => GgmlType::I32,
        TensorDtype::I64 => GgmlType::I64,
        TensorDtype::Q4_0 => GgmlType::Q4_0,
        TensorDtype::Q4_1 => GgmlType::Q4_1,
        TensorDtype::Q5_0 => GgmlType::Q5_0,
        TensorDtype::Q5_1 => GgmlType::Q5_1,
        TensorDtype::Q8_0 => GgmlType::Q8_0,
        TensorDtype::Q8_1 => GgmlType::Q8_1,
        TensorDtype::Q2K => GgmlType::Q2K,
        TensorDtype::Q3K => GgmlType::Q3K,
        TensorDtype::Q4K => GgmlType::Q4K,
        TensorDtype::Q5K => GgmlType::Q5K,
        TensorDtype::Q6K => GgmlType::Q6K,
        TensorDtype::Q8K => GgmlType::Q8K,
        _ => return None,
    };
    gguf_dequant::dequantize(gg_ty, bytes, max_elems)
}

/// Pick the top-N prunable blocks (highest `removable`), where N is 15% of
/// prunable blocks rounded.
fn recommend(blocks: &[BlockAnalysis], total_bytes: u64) -> (Vec<i32>, usize, u64) {
    let mut prunable: Vec<&BlockAnalysis> =
        blocks.iter().filter(|b| b.role.is_prunable()).collect();
    prunable.sort_by(|a, b| {
        b.removable
            .partial_cmp(&a.removable)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = ((prunable.len() as f64) * 0.15).round() as usize;
    let n = n.max(1).min(prunable.len());
    let recommended: Vec<i32> = prunable.iter().take(n).map(|b| b.index).collect();
    let saved: u64 = prunable.iter().take(n).map(|b| b.total_bytes).sum();
    (recommended, n, total_bytes.saturating_sub(saved))
}

/// Compute spectra for a small selection of tensors and attach them to the
/// corresponding `BlockAnalysis` entries.
///
/// Selection:
///   * Top-`SPECTRA_PRUNABLE` prunable tensors (role == Block), ranked by
///     the per-tensor removable score.
///   * `SPECTRA_NON_PRUNABLE` non-prunable tensors, taken as a stride-based
///     pseudo-random sample for determinism.
///   * At most `MAX_SPECTRA_PER_ANALYSIS` spectra overall.
fn populate_spectra(
    blocks: &mut [BlockAnalysis],
    per_tensor: &[(Tensor, TensorStats, Vec<f32>)],
) {
    if per_tensor.is_empty() {
        return;
    }

    // Map tensor name -> block index (for attaching spectra later).
    let mut name_to_block: HashMap<String, usize> = HashMap::new();
    for (bi, b) in blocks.iter().enumerate() {
        for ta in &b.tensors {
            name_to_block.insert(ta.name.clone(), bi);
        }
    }

    // Split into prunable vs non-prunable.
    let mut prunable: Vec<&(Tensor, TensorStats, Vec<f32>)> = Vec::new();
    let mut non_prunable: Vec<&(Tensor, TensorStats, Vec<f32>)> = Vec::new();
    for entry in per_tensor {
        let (role, _, _) = classify(&entry.0.name);
        if role.is_prunable() {
            prunable.push(entry);
        } else {
            non_prunable.push(entry);
        }
    }

    // Prunable: top-N by per-tensor removable score, descending.
    prunable.sort_by(|a, b| {
        let sa = score_tensor(&a.0.name, &a.1);
        let sb = score_tensor(&b.0.name, &b.1);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_prunable = prunable.len().min(SPECTRA_PRUNABLE);

    // Non-prunable: stride-based deterministic sample so reports are
    // reproducible across runs.
    let n_non_prunable_target = non_prunable.len().min(SPECTRA_NON_PRUNABLE);
    let stride = if non_prunable.is_empty() || n_non_prunable_target == 0 {
        1
    } else {
        (non_prunable.len() / n_non_prunable_target).max(1)
    };
    let non_prunable_sample: Vec<&(Tensor, TensorStats, Vec<f32>)> = non_prunable
        .iter()
        .step_by(stride)
        .take(n_non_prunable_target)
        .copied()
        .collect();

    let mut spectra_count = 0usize;
    for entry in prunable
        .iter()
        .take(n_prunable)
        .chain(non_prunable_sample.iter())
    {
        if spectra_count >= MAX_SPECTRA_PER_ANALYSIS {
            break;
        }
        let (t, _, values) = entry;
        if t.shape.len() != 2 {
            continue;
        }
        let m = t.shape[0] as usize;
        let n = t.shape[1] as usize;
        let Some(spec) = tensor_spectrum(values, m, n, Some(SPECTRA_SVD_MAX_ELEMS)) else {
            continue;
        };
        if let Some(&bi) = name_to_block.get(&t.name) {
            blocks[bi].spectra.insert(t.name.clone(), spec);
            spectra_count += 1;
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/analysis/analyzer.rs"]
mod tests;
