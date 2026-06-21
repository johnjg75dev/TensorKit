//! Per-tensor + per-block scoring. Ported from the original scorer, with
//! types renamed to live in `analysis::score`.

use crate::analysis::stats::TensorStats;
use crate::model::Tensor;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorAnalysis {
    pub name: String,
    pub removable: f64,
    pub stats: TensorStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct BlockAnalysis {
    pub index: i32,
    pub label: String,
    pub role: BlockRole,
    pub removable: f64,
    pub total_bytes: u64,
    pub tensor_count: usize,
    pub neighbor_similarity: Option<f64>,
    pub tensors: Vec<TensorAnalysis>,
    /// Singular-value spectra for a small selection of tensors in this
    /// block, keyed by tensor name. Populated by the analyzer for at most
    /// a handful of high-signal tensors; empty otherwise.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub spectra: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BlockRole {
    Embedding,
    OutputHead,
    FinalNorm,
    Block,
    Other,
}

impl BlockRole {
    #[inline]
    pub fn is_prunable(self) -> bool {
        matches!(self, BlockRole::Block)
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Embedding => "embed",
            Self::OutputHead => "output",
            Self::FinalNorm => "norm",
            Self::Other => "other",
        }
    }
}

pub fn classify(name: &str) -> (BlockRole, i32, String) {
    if name.starts_with("token_embd") || name == "rope_freqs.weight" {
        return (BlockRole::Embedding, -1, "embed".into());
    }
    if name.starts_with("output") || name.starts_with("lm_head") {
        return (BlockRole::OutputHead, -1, "output".into());
    }
    if name.contains("output_norm") || name == "norm.weight" {
        return (BlockRole::FinalNorm, -1, "norm".into());
    }
    if let Some(rest) = name.strip_prefix("blk.") {
        let mut parts = rest.split('.');
        if let Some(idx_str) = parts.next()
            && let Ok(idx) = idx_str.parse::<i32>() {
                return (BlockRole::Block, idx, format!("blk.{idx}"));
            }
    }
    (BlockRole::Other, -1, "other".into())
}

pub fn score_tensor(name: &str, st: &TensorStats) -> f64 {
    let _ = name;
    if st.n == 0 {
        return 0.0;
    }
    let s_sparse = st.sparsity_abs.clamp(0.0, 1.0);
    let s_no_outlier = (1.0 - st.outlier_ratio).clamp(0.0, 1.0);
    let mag = st.abs_mean.max(1e-9).log10();
    let mag_norm = ((mag + 4.0) / 4.0).clamp(0.0, 1.0);
    let s_small = 1.0 - mag_norm;
    let s_low_entropy = (1.0 - st.entropy_bits / 12.0).clamp(0.0, 1.0);
    0.30 * s_sparse + 0.30 * s_no_outlier + 0.20 * s_small + 0.20 * s_low_entropy
}

pub fn per_block_scores(tensors: &[(Tensor, TensorStats)]) -> Vec<BlockAnalysis> {
    use std::collections::BTreeMap;
    let mut by_block: BTreeMap<(BlockRole, i32, String), Vec<(Tensor, TensorStats)>> =
        BTreeMap::new();
    for (ti, st) in tensors {
        let (role, idx, label) = classify(&ti.name);
        by_block
            .entry((role, idx, label))
            .or_default()
            .push((ti.clone(), st.clone()));
    }

    let mut out = Vec::new();
    for ((role, idx, label), entries) in by_block {
        let mut total_bytes: u64 = 0;
        let mut weighted: f64 = 0.0;
        let mut scored = Vec::with_capacity(entries.len());

        for (ti, st) in &entries {
            let s = score_tensor(&ti.name, st);
            let w = (ti.byte_size as f64).max(1.0);
            weighted += s * w;
            total_bytes += ti.byte_size;
            scored.push(TensorAnalysis {
                name: ti.name.clone(),
                removable: s,
                stats: st.clone(),
            });
        }

        let denom = (total_bytes as f64).max(1.0);
        let mut removable = weighted / denom;
        if !role.is_prunable() {
            removable = 0.0;
        }

        out.push(BlockAnalysis {
            index: idx,
            label,
            role,
            removable,
            total_bytes,
            tensor_count: entries.len(),
            neighbor_similarity: None,
            tensors: scored,
            spectra: HashMap::new(),
        });
    }
    out
}

/// Apply inter-block cosine similarity as an additional signal.
pub fn apply_neighbor_similarity(
    blocks: &mut [BlockAnalysis],
    block_features: &std::collections::HashMap<i32, Vec<f32>>,
) {
    const MAX_BONUS: f64 = 0.25;
    for b in blocks.iter_mut() {
        if b.index < 0 {
            continue;
        }
        let here = match block_features.get(&b.index) {
            Some(v) => v,
            None => continue,
        };
        let mut best: f64 = 0.0;
        for nb in [b.index - 1, b.index + 1] {
            if let Some(there) = block_features.get(&nb) {
                let s = cosine(here, there);
                if s.is_finite() && s > best {
                    best = s;
                }
            }
        }
        if best > 0.0 {
            b.neighbor_similarity = Some(best);
            b.removable = (b.removable + MAX_BONUS * best).clamp(0.0, 1.0);
        }
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}
