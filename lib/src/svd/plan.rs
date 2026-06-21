//! Build an `SvdPlan` from an `SvdConfig` and a model.
//!
//! The plan is the data structure the writer consumes. For each eligible
//! tensor we record the original name, the resolved rank, and the names of
//! the two new factors (A: m x k, B: k x n).

use crate::analysis::score::{classify, BlockRole};
use crate::error::Result;
use crate::model::Model;
use crate::svd::config::{is_2d_weight, LayerSelection, SvdConfig};
use std::collections::HashSet;

/// One tensor to be replaced by an (A, B) low-rank factorization.
#[derive(Debug, Clone)]
pub struct SvdTarget {
    pub name: String,
    pub name_a: String,
    pub name_b: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub orig_bytes: u64,
    pub new_bytes: u64,
}

/// Concrete plan describing every compression to perform.
#[derive(Debug, Clone)]
pub struct SvdPlan {
    pub targets: Vec<SvdTarget>,
    pub skipped: Vec<SkippedTensor>,
    pub config: SvdConfig,
    pub original_block_count: i32,
}

#[derive(Debug, Clone)]
pub struct SkippedTensor {
    pub name: String,
    pub reason: String,
}

pub fn build_plan<M: Model + ?Sized>(model: &M, cfg: &SvdConfig) -> Result<SvdPlan> {
    // 1) Discover all block indices present in the model.
    let mut all_blocks: HashSet<i32> = HashSet::new();
    for t in model.tensors() {
        let (role, idx, _) = classify(&t.name);
        if role == BlockRole::Block {
            all_blocks.insert(idx);
        }
    }
    let mut sorted_blocks: Vec<i32> = all_blocks.into_iter().collect();
    sorted_blocks.sort();

    // 2) Build the set of allowed block indices from the layer selection.
    let allowed_blocks: Option<HashSet<i32>> = match &cfg.layers {
        LayerSelection::All
        | LayerSelection::AllAttn
        | LayerSelection::AllFfn
        | LayerSelection::AllMlp => None,
        LayerSelection::Indices(v) => Some(v.iter().copied().collect()),
        LayerSelection::Pattern(_) => None, // matched per-tensor below
    };

    let mut targets = Vec::new();
    let mut skipped = Vec::new();

    for t in model.tensors() {
        // Layer filter
        let (role, idx, _) = classify(&t.name);
        if role != BlockRole::Block {
            continue;
        }
        if let Some(set) = &allowed_blocks
            && !set.contains(&idx) {
                continue;
            }
        if let LayerSelection::Pattern(re) = &cfg.layers
            && !re.is_match(&t.name) {
                continue;
            }
        // Convenience aliases: only act on the relevant suffix family.
        match &cfg.layers {
            LayerSelection::AllAttn | LayerSelection::AllFfn | LayerSelection::AllMlp => {
                let ok = match &cfg.layers {
                    LayerSelection::AllAttn => {
                        cfg.tensors.matches(&t.name)
                            && crate::svd::config::suffix_in(
                                &t.name,
                                crate::svd::config::ATTN_SUFFIXES,
                            )
                    }
                    LayerSelection::AllFfn => {
                        crate::svd::config::suffix_in(&t.name, crate::svd::config::FFN_SUFFIXES)
                    }
                    LayerSelection::AllMlp => {
                        crate::svd::config::suffix_in(&t.name, crate::svd::config::ATTN_SUFFIXES)
                            || crate::svd::config::suffix_in(
                                &t.name,
                                crate::svd::config::FFN_SUFFIXES,
                            )
                    }
                    _ => unreachable!(),
                };
                if !ok {
                    continue;
                }
            }
            _ => {}
        }

        // Tensor filter
        if !cfg.tensors.matches(&t.name) {
            continue;
        }
        if !is_2d_weight(&t.name) {
            continue;
        }

        // Shape filter (2D weight matrix m x n)
        let shape: Vec<u64> = t.shape.iter().copied().filter(|&d| d > 0).collect();
        if shape.len() != 2 {
            skipped.push(SkippedTensor {
                name: t.name.clone(),
                reason: format!("non-2D shape {:?}", t.shape),
            });
            continue;
        }
        let m = shape[0] as usize;
        let n = shape[1] as usize;

        if m.min(n) < cfg.min_dim {
            skipped.push(SkippedTensor {
                name: t.name.clone(),
                reason: format!("min dim {} < {}", m.min(n), cfg.min_dim),
            });
            continue;
        }

        // For Energy rank specs we have to read the full tensor to compute S.
        // For other specs we can resolve the rank from shape alone.
        let (name_a, name_b) = cfg.factor_names(&t.name);
        let k = match &cfg.rank.spec {
            crate::svd::config::RankSpec::Energy(_) => {
                // Defer rank resolution to apply time (where we already have the bytes).
                // Plan stores k = 0 as a sentinel; apply replaces it.
                0
            }
            _ => cfg.resolve_rank(&t.name, idx, m, n, None),
        };
        // Compute output element size for byte estimate.
        let esz = match cfg.dtype {
            crate::svd::config::OutputDtype::F32 => 4,
            crate::svd::config::OutputDtype::F16
            | crate::svd::config::OutputDtype::Bf16 => 2,
            crate::svd::config::OutputDtype::AutoQuant => {
                // Match auto_pick_quant logic: float/int sources → F16 (esz=2),
                // quantized sources → Q8_0 (esz from block_bytes).
                let src_is_float = matches!(
                    t.dtype,
                    crate::model::TensorDtype::F32
                        | crate::model::TensorDtype::F16
                        | crate::model::TensorDtype::Bf16
                        | crate::model::TensorDtype::F64
                        | crate::model::TensorDtype::I8
                        | crate::model::TensorDtype::I16
                        | crate::model::TensorDtype::I32
                        | crate::model::TensorDtype::I64
                );
                if src_is_float {
                    2
                } else {
                    crate::formats::gguf::types::GgmlType::Q8_0
                        .block_bytes()
                        .unwrap_or(34) as u64
                        / 32
                }
            }
            crate::svd::config::OutputDtype::Ggml(t) => {
                // Quantized: 1 byte per 32 values is a safe lower-bound estimate
                // for the compression_ratio reporting. Real byte size is
                // computed precisely in apply.rs.
                t.block_bytes().unwrap_or(34) as u64 / 32
            }
        };
        let new_bytes = ((m as u64 * k as u64) + (k as u64 * n as u64)) * esz;
        targets.push(SvdTarget {
            name: t.name.clone(),
            name_a,
            name_b,
            m,
            n,
            k,
            orig_bytes: t.byte_size,
            new_bytes,
        });
    }

    targets.sort_by(|a, b| a.name.cmp(&b.name));

    // 3) Adjacent pass: for every primary target, add targets for each
    //    (block_idx + offset, role) pair from cfg.adjacent. Adjacent
    //    targets bypass the layer + tensor selection filters; the user is
    //    explicitly saying "also compress this". Out-of-range offsets are
    //    recorded as SkippedTensor rather than failing the whole plan.
    if let Some(adj) = &cfg.adjacent {
        let (min_block, max_block) = match (sorted_blocks.first(), sorted_blocks.last()) {
            (Some(lo), Some(hi)) => (*lo, *hi),
            _ => (0, -1), // no blocks in the model
        };
        let mut existing: HashSet<String> = targets.iter().map(|t| t.name.clone()).collect();
        let mut added: Vec<SvdTarget> = Vec::new();
        for primary in &targets {
            let (role, primary_idx, _) = classify(&primary.name);
            if role != BlockRole::Block {
                continue;
            }
            for entry in &adj.entries {
                let adj_idx = primary_idx + entry.offset;
                let adj_name =
                    format!("blk.{adj_idx}.{}.weight", entry.role.as_str());

                if adj_idx < min_block || adj_idx > max_block {
                    skipped.push(SkippedTensor {
                        name: adj_name,
                        reason: "out-of-range block offset".into(),
                    });
                    continue;
                }
                if existing.contains(&adj_name) {
                    // Either already a primary, or already added by a
                    // previous iteration. Keep one copy.
                    continue;
                }
                let t = match model.tensor(&adj_name) {
                    Some(t) => t,
                    None => {
                        skipped.push(SkippedTensor {
                            name: adj_name,
                            reason: "adjacent tensor not found in model".into(),
                        });
                        continue;
                    }
                };
                let shape: Vec<u64> = t.shape.iter().copied().filter(|&d| d > 0).collect();
                if shape.len() != 2 {
                    skipped.push(SkippedTensor {
                        name: adj_name,
                        reason: format!("non-2D shape {:?}", t.shape),
                    });
                    continue;
                }
                let m = shape[0] as usize;
                let n = shape[1] as usize;
                if m.min(n) < cfg.min_dim {
                    skipped.push(SkippedTensor {
                        name: adj_name,
                        reason: format!("min dim {} < {}", m.min(n), cfg.min_dim),
                    });
                    continue;
                }
                if !is_2d_weight(&t.name) {
                    skipped.push(SkippedTensor {
                        name: adj_name,
                        reason: "not a 2D weight".into(),
                    });
                    continue;
                }
                let (name_a, name_b) = cfg.factor_names(&t.name);
                let k = match &cfg.rank.spec {
                    crate::svd::config::RankSpec::Energy(_) => 0,
                    _ => cfg.resolve_rank(&t.name, adj_idx, m, n, None),
                };
                let esz = match cfg.dtype {
                    crate::svd::config::OutputDtype::F32 => 4,
                    crate::svd::config::OutputDtype::F16
                    | crate::svd::config::OutputDtype::Bf16
                    | crate::svd::config::OutputDtype::AutoQuant => 2,
                    crate::svd::config::OutputDtype::Ggml(t) => {
                        t.block_bytes().unwrap_or(34) as u64 / 32
                    }
                };
                let new_bytes = ((m as u64 * k as u64) + (k as u64 * n as u64)) * esz;
                existing.insert(adj_name.clone());
                added.push(SvdTarget {
                    name: adj_name,
                    name_a,
                    name_b,
                    m,
                    n,
                    k,
                    orig_bytes: t.byte_size,
                    new_bytes,
                });
            }
        }
        targets.extend(added);
        targets.sort_by(|a, b| a.name.cmp(&b.name));
    }

    Ok(SvdPlan {
        targets,
        skipped,
        config: cfg.clone(),
        original_block_count: sorted_blocks.len() as i32,
    })
}

impl SvdPlan {
    pub fn orig_bytes(&self) -> u64 {
        self.targets.iter().map(|t| t.orig_bytes).sum()
    }
    pub fn new_bytes(&self) -> u64 {
        self.targets.iter().map(|t| t.new_bytes).sum()
    }
    pub fn compression_ratio(&self) -> f64 {
        let o = self.orig_bytes() as f64;
        if o == 0.0 {
            0.0
        } else {
            1.0 - (self.new_bytes() as f64 / o)
        }
    }
    /// Names of tensors that the writer should drop from the source.
    pub fn dropped_names(&self) -> HashSet<&str> {
        self.targets.iter().map(|t| t.name.as_str()).collect()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/svd/plan.rs"]
mod tests;
