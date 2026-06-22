//! Interpretability analysis tools for captured activations.
//!
//! Provides functions for analyzing attention patterns, MoE expert
//! utilization, neuron-level statistics, and activation properties
//! across transformer layers.

use super::activations::InterpretationSnapshot;
use super::layer::BlockActivations;
use super::ops;

/// Per-expert statistics across all MoE layers.
#[derive(Debug, Clone)]
pub struct ExpertStats {
    pub block_idx: usize,
    pub expert_idx: usize,
    pub selection_count: usize,
    pub avg_weight: f32,
    pub total_weight: f32,
    pub avg_gate_l2: f32,
    pub avg_up_l2: f32,
    pub avg_down_l2: f32,
    pub activation_sparsity: f32,
}

/// Summary of MoE routing behavior across the model.
#[derive(Debug, Clone)]
pub struct MoERoutingStats {
    pub total_tokens: usize,
    pub expert_stats: Vec<ExpertStats>,
    pub per_layer_load_balance: Vec<f32>,
}

/// Per-head attention statistics.
#[derive(Debug, Clone)]
pub struct PerHeadStat {
    pub head_idx: usize,
    pub kv_head_idx: usize,
    pub score: f32,
    pub weight: f32,
    pub q_norm: f32,
    pub k_norm: f32,
    pub v_norm: f32,
    pub output_norm: f32,
}

/// A norm value at a specific point in the residual stream.
#[derive(Debug, Clone)]
pub struct ResidualNorm {
    pub block_idx: Option<usize>,
    pub stage: ResidualStage,
    pub l2: f32,
}

/// Which point in the residual stream a norm measurement comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualStage {
    Embedding,
    BlockInput,
    PostAttention,
    BlockOutput,
    FinalNorm,
}

/// Router probability distribution for a single block.
#[derive(Debug, Clone)]
pub struct RouterDistribution {
    pub block_idx: usize,
    pub probs: Vec<f32>,
    pub selected: Vec<usize>,
    pub weights: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Attention analysis
// ---------------------------------------------------------------------------

/// Compute attention entropy for a single head.
/// Higher entropy = more uniform attention (less focused).
/// Lower entropy = more peaked attention (more focused).
pub fn attention_entropy(head: &super::layer::HeadActivations) -> f32 {
    let w = head.weight;
    if w <= 0.0 {
        return 0.0;
    }
    -w * w.ln()
}

/// Compute relative importance of each attention head in a block.
pub fn attention_head_importance(block: &BlockActivations) -> Vec<f32> {
    let mut scores: Vec<f32> = block
        .head_activations
        .iter()
        .map(|h| {
            let l2 = ops::l2_norm(&h.output);
            l2 * h.weight.abs()
        })
        .collect();

    let total: f32 = scores.iter().sum();
    if total > 0.0 {
        for s in scores.iter_mut() {
            *s /= total;
        }
    }
    scores
}

/// Compute per-head attention statistics.
pub fn per_head_stats(block: &BlockActivations) -> Vec<PerHeadStat> {
    block
        .head_activations
        .iter()
        .map(|h| PerHeadStat {
            head_idx: h.head_idx,
            kv_head_idx: h.kv_head_idx,
            score: h.score,
            weight: h.weight,
            q_norm: ops::l2_norm(&h.q),
            k_norm: ops::l2_norm(&h.k),
            v_norm: ops::l2_norm(&h.v),
            output_norm: ops::l2_norm(&h.output),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Residual stream analysis
// ---------------------------------------------------------------------------

/// Track how norms evolve through the residual stream.
pub fn residual_stream_norms(snapshot: &InterpretationSnapshot) -> Vec<ResidualNorm> {
    let mut norms = Vec::with_capacity(snapshot.blocks.len() + 2);

    norms.push(ResidualNorm {
        block_idx: None,
        stage: ResidualStage::Embedding,
        l2: ops::l2_norm(&snapshot.embedding),
    });

    for blk in &snapshot.blocks {
        norms.push(ResidualNorm {
            block_idx: Some(blk.block_idx),
            stage: ResidualStage::BlockInput,
            l2: ops::l2_norm(&blk.hidden_input),
        });
        norms.push(ResidualNorm {
            block_idx: Some(blk.block_idx),
            stage: ResidualStage::PostAttention,
            l2: ops::l2_norm(&blk.attn_residual),
        });
        norms.push(ResidualNorm {
            block_idx: Some(blk.block_idx),
            stage: ResidualStage::BlockOutput,
            l2: ops::l2_norm(&blk.hidden_output),
        });
    }

    norms.push(ResidualNorm {
        block_idx: None,
        stage: ResidualStage::FinalNorm,
        l2: ops::l2_norm(&snapshot.final_norm),
    });

    norms
}

// ---------------------------------------------------------------------------
// FFN / Neuron analysis
// ---------------------------------------------------------------------------

/// Find the top-k most active FFN neurons in a block.
pub fn top_ffn_neurons(block: &BlockActivations, k: usize) -> Vec<(usize, f32)> {
    if block.ffn_gated.is_empty() {
        return vec![];
    }

    let mut indexed: Vec<(usize, f32)> = block
        .ffn_gated
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).collect()
}

/// Find neurons that are "dead" (nearly zero activation) across all layers.
pub fn dead_ffn_neurons(blocks: &[BlockActivations], threshold: f32) -> Vec<Vec<usize>> {
    blocks
        .iter()
        .map(|b| {
            if b.ffn_gated.is_empty() {
                return vec![];
            }
            b.ffn_gated
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v.abs() < threshold)
                .map(|(i, _)| i)
                .collect()
        })
        .collect()
}

/// Compute activation sparsity per layer (fraction of near-zero activations).
pub fn activation_sparsity_per_layer(blocks: &[BlockActivations]) -> Vec<f32> {
    blocks
        .iter()
        .map(|b| {
            if b.ffn_gated.is_empty() {
                return 0.0;
            }
            let n = b.ffn_gated.len() as f32;
            let dead = b.ffn_gated.iter().filter(|&&v| v.abs() < 1e-6).count() as f32;
            dead / n
        })
        .collect()
}

// ---------------------------------------------------------------------------
// MoE analysis
// ---------------------------------------------------------------------------

/// Compute per-expert statistics across the model.
pub fn expert_utilization(snapshot: &InterpretationSnapshot) -> MoERoutingStats {
    let mut expert_map: std::collections::HashMap<(usize, usize), ExpertAccumulator> =
        std::collections::HashMap::new();
    let mut per_layer_balance = Vec::new();
    let mut total_tokens = 0;

    for blk in &snapshot.blocks {
        if let Some(ref router) = blk.router {
            total_tokens += 1;

            for (i, &expert_idx) in router.selected_experts.iter().enumerate() {
                let key = (blk.block_idx, expert_idx);
                let acc = expert_map.entry(key).or_default();
                acc.selection_count += 1;
                acc.total_weight += router.expert_weights[i];
            }

            let balance = gini_coefficient(&router.router_probs);
            per_layer_balance.push(balance);

            for ea in &blk.expert_activations {
                let key = (blk.block_idx, ea.expert_idx);
                let acc = expert_map.entry(key).or_default();
                acc.total_gate_l2 += ops::l2_norm(&ea.gate_pre_silu);
                acc.total_up_l2 += ops::l2_norm(&ea.up);
                acc.total_down_l2 += ops::l2_norm(&ea.down);
                acc.activation_count += 1;

                let n_dead = ea.gated.iter().filter(|&&v| v.abs() < 1e-6).count();
                acc.total_dead_neurons += n_dead;
                acc.total_neurons = ea.gated.len();
            }
        }
    }

    let expert_stats: Vec<ExpertStats> = expert_map
        .into_iter()
        .map(|((block_idx, expert_idx), acc)| {
            let n = acc.activation_count.max(1) as f32;
            ExpertStats {
                block_idx,
                expert_idx,
                selection_count: acc.selection_count,
                avg_weight: acc.total_weight / n,
                total_weight: acc.total_weight,
                avg_gate_l2: acc.total_gate_l2 / n,
                avg_up_l2: acc.total_up_l2 / n,
                avg_down_l2: acc.total_down_l2 / n,
                activation_sparsity: if acc.total_neurons > 0 {
                    acc.total_dead_neurons as f32 / (acc.total_neurons as f32 * n)
                } else {
                    0.0
                },
            }
        })
        .collect();

    MoERoutingStats {
        total_tokens,
        expert_stats,
        per_layer_load_balance: per_layer_balance,
    }
}

#[derive(Default)]
struct ExpertAccumulator {
    selection_count: usize,
    total_weight: f32,
    total_gate_l2: f32,
    total_up_l2: f32,
    total_down_l2: f32,
    activation_count: usize,
    total_dead_neurons: usize,
    total_neurons: usize,
}

/// Compute the Gini coefficient of a distribution.
pub fn gini_coefficient(values: &[f32]) -> f32 {
    let n = values.len();
    if n <= 1 {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let sum: f32 = sorted.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }

    let mut gini = 0.0;
    for (i, &v) in sorted.iter().enumerate() {
        gini += (2.0 * (i as f32 + 1.0) - n as f32 - 1.0) * v;
    }

    gini / (n as f32 * sum)
}

/// Get router probability distribution for each MoE block.
pub fn router_distributions(snapshot: &InterpretationSnapshot) -> Vec<RouterDistribution> {
    snapshot
        .blocks
        .iter()
        .filter_map(|blk| {
            blk.router.as_ref().map(|r| RouterDistribution {
                block_idx: blk.block_idx,
                probs: r.router_probs.clone(),
                selected: r.selected_experts.clone(),
                weights: r.expert_weights.clone(),
            })
        })
        .collect()
}

/// Export full interpretability analysis as JSON.
pub fn to_json(snapshot: &InterpretationSnapshot) -> String {
    let lens = snapshot.logit_lens_summary();
    let lens_json: Vec<String> = lens.iter().map(|(blk, topk)| {
        let entries: Vec<String> = topk.iter().map(|(id, v)| {
            format!(r#"{{"token_id":{},"logit":{:.4}}}"#, id, v)
        }).collect();
        format!(r#"  {{"block":{},"top_k":[{}]}}"#, blk, entries.join(","))
    }).collect();

    let moe = expert_utilization(snapshot);
    let expert_json: Vec<String> = moe.expert_stats.iter().map(|e| {
        format!(
            r#"  {{"block":{},"expert":{},"selection_count":{},"avg_weight":{:.6},"total_weight":{:.6},"sparsity":{:.4}}}"#,
            e.block_idx, e.expert_idx, e.selection_count, e.avg_weight, e.total_weight, e.activation_sparsity
        )
    }).collect();

    let balance_json: Vec<String> = moe.per_layer_load_balance.iter().enumerate().map(|(i, &b)| {
        format!(r#"  {{"block":{},"gini":{:.4}}}"#, i, b)
    }).collect();

    format!(
        r#"{{"token_ids":{:?},"logit_lens":[{}],"moe":{{"total_tokens":{},"expert_stats":[{}],"load_balance":[{}]}}}}"#,
        snapshot.token_ids,
        lens_json.join(",\n"),
        moe.total_tokens,
        expert_json.join(",\n"),
        balance_json.join(",\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gini_balanced() {
        let values = vec![0.25, 0.25, 0.25, 0.25];
        let gini = gini_coefficient(&values);
        assert!(gini < 0.01, "perfect balance should have Gini near 0, got {}", gini);
    }

    #[test]
    fn gini_imbalanced() {
        let values = vec![1.0, 0.0, 0.0, 0.0];
        let gini = gini_coefficient(&values);
        assert!(gini > 0.7, "maximum imbalance should have high Gini, got {}", gini);
    }

    #[test]
    fn gini_empty() {
        assert_eq!(gini_coefficient(&[]), 0.0);
        assert_eq!(gini_coefficient(&[0.5]), 0.0);
    }
}
