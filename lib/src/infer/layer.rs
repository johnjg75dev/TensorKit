use super::kv_cache::KvCache;
use super::ops;
use super::WeightProvider;
use crate::error::Result;

/// Activations from a single attention head.
#[derive(Debug, Clone)]
pub struct HeadActivations {
    pub head_idx: usize,
    pub kv_head_idx: usize,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub score: f32,
    pub weight: f32,
    pub output: Vec<f32>,
}

/// Activations from a single MoE expert.
#[derive(Debug, Clone)]
pub struct ExpertActivations {
    pub expert_idx: usize,
    pub gate_pre_silu: Vec<f32>,
    pub gate_silu: Vec<f32>,
    pub up: Vec<f32>,
    pub gated: Vec<f32>,
    pub down: Vec<f32>,
}

/// Router decision for a single block (MoE).
#[derive(Debug, Clone)]
pub struct RouterActivations {
    pub router_logits: Vec<f32>,
    pub router_probs: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub expert_weights: Vec<f32>,
}

/// Detailed activations captured from a single transformer block.
#[derive(Debug, Clone)]
pub struct BlockActivations {
    pub block_idx: usize,

    // --- Residual streams ---
    pub hidden_input: Vec<f32>,
    pub attn_norm_out: Vec<f32>,
    pub attn_residual: Vec<f32>,
    pub ffn_norm_out: Vec<f32>,
    pub hidden_output: Vec<f32>,

    // --- Attention ---
    pub q_pre_rope: Vec<f32>,
    pub k_pre_rope: Vec<f32>,
    pub q_post_rope: Vec<f32>,
    pub k_post_rope: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub attn_weights: Vec<f32>,
    pub attn_out_pre_proj: Vec<f32>,
    pub attn_out_post_proj: Vec<f32>,
    pub head_activations: Vec<HeadActivations>,

    // --- Dense FFN ---
    pub gate_pre_silu: Vec<f32>,
    pub gate_silu: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_gated: Vec<f32>,
    pub ffn_down: Vec<f32>,

    // --- MoE (empty vecs if not MoE) ---
    pub router: Option<RouterActivations>,
    pub expert_activations: Vec<ExpertActivations>,
}

/// Run one transformer block and return detailed activations.
///
/// Supports both dense and MoE blocks. For MoE, computes the router
/// decision and runs the selected experts.
pub fn forward_block(
    model: &dyn WeightProvider,
    block_idx: usize,
    h: &[f32],
    pos: usize,
    mut cache: Option<&mut KvCache>,
) -> Result<(Vec<f32>, BlockActivations)> {
    let p = model.params();
    let hd = p.hidden_dim;
    let nh = p.n_heads;
    let nkv = p.n_kv_heads;
    let hd_head = p.head_dim;

    // --- Attention pre-norm ---
    let attn_norm_w = model.block_f32(block_idx, "attn_norm.weight")?;
    let h_normed = ops::rmsnorm(h, &attn_norm_w, p.norm_eps);

    // --- Q, K, V projections ---
    let q_w = model.block_f32(block_idx, "attn_q.weight")?;
    let k_w = model.block_f32(block_idx, "attn_k.weight")?;
    let v_w = model.block_f32(block_idx, "attn_v.weight")?;
    let q = ops::matmul(&h_normed, &q_w, 1, hd, hd);
    let k = ops::matmul(&h_normed, &k_w, 1, hd, nkv * hd_head);
    let v = ops::matmul(&h_normed, &v_w, 1, hd, nkv * hd_head);

    // --- RoPE ---
    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    ops::rope_inplace(&mut q_rope, &mut k_rope, pos, hd_head);

    // --- Grouped-query attention ---
    let n_rep = nh / nkv;

    // Append to cache if provided
    if let Some(ref mut cache) = cache {
        cache.append(block_idx, &k_rope, &v);
    }

    let mut head_activations = Vec::with_capacity(nh);
    let mut all_scores = Vec::with_capacity(nh);
    let mut all_weights = Vec::with_capacity(nh);

    // Compute attention scores and output for each query head
    for h_idx in 0..nh {
        let kv_head = h_idx / n_rep;
        let q_base = h_idx * hd_head;
        let k_base = kv_head * hd_head;

        // dot product (current token with itself)
        let mut score = 0.0f32;
        for d in 0..hd_head {
            score += q_rope[q_base + d] * k_rope[k_base + d];
        }
        score /= (hd_head as f32).sqrt();
        let weight = score.exp(); // softmax over 1 token

        let mut h_output = vec![0.0f32; hd_head];
        for d in 0..hd_head {
            h_output[d] = weight * v[kv_head * hd_head + d];
        }

        head_activations.push(HeadActivations {
            head_idx: h_idx,
            kv_head_idx: kv_head,
            q: q[q_base..q_base + hd_head].to_vec(),
            k: k[k_base..k_base + hd_head].to_vec(),
            v: v[kv_head * hd_head..(kv_head + 1) * hd_head].to_vec(),
            score,
            weight,
            output: h_output,
        });

        all_scores.push(score);
        all_weights.push(weight);
    }

    // Normalize softmax
    let total: f32 = all_weights.iter().sum();
    let inv_total = 1.0 / total;
    for w in all_weights.iter_mut() {
        *w *= inv_total;
    }

    // Reconstruct attention output (correctly weighted)
    let mut attn_out_pre_proj = vec![0.0f32; hd];
    for h_idx in 0..nh {
        let kv_head = h_idx / n_rep;
        let o_base = h_idx * hd_head;
        for d in 0..hd_head {
            attn_out_pre_proj[o_base + d] = all_weights[h_idx] * v[kv_head * hd_head + d];
        }
    }

    // --- Output projection ---
    let o_w = model.block_f32(block_idx, "attn_output.weight")?;
    let attn_proj = ops::matmul(&attn_out_pre_proj, &o_w, 1, hd, hd);

    // --- Residual ---
    let mut h_res1 = h.to_vec();
    ops::add_inplace(&mut h_res1, &attn_proj);

    // --- FFN pre-norm ---
    let ffn_norm_w = model.block_f32(block_idx, "ffn_norm.weight")?;
    let h_ffn_normed = ops::rmsnorm(&h_res1, &ffn_norm_w, p.norm_eps);

    // --- FFN (dense or MoE) ---
    let (ffn_gate_pre, ffn_gate_silu, ffn_up_vec, ffn_gated_vec, ffn_down_vec, router_opt, expert_acts) =
        if model.is_moe_block(block_idx) {
            forward_moe_block(model, block_idx, &h_ffn_normed, hd)?
        } else {
            forward_dense_ffn(model, block_idx, &h_ffn_normed, hd)?
        };

    // --- Residual ---
    let mut h_out = h_res1.clone();
    ops::add_inplace(&mut h_out, &ffn_down_vec);

    let activations = BlockActivations {
        block_idx,
        hidden_input: h.to_vec(),
        attn_norm_out: h_normed,
        attn_residual: h_res1,
        ffn_norm_out: h_ffn_normed,
        hidden_output: h_out.clone(),
        q_pre_rope: q,
        k_pre_rope: k,
        q_post_rope: q_rope,
        k_post_rope: k_rope,
        v_proj: v,
        attn_scores: all_scores,
        attn_weights: all_weights,
        attn_out_pre_proj,
        attn_out_post_proj: attn_proj,
        head_activations,
        gate_pre_silu: ffn_gate_pre,
        gate_silu: ffn_gate_silu,
        ffn_up: ffn_up_vec,
        ffn_gated: ffn_gated_vec,
        ffn_down: ffn_down_vec,
        router: router_opt,
        expert_activations: expert_acts,
    };

    Ok((h_out, activations))
}

/// Dense FFN forward pass (non-MoE).
fn forward_dense_ffn(
    model: &dyn WeightProvider,
    block_idx: usize,
    h_normed: &[f32],
    hd: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Option<RouterActivations>, Vec<ExpertActivations>)> {
    let p = model.params();
    let gate_w = model.block_f32(block_idx, "ffn_gate.weight")?;
    let up_w = model.block_f32(block_idx, "ffn_up.weight")?;
    let down_w = model.block_f32(block_idx, "ffn_down.weight")?;

    let gate = ops::matmul(h_normed, &gate_w, 1, hd, p.ffn_dim);
    let up = ops::matmul(h_normed, &up_w, 1, hd, p.ffn_dim);

    let gate_silu = gate.iter().map(|&x| ops::silu_one(x)).collect::<Vec<_>>();
    let mut gated = gate_silu.clone();
    for (g, u) in gated.iter_mut().zip(up.iter()) {
        *g *= *u;
    }

    let ffn_out = ops::matmul(&gated, &down_w, 1, p.ffn_dim, hd);

    Ok((gate, gate_silu, up, gated, ffn_out, None, vec![]))
}

/// MoE forward pass: router + selected experts.
fn forward_moe_block(
    model: &dyn WeightProvider,
    block_idx: usize,
    h_normed: &[f32],
    hd: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Option<RouterActivations>, Vec<ExpertActivations>)> {
    let p = model.params();
    let n_experts = p.n_experts;
    let n_experts_per_tok = p.n_experts_per_tok;

    // --- Router gate ---
    let router_w = model.block_router_weight(block_idx)
        .ok_or_else(|| crate::error::Error::Infer(format!(
            "block {block_idx} marked as MoE but no router weight found"
        )))??;

    // router logits = h_normed @ router_w^T
    // router_w is [hidden_dim, n_experts] → router_out is [n_experts]
    let router_logits = ops::matmul(h_normed, &router_w, 1, hd, n_experts);

    // Softmax over experts
    let mut router_probs = router_logits.clone();
    let max_logit = router_probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for v in router_probs.iter_mut() {
        *v = (*v - max_logit).exp();
        sum_exp += *v;
    }
    for v in router_probs.iter_mut() {
        *v /= sum_exp;
    }

    // Select top-k experts
    let mut indexed: Vec<(usize, f32)> = router_probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let selected: Vec<usize> = indexed.iter().take(n_experts_per_tok).map(|&(i, _)| i).collect();
    let selected_weights: Vec<f32> = indexed.iter().take(n_experts_per_tok).map(|&(_, v)| v).collect();

    // Renormalize selected weights
    let w_sum: f32 = selected_weights.iter().sum();
    let renorm_w: Vec<f32> = selected_weights.iter().map(|&w| w / w_sum).collect();

    // --- Run selected experts ---
    let mut expert_acts = Vec::with_capacity(n_experts_per_tok);
    let mut ffn_out = vec![0.0f32; hd];

    for (i, &expert_idx) in selected.iter().enumerate() {
        let gate_w = model.block_expert_weights(block_idx, expert_idx, "ffn_gate.weight")?;
        let up_w = model.block_expert_weights(block_idx, expert_idx, "ffn_up.weight")?;
        let down_w = model.block_expert_weights(block_idx, expert_idx, "ffn_down.weight")?;

        let gate = ops::matmul(h_normed, &gate_w, 1, hd, p.ffn_dim);
        let up = ops::matmul(h_normed, &up_w, 1, hd, p.ffn_dim);

        let gate_silu = gate.iter().map(|&x| ops::silu_one(x)).collect::<Vec<_>>();
        let mut gated = gate_silu.clone();
        for (g, u) in gated.iter_mut().zip(up.iter()) {
            *g *= *u;
        }

        let expert_out = ops::matmul(&gated, &down_w, 1, p.ffn_dim, hd);

        // Weighted contribution to final output
        for (j, val) in expert_out.iter().enumerate() {
            ffn_out[j] += renorm_w[i] * val;
        }

        expert_acts.push(ExpertActivations {
            expert_idx,
            gate_pre_silu: gate,
            gate_silu,
            up,
            gated,
            down: expert_out,
        });
    }

    let router = RouterActivations {
        router_logits,
        router_probs,
        selected_experts: selected,
        expert_weights: renorm_w,
    };

    // For MoE, gate/up/down represent the combined expert output
    Ok((vec![], vec![], vec![], vec![], ffn_out, Some(router), expert_acts))
}
