use super::model::InferenceModel;
use super::ops;
use crate::error::Result;

/// Activations captured from a single transformer block.
#[derive(Debug, Clone)]
pub struct BlockActivations {
    pub block_idx: usize,
    pub attn_norm_in: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_norm_in: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
}

/// Run one transformer block and return intermediate activations.
pub fn forward_block(
    model: &InferenceModel,
    block_idx: usize,
    h: &[f32],       // hidden_dim
    pos: usize,
) -> Result<(Vec<f32>, BlockActivations)> {
    let hd = model.hidden_dim;
    let nh = model.n_heads;
    let nkv = model.n_kv_heads;
    let hd_head = model.head_dim;

    // --- Attention pre-norm ---
    let attn_norm_w = model.block_f32(block_idx, "attn_norm.weight")?;
    let h_normed = ops::rmsnorm(h, &attn_norm_w, model.norm_eps);

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
    let _seq_len = 1; // single-token forward pass
    let mut attn_out = vec![0.0f32; hd];

    // Compute attention scores and output for each query head
    for h_idx in 0..nh {
        let kv_head = h_idx / n_rep;
        let q_head_base = h_idx * hd_head;
        let k_head_base = kv_head * hd_head;

        // score = q @ k^T (single token, so just dot product)
        let mut score = 0.0f32;
        for d in 0..hd_head {
            score += q_rope[q_head_base + d] * k_rope[k_head_base + d];
        }
        score /= (hd_head as f32).sqrt();
        let weight = score.exp(); // softmax over 1 token is just exp (normalized later)

        // output += weight * v
        for d in 0..hd_head {
            attn_out[h_idx * hd_head + d] += weight * v[k_head_base + d];
        }
    }
    // Normalize softmax (single token, weight sums to exp(score), divide by sum)
    let mut total = 0.0f32;
    for h_idx in 0..nh {
        let kv_head = h_idx / n_rep;
        let q_head_base = h_idx * hd_head;
        let k_head_base = kv_head * hd_head;
        let mut score = 0.0f32;
        for d in 0..hd_head {
            score += q_rope[q_head_base + d] * k_rope[k_head_base + d];
        }
        total += (score / (hd_head as f32).sqrt()).exp();
    }
    let inv_total = 1.0 / total;
    for v in attn_out.iter_mut() {
        *v *= inv_total;
    }

    // --- Output projection ---
    let o_w = model.block_f32(block_idx, "attn_output.weight")?;
    let attn_proj = ops::matmul(&attn_out, &o_w, 1, hd, hd);

    // --- Residual ---
    let mut h_res1 = h.to_vec();
    ops::add_inplace(&mut h_res1, &attn_proj);

    // --- FFN pre-norm ---
    let ffn_norm_w = model.block_f32(block_idx, "ffn_norm.weight")?;
    let h_ffn_normed = ops::rmsnorm(&h_res1, &ffn_norm_w, model.norm_eps);

    // --- SwiGLU FFN ---
    let gate_w = model.block_f32(block_idx, "ffn_gate.weight")?;
    let up_w = model.block_f32(block_idx, "ffn_up.weight")?;
    let mut gate = ops::matmul(&h_ffn_normed, &gate_w, 1, hd, model.ffn_dim);
    let up = ops::matmul(&h_ffn_normed, &up_w, 1, hd, model.ffn_dim);

    // gate = silu(gate)
    ops::silu(&mut gate);

    // element-wise gate * up
    let mut gated = gate.clone();
    for (g, u) in gated.iter_mut().zip(up.iter()) {
        *g *= *u;
    }

    // --- Down projection ---
    let down_w = model.block_f32(block_idx, "ffn_down.weight")?;
    let ffn_out = ops::matmul(&gated, &down_w, 1, model.ffn_dim, hd);

    // --- Residual ---
    let mut h_out = h_res1.clone();
    ops::add_inplace(&mut h_out, &ffn_out);

    let activations = BlockActivations {
        block_idx,
        attn_norm_in: h_normed,
        q_proj: q,
        k_proj: k,
        v_proj: v,
        attn_out: attn_proj,
        ffn_norm_in: h_ffn_normed,
        ffn_gate: gated,
        ffn_up: up,
        ffn_down: ffn_out,
    };

    Ok((h_out, activations))
}
