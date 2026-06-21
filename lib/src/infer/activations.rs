use super::layer::BlockActivations;
use super::model::InferenceModel;
use super::ops;
use crate::error::Result;
use std::path::Path;

/// Full activation snapshot for a forward pass.
#[derive(Debug, Clone)]
pub struct ActivationSnapshot {
    pub token_ids: Vec<u32>,
    pub token_text: Vec<String>,
    pub embedding: Vec<f32>,
    pub blocks: Vec<BlockActivations>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Per-layer summary for display.
#[derive(Debug, Clone)]
pub struct LayerActivations {
    pub block_idx: usize,
    pub attn_norm_l2: f32,
    pub q_l2: f32,
    pub k_l2: f32,
    pub v_l2: f32,
    pub attn_out_l2: f32,
    pub attn_residual_l2: f32,
    pub ffn_norm_l2: f32,
    pub gate_l2: f32,
    pub up_l2: f32,
    pub down_l2: f32,
    pub ffn_residual_l2: f32,
}

impl ActivationSnapshot {
    /// Run a single forward pass and capture all activations.
    ///
    /// `token_ids` is the input token sequence. For demonstration purposes,
    /// this uses a simple embedding lookup (no actual tokenizer needed).
    pub fn run(model: &InferenceModel, token_ids: &[u32]) -> Result<Self> {
        let hd = model.hidden_dim;
        let emb_w = model.read_f32("token_embd.weight")
            .or_else(|_| model.read_f32("tok_embeddings.weight"))?;

        // Embed tokens
        let mut h = vec![0.0f32; hd];
        for &tid in token_ids {
            let offset = tid as usize * hd;
            if offset + hd <= emb_w.len() {
                for d in 0..hd {
                    h[d] += emb_w[offset + d];
                }
            }
        }
        // Average embedding for multi-token input
        if token_ids.len() > 1 {
            let inv = 1.0 / token_ids.len() as f32;
            for v in h.iter_mut() {
                *v *= inv;
            }
        }

        let embedding = h.clone();
        let mut blocks = Vec::with_capacity(model.block_count);

        // Run through transformer blocks
        for blk in 0..model.block_count {
            let (h_next, activations) = super::layer::forward_block(model, blk, &h, 0)?;
            blocks.push(activations);
            h = h_next;
        }

        // Final norm
        let norm_w = model.read_f32("output_norm.weight")
            .or_else(|_| model.read_f32("norm.weight"))?;
        let final_norm = ops::rmsnorm(&h, &norm_w, model.norm_eps);

        // LM head (if present)
        let logits = if model.gguf.get_tensor("output.weight").is_some()
            || model.gguf.get_tensor("lm_head.weight").is_some()
        {
            let head_w = model.read_f32("output.weight")
                .or_else(|_| model.read_f32("lm_head.weight"))?;
            let n_vocab = model.vocab_size;
            ops::matmul(&final_norm, &head_w, 1, hd, n_vocab)
        } else {
            // Tied embedding: use token_embd.weight^T
            let emb = model.read_f32("token_embd.weight")
                .or_else(|_| model.read_f32("tok_embeddings.weight"))?;
            let n_vocab = model.vocab_size;
            ops::matmul(&final_norm, &emb, 1, hd, n_vocab)
        };

        Ok(Self {
            token_ids: token_ids.to_vec(),
            token_text: token_ids.iter().map(|t| format!("<{t}>")).collect(),
            embedding,
            blocks,
            final_norm,
            logits,
        })
    }

    /// Convert to per-layer summaries.
    pub fn layer_summaries(&self) -> Vec<LayerActivations> {
        let mut result = Vec::with_capacity(self.blocks.len());
        let mut h_attn_out = self.embedding.clone();

        for blk in &self.blocks {
            let attn_residual = {
                let mut v = h_attn_out.clone();
                ops::add_inplace(&mut v, &blk.attn_out);
                v
            };
            let ffn_residual = {
                let mut v = attn_residual.clone();
                ops::add_inplace(&mut v, &blk.ffn_down);
                v
            };

            result.push(LayerActivations {
                block_idx: blk.block_idx,
                attn_norm_l2: ops::l2_norm(&blk.attn_norm_in),
                q_l2: ops::l2_norm(&blk.q_proj),
                k_l2: ops::l2_norm(&blk.k_proj),
                v_l2: ops::l2_norm(&blk.v_proj),
                attn_out_l2: ops::l2_norm(&blk.attn_out),
                attn_residual_l2: ops::l2_norm(&attn_residual),
                ffn_norm_l2: ops::l2_norm(&blk.ffn_norm_in),
                gate_l2: ops::l2_norm(&blk.ffn_gate),
                up_l2: ops::l2_norm(&blk.ffn_up),
                down_l2: ops::l2_norm(&blk.ffn_down),
                ffn_residual_l2: ops::l2_norm(&ffn_residual),
            });

            h_attn_out = ffn_residual;
        }

        result
    }

    /// Print a formatted report of activations.
    pub fn print_report(&self) {
        let summaries = self.layer_summaries();

        println!("╔══════════════════════════════════════════════════════════════════════════╗");
        println!("║                    TensorKit Inference Activation Report                 ║");
        println!("╚══════════════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Input tokens: {:?}", self.token_text);
        println!("Token IDs:    {:?}", self.token_ids);
        println!("Embedding L2: {:.6}", ops::l2_norm(&self.embedding));
        println!("Final norm L2: {:.6}", ops::l2_norm(&self.final_norm));
        println!();

        if !self.logits.is_empty() {
            let top_k = 5;
            let mut indexed: Vec<(usize, f32)> = self.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("Top {top_k} logits:");
            for (rank, &(idx, val)) in indexed.iter().take(top_k).enumerate() {
                let rank_num = rank + 1;
                println!("  {rank_num}. token_id={idx:>6}  logit={val:>8.4}");
            }
            println!();
        }

        println!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Block", "Anorm", "Q", "K", "V", "AttnOut", "AttnRes", "Fnorm", "Gate", "Up", "Down", "FfnRes");
        println!("{}", "-".repeat(126));

        for s in &summaries {
            println!("{:<6} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                s.block_idx,
                s.attn_norm_l2,
                s.q_l2,
                s.k_l2,
                s.v_l2,
                s.attn_out_l2,
                s.attn_residual_l2,
                s.ffn_norm_l2,
                s.gate_l2,
                s.up_l2,
                s.down_l2,
                s.ffn_residual_l2);
        }
    }

    /// Export activations as JSON.
    pub fn to_json(&self) -> String {
        let summaries = self.layer_summaries();
        let layers_json: Vec<String> = summaries.iter().map(|s| {
            format!(
                r#"  {{"block":{},"attn_norm_l2":{},"q_l2":{},"k_l2":{},"v_l2":{},"attn_out_l2":{},"attn_residual_l2":{},"ffn_norm_l2":{},"gate_l2":{},"up_l2":{},"down_l2":{},"ffn_residual_l2":{}}}"#,
                s.block_idx, s.attn_norm_l2, s.q_l2, s.k_l2, s.v_l2,
                s.attn_out_l2, s.attn_residual_l2, s.ffn_norm_l2,
                s.gate_l2, s.up_l2, s.down_l2, s.ffn_residual_l2
            )
        }).collect();

        format!(
            r#"{{"token_ids":{:?},"embedding_l2":{},"final_norm_l2":{},"logits_len":{},"layers":[{}]}}"#,
            self.token_ids,
            ops::l2_norm(&self.embedding),
            ops::l2_norm(&self.final_norm),
            self.logits.len(),
            layers_json.join(",\n")
        )
    }
}

/// Run inference on a GGUF model and print activation report.
pub fn run_inference(model_path: &Path, token_ids: &[u32], json: bool) -> Result<()> {
    let model = InferenceModel::open(model_path)?;

    eprintln!("[infer] model: {}", model_path.display());
    eprintln!("[infer] arch: {}, blocks: {}, hidden: {}, heads: {}/{}, ffn: {}",
        model.arch, model.block_count, model.hidden_dim, model.n_heads, model.n_kv_heads, model.ffn_dim);
    eprintln!("[infer] running forward pass for {} token(s)...", token_ids.len());

    let snapshot = ActivationSnapshot::run(&model, token_ids)?;

    if json {
        println!("{}", snapshot.to_json());
    } else {
        snapshot.print_report();
    }

    Ok(())
}
