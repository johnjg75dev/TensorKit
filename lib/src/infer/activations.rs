use super::layer::BlockActivations;
use super::ops;
use super::WeightProvider;
use crate::error::Result;
use std::path::Path;

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
    pub is_moe: bool,
    pub n_active_experts: usize,
}

/// Logit lens entry: what the model would predict at a given layer.
#[derive(Debug, Clone)]
pub struct LogitLensEntry {
    pub block_idx: usize,
    pub logits: Vec<f32>,
    pub top_k: Vec<(u32, f32)>,
}

/// Full activation snapshot for a forward pass (basic).
#[derive(Debug, Clone)]
pub struct ActivationSnapshot {
    pub token_ids: Vec<u32>,
    pub token_text: Vec<String>,
    pub embedding: Vec<f32>,
    pub blocks: Vec<BlockActivations>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

/// Rich activation snapshot for interpretability.
///
/// Includes everything from `ActivationSnapshot` plus:
/// - Logit lens (projection through LM head at each layer)
/// - Full residual stream norms at each layer
/// - Per-head attention details
/// - MoE router decisions and per-expert activations
#[derive(Debug, Clone)]
pub struct InterpretationSnapshot {
    pub token_ids: Vec<u32>,
    pub token_text: Vec<String>,
    pub embedding: Vec<f32>,
    pub blocks: Vec<BlockActivations>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
    pub logit_lens: Vec<LogitLensEntry>,
}

impl ActivationSnapshot {
    /// Run a single forward pass and capture all activations.
    pub fn run(model: &dyn WeightProvider, token_ids: &[u32]) -> Result<Self> {
        let p = model.params();
        let hd = p.hidden_dim;
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
        let mut blocks = Vec::with_capacity(p.block_count);

        // Run through transformer blocks
        for blk in 0..p.block_count {
            let (h_next, activations) = super::layer::forward_block(model, blk, &h, 0, None)?;
            blocks.push(activations);
            h = h_next;
        }

        // Final norm
        let norm_w = model.read_f32("output_norm.weight")
            .or_else(|_| model.read_f32("norm.weight"))?;
        let final_norm = ops::rmsnorm(&h, &norm_w, p.norm_eps);

        // LM head
        let logits = if model.read_f32("output.weight").is_ok()
            || model.read_f32("lm_head.weight").is_ok()
        {
            let head_w = model.read_f32("output.weight")
                .or_else(|_| model.read_f32("lm_head.weight"))?;
            let n_vocab = p.vocab_size;
            ops::matmul(&final_norm, &head_w, 1, hd, n_vocab)
        } else {
            // Tied embedding
            let emb = model.read_f32("token_embd.weight")
                .or_else(|_| model.read_f32("tok_embeddings.weight"))?;
            let n_vocab = p.vocab_size;
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
                ops::add_inplace(&mut v, &blk.attn_out_post_proj);
                v
            };
            let ffn_residual = {
                let mut v = attn_residual.clone();
                ops::add_inplace(&mut v, &blk.ffn_down);
                v
            };

            result.push(LayerActivations {
                block_idx: blk.block_idx,
                attn_norm_l2: ops::l2_norm(&blk.attn_norm_out),
                q_l2: ops::l2_norm(&blk.q_post_rope),
                k_l2: ops::l2_norm(&blk.k_post_rope),
                v_l2: ops::l2_norm(&blk.v_proj),
                attn_out_l2: ops::l2_norm(&blk.attn_out_post_proj),
                attn_residual_l2: ops::l2_norm(&attn_residual),
                ffn_norm_l2: ops::l2_norm(&blk.ffn_norm_out),
                gate_l2: if blk.gate_pre_silu.is_empty() { 0.0 } else { ops::l2_norm(&blk.gate_pre_silu) },
                up_l2: if blk.ffn_up.is_empty() { 0.0 } else { ops::l2_norm(&blk.ffn_up) },
                down_l2: ops::l2_norm(&blk.ffn_down),
                ffn_residual_l2: ops::l2_norm(&ffn_residual),
                is_moe: blk.router.is_some(),
                n_active_experts: blk.expert_activations.len(),
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
            let moe_marker = if s.is_moe { "*" } else { " " };
            println!("{:<5}{} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                s.block_idx,
                moe_marker,
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

        // Print MoE expert utilization summary
        let moe_blocks: Vec<&LayerActivations> = summaries.iter().filter(|s| s.is_moe).collect();
        if !moe_blocks.is_empty() {
            println!();
            println!("MoE Expert Utilization:");
            for s in &moe_blocks {
                println!("  Block {}: {} active experts", s.block_idx, s.n_active_experts);
            }
        }
    }

    /// Export activations as JSON.
    pub fn to_json(&self) -> String {
        let summaries = self.layer_summaries();
        let layers_json: Vec<String> = summaries.iter().map(|s| {
            format!(
                r#"  {{"block":{},"attn_norm_l2":{},"q_l2":{},"k_l2":{},"v_l2":{},"attn_out_l2":{},"attn_residual_l2":{},"ffn_norm_l2":{},"gate_l2":{},"up_l2":{},"down_l2":{},"ffn_residual_l2":{},"is_moe":{},"n_active_experts":{}}}"#,
                s.block_idx, s.attn_norm_l2, s.q_l2, s.k_l2, s.v_l2,
                s.attn_out_l2, s.attn_residual_l2, s.ffn_norm_l2,
                s.gate_l2, s.up_l2, s.down_l2, s.ffn_residual_l2,
                s.is_moe, s.n_active_experts
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

impl InterpretationSnapshot {
    /// Run a full forward pass with logit lens and all intermediates.
    pub fn run(model: &dyn WeightProvider, token_ids: &[u32]) -> Result<Self> {
        let p = model.params();
        let hd = p.hidden_dim;
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
        if token_ids.len() > 1 {
            let inv = 1.0 / token_ids.len() as f32;
            for v in h.iter_mut() {
                *v *= inv;
            }
        }

        let embedding = h.clone();
        let mut blocks = Vec::with_capacity(p.block_count);
        let mut logit_lens = Vec::with_capacity(p.block_count + 1);

        // Read LM head weights once (reused at each layer for logit lens)
        let lm_head_w: Option<Vec<f32>> =
            model.read_f32("output.weight")
                .or_else(|_| model.read_f32("lm_head.weight"))
                .ok();

        // Run through transformer blocks with logit lens
        for blk in 0..p.block_count {
            let (h_next, activations) = super::layer::forward_block(model, blk, &h, 0, None)?;

            // Logit lens: project current hidden state through LM head
            if let Some(ref head_w) = lm_head_w {
                let logits = ops::matmul(&h_next, head_w, 1, hd, p.vocab_size);
                let top_k = top_k_from_logits(&logits, 5);
                logit_lens.push(LogitLensEntry {
                    block_idx: blk,
                    logits,
                    top_k,
                });
            }

            blocks.push(activations);
            h = h_next;
        }

        // Final norm
        let norm_w = model.read_f32("output_norm.weight")
            .or_else(|_| model.read_f32("norm.weight"))?;
        let final_norm = ops::rmsnorm(&h, &norm_w, p.norm_eps);

        // Final logits
        let logits = if let Some(ref head_w) = lm_head_w {
            ops::matmul(&final_norm, head_w, 1, hd, p.vocab_size)
        } else {
            let emb = model.read_f32("token_embd.weight")
                .or_else(|_| model.read_f32("tok_embeddings.weight"))?;
            ops::matmul(&final_norm, &emb, 1, hd, p.vocab_size)
        };
        let top_k = top_k_from_logits(&logits, 5);
        logit_lens.push(LogitLensEntry {
            block_idx: p.block_count,
            logits: logits.clone(),
            top_k,
        });

        Ok(Self {
            token_ids: token_ids.to_vec(),
            token_text: token_ids.iter().map(|t| format!("<{t}>")).collect(),
            embedding,
            blocks,
            final_norm,
            logits,
            logit_lens,
        })
    }

    /// Get logit lens top-k predictions at each layer.
    pub fn logit_lens_summary(&self) -> Vec<(usize, Vec<(u32, f32)>)> {
        self.logit_lens
            .iter()
            .map(|e| (e.block_idx, e.top_k.clone()))
            .collect()
    }

    /// Convert to per-layer summaries (delegates to `ActivationSnapshot` logic).
    pub fn layer_summaries(&self) -> Vec<LayerActivations> {
        let tmp = ActivationSnapshot {
            token_ids: self.token_ids.clone(),
            token_text: self.token_text.clone(),
            embedding: self.embedding.clone(),
            blocks: self.blocks.clone(),
            final_norm: self.final_norm.clone(),
            logits: self.logits.clone(),
        };
        tmp.layer_summaries()
    }

    /// Export as JSON with logit lens data.
    pub fn to_json(&self) -> String {
        let summaries = self.layer_summaries();
        let layers_json: Vec<String> = summaries.iter().map(|s| {
            format!(
                r#"  {{"block":{},"attn_norm_l2":{},"q_l2":{},"k_l2":{},"v_l2":{},"attn_out_l2":{},"attn_residual_l2":{},"ffn_norm_l2":{},"gate_l2":{},"up_l2":{},"down_l2":{},"ffn_residual_l2":{},"is_moe":{},"n_active_experts":{}}}"#,
                s.block_idx, s.attn_norm_l2, s.q_l2, s.k_l2, s.v_l2,
                s.attn_out_l2, s.attn_residual_l2, s.ffn_norm_l2,
                s.gate_l2, s.up_l2, s.down_l2, s.ffn_residual_l2,
                s.is_moe, s.n_active_experts
            )
        }).collect();

        let lens_json: Vec<String> = self.logit_lens.iter().map(|e| {
            let top: Vec<String> = e.top_k.iter().map(|(id, v)| {
                format!(r#"{{"token_id":{},"logit":{:.4}}}"#, id, v)
            }).collect();
            format!(
                r#"  {{"block":{},"top_k":[{}]}}"#,
                e.block_idx,
                top.join(",")
            )
        }).collect();

        format!(
            r#"{{"token_ids":{:?},"embedding_l2":{},"final_norm_l2":{},"logits_len":{},"logit_lens":[{}],"layers":[{}]}}"#,
            self.token_ids,
            ops::l2_norm(&self.embedding),
            ops::l2_norm(&self.final_norm),
            self.logits.len(),
            lens_json.join(",\n"),
            layers_json.join(",\n")
        )
    }
}

/// Run inference on any supported model format.
pub fn run_inference(model_path: &Path, token_ids: &[u32], json: bool) -> Result<()> {
    let model = super::format::open_infer(model_path)?;

    let p = model.params();
    eprintln!("[infer] model: {}", model_path.display());
    eprintln!("[infer] arch: {}, blocks: {}, hidden: {}, heads: {}/{}, ffn: {}, experts: {}/{}",
        p.arch, p.block_count, p.hidden_dim, p.n_heads, p.n_kv_heads, p.ffn_dim, p.n_experts, p.n_experts_per_tok);

    if p.n_experts > 0 {
        eprintln!("[infer] MoE model detected with {} experts, routing top-{} per token",
            p.n_experts, p.n_experts_per_tok);
    }

    eprintln!("[infer] running forward pass for {} token(s)...", token_ids.len());

    let snapshot = ActivationSnapshot::run(model.as_ref(), token_ids)?;

    if json {
        println!("{}", snapshot.to_json());
    } else {
        snapshot.print_report();
    }

    Ok(())
}

/// Run full interpretability analysis.
pub fn run_interpret(model_path: &Path, token_ids: &[u32]) -> Result<()> {
    let model = super::format::open_infer(model_path)?;

    let p = model.params();
    eprintln!("[interpret] model: {}", model_path.display());
    eprintln!("[interpret] arch: {}, blocks: {}, experts: {}/{}",
        p.arch, p.block_count, p.n_experts, p.n_experts_per_tok);

    let snapshot = InterpretationSnapshot::run(model.as_ref(), token_ids)?;

    println!("{}", snapshot.to_json());

    Ok(())
}

/// Extract top-k token predictions from raw logits.
fn top_k_from_logits(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).collect()
}
