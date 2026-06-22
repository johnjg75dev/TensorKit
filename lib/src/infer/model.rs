use crate::error::{Error, Result};
use crate::formats::gguf::dequant::dequantize;
use crate::formats::gguf::GgufFile;
use std::path::Path;

use super::WeightProvider;

/// Model hyperparameters, format-agnostic.
#[derive(Debug, Clone)]
pub struct ModelHyperparams {
    pub arch: String,
    pub block_count: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub n_experts: usize,
    pub n_experts_per_tok: usize,
}

/// Loaded GGUF model ready for inference.
pub struct InferenceModel {
    pub gguf: GgufFile,
    pub params: ModelHyperparams,
}

impl InferenceModel {
    /// Open a GGUF file and parse model hyperparameters from metadata.
    pub fn open(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let params = Self::parse_hyperparams(&gguf)?;
        Ok(Self { gguf, params })
    }

    fn parse_hyperparams(gguf: &GgufFile) -> Result<ModelHyperparams> {
        let arch = gguf
            .metadata_str("general.architecture")
            .ok_or_else(|| Error::Infer("missing general.architecture".into()))?
            .to_string();

        let block_count = gguf
            .metadata_u32(&format!("{arch}.block_count"))
            .ok_or_else(|| Error::Infer(format!("missing {arch}.block_count")))?
            as usize;

        // Infer hidden_dim from token_embd.weight shape: [vocab_size, hidden_dim]
        let hidden_dim = gguf
            .get_tensor("token_embd.weight")
            .or_else(|| gguf.get_tensor("tok_embeddings.weight"))
            .ok_or_else(|| Error::Infer("missing token embedding tensor".into()))?
            .dims
            .get(1)
            .copied()
            .ok_or_else(|| Error::Infer("token_embd.weight has no dim 1".into()))?
            as usize;

        let n_heads = gguf
            .metadata_u32(&format!("{arch}.attention.head_count"))
            .ok_or_else(|| Error::Infer(format!("missing {arch}.attention.head_count")))?
            as usize;

        let n_kv_heads = gguf
            .metadata_u32(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32)
            as usize;

        let head_dim = hidden_dim / n_heads;

        let ffn_dim = gguf
            .metadata_u32(&format!("{arch}.feed_forward_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                // Infer from ffn_up.weight shape: [ffn_dim, hidden_dim]
                let name = "blk.0.ffn_up.weight";
                gguf.get_tensor(name)
                    .and_then(|t| t.dims.first().copied())
                    .map(|v| v as usize)
                    .unwrap_or(hidden_dim * 4)
            });

        let vocab_size = gguf
            .get_tensor("token_embd.weight")
            .or_else(|| gguf.get_tensor("tok_embeddings.weight"))
            .and_then(|t| t.dims.first().copied())
            .map(|v| v as usize)
            .unwrap_or(32000);

        let norm_eps = gguf
            .metadata_str(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1e-5) as f32;

        // MoE detection: check for expert-related metadata or tensors
        let n_experts = gguf
            .metadata_u32(&format!("{arch}.expert_count"))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                // Check if blk.0.experts.0.ffn_gate.weight exists → MoE
                if gguf.get_tensor("blk.0.experts.0.ffn_gate.weight").is_some() {
                    // Count experts by probing until we find one that doesn't exist
                    let mut count = 0;
                    for e in 0..256 {
                        if gguf.get_tensor(&format!("blk.0.experts.{e}.ffn_gate.weight")).is_some() {
                            count = e + 1;
                        } else {
                            break;
                        }
                    }
                    count
                } else {
                    0
                }
            });

        let n_experts_per_tok = if n_experts > 0 {
            gguf.metadata_u32(&format!("{arch}.expert_used_count"))
                .unwrap_or(2) as usize
        } else {
            0
        };

        Ok(ModelHyperparams {
            arch,
            block_count,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            vocab_size,
            norm_eps,
            n_experts,
            n_experts_per_tok,
        })
    }

    /// Dequantize a tensor by name to f32.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let ti = self
            .gguf
            .get_tensor(name)
            .ok_or_else(|| Error::Infer(format!("tensor not found: {name}")))?;
        let raw = self
            .gguf
            .tensor_slice(ti)
            .ok_or_else(|| Error::Infer(format!("no data for tensor: {name}")))?;
        dequantize(ti.ggml_type, raw, None)
            .ok_or_else(|| Error::Infer(format!("dequantize failed for: {name}")))
    }
}

impl WeightProvider for InferenceModel {
    fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        InferenceModel::read_f32(self, name)
    }

    fn params(&self) -> &ModelHyperparams {
        &self.params
    }

    fn n_experts(&self) -> usize {
        self.params.n_experts
    }

    fn n_experts_per_tok(&self) -> usize {
        self.params.n_experts_per_tok
    }

    fn is_moe_block(&self, block: usize) -> bool {
        self.params.n_experts > 0
            && self
                .gguf
                .get_tensor(&format!("blk.{block}.experts.0.ffn_gate.weight"))
                .is_some()
    }

    fn block_router_weight(&self, block: usize) -> Option<Result<Vec<f32>>> {
        // GGUF MoE: router weight is at blk.N.experts_weights.weight
        let name = format!("blk.{block}.experts_weights.weight");
        if self.gguf.get_tensor(&name).is_some() {
            Some(self.read_f32(&name))
        } else {
            None
        }
    }

    fn block_expert_weights(
        &self,
        block: usize,
        expert_idx: usize,
        suffix: &str,
    ) -> Result<Vec<f32>> {
        self.read_f32(&format!("blk.{block}.experts.{expert_idx}.{suffix}"))
    }
}
