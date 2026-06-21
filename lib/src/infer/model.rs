use crate::error::{Error, Result};
use crate::formats::gguf::dequant::dequantize;
use crate::formats::gguf::GgufFile;
use std::path::Path;

/// Loaded GGUF model ready for inference.
pub struct InferenceModel {
    pub gguf: GgufFile,
    pub arch: String,
    pub block_count: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
}

impl InferenceModel {
    /// Open a GGUF file and parse model hyperparameters from metadata.
    pub fn open(path: &Path) -> Result<Self> {
        let gguf = GgufFile::open(path)?;

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

        Ok(Self {
            gguf,
            arch,
            block_count,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            vocab_size,
            norm_eps,
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

    /// Read a per-block tensor as f32. E.g. `block_f32(0, "attn_q.weight")`.
    pub fn block_f32(&self, block: usize, suffix: &str) -> Result<Vec<f32>> {
        self.read_f32(&format!("blk.{block}.{suffix}"))
    }
}
