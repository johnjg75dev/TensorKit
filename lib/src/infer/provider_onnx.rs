//! ONNX inference provider.
//!
//! Loads an ONNX model, reads initializer weights as f32, and implements
//! `WeightProvider` for the inference engine.

use crate::error::{Error, Result};
use crate::formats::onnx::OnnxFile;
use crate::model::Model;
use std::collections::HashMap;
use std::path::Path;

use super::config::{hf_config_to_hyperparams, read_hf_config};
use super::naming;
use super::WeightProvider;
use super::ModelHyperparams;

/// ONNX inference provider.
pub struct OnnxProvider {
    file: OnnxFile,
    params: ModelHyperparams,
    name_map: HashMap<String, String>,
    tensor_names: Vec<String>,
}

impl OnnxProvider {
    /// Open an ONNX file for inference.
    pub fn open(path: &Path) -> Result<Self> {
        let file = OnnxFile::open(path)?;
        let tensor_names: Vec<String> = file.tensor_names().map(String::from).collect();
        let name_map = naming::build_name_map(&tensor_names);

        // Try config.json from same directory
        let hf_config = path.parent().and_then(read_hf_config);

        let params = if let Some(config) = &hf_config {
            hf_config_to_hyperparams(config, None, 0).unwrap_or_else(|| {
                infer_params_from_onnx(&file, &tensor_names)
            })
        } else {
            infer_params_from_onnx(&file, &tensor_names)
        };

        Ok(Self {
            file,
            params,
            name_map,
            tensor_names,
        })
    }

    /// Resolve a canonical name to the actual tensor name in the model.
    fn resolve(&self, canonical: &str) -> Option<String> {
        // Check name map
        if let Some(mapped) = self.name_map.get(canonical) {
            if self.tensor_names.iter().any(|n| n == mapped) {
                return Some(mapped.clone());
            }
        }

        // Exact match
        if self.tensor_names.iter().any(|n| n == canonical) {
            return Some(canonical.to_string());
        }

        // Try ONNX-style variants
        for variant in naming::onnx_variants(canonical) {
            if self.tensor_names.iter().any(|n| n == &variant) {
                return Some(variant);
            }
        }

        None
    }

    /// Read raw bytes and convert to f32.
    fn read_raw_f32(&self, name: &str) -> Result<Vec<f32>> {
        let raw = Model::read_tensor_bytes(&self.file, name)
            .map_err(|e| Error::Infer(format!("read failed for '{name}': {e}")))?;

        let tp = self
            .file
            .tensor_proto(name)
            .ok_or_else(|| Error::Infer(format!("tensor proto not found: {name}")))?;

        match tp.data_type {
            // FLOAT (1) → raw_data is f32 bytes
            1 => {
                if !tp.raw_data.is_empty() {
                    let mut out = Vec::with_capacity(tp.raw_data.len() / 4);
                    for chunk in tp.raw_data.chunks_exact(4) {
                        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                    Ok(out)
                } else {
                    Ok(tp.float_data.clone())
                }
            }
            // FLOAT16 (10) → convert
            10 => {
                let mut out = Vec::with_capacity(raw.len() / 2);
                for chunk in raw.chunks_exact(2) {
                    let u16_val = u16::from_le_bytes(chunk.try_into().unwrap());
                    out.push(crate::formats::gguf::dequant::f16_to_f32(u16_val));
                }
                Ok(out)
            }
            // BFLOAT16 (16) → convert
            16 => {
                let mut out = Vec::with_capacity(raw.len() / 2);
                for chunk in raw.chunks_exact(2) {
                    let u16_val = u16::from_le_bytes(chunk.try_into().unwrap());
                    out.push(crate::formats::gguf::dequant::bf16_to_f32(u16_val));
                }
                Ok(out)
            }
            // DOUBLE (11) → f64 to f32
            11 => {
                if !tp.raw_data.is_empty() {
                    let mut out = Vec::with_capacity(tp.raw_data.len() / 8);
                    for chunk in tp.raw_data.chunks_exact(8) {
                        out.push(f64::from_le_bytes(chunk.try_into().unwrap()) as f32);
                    }
                    Ok(out)
                } else {
                    Ok(tp.double_data.iter().map(|&v| v as f32).collect())
                }
            }
            _ => Err(Error::Infer(format!(
                "unsupported ONNX dtype {} for tensor '{name}'",
                tp.data_type
            ))),
        }
    }
}

impl WeightProvider for OnnxProvider {
    fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let resolved = self
            .resolve(name)
            .ok_or_else(|| Error::Infer(format!("tensor not found: {name}")))?;
        self.read_raw_f32(&resolved)
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
            && (self.resolve(&format!("blk.{block}.experts.0.ffn_gate.weight")).is_some()
                || self
                    .resolve(&format!(
                        "model.layers.{block}.block_sparse_moe.experts.0.gate_proj.weight"
                    ))
                    .is_some())
    }

    fn block_router_weight(&self, block: usize) -> Option<Result<Vec<f32>>> {
        // Try canonical
        let canonical = format!("blk.{block}.experts_weights.weight");
        if self.resolve(&canonical).is_some() {
            return Some(self.read_f32(&canonical));
        }
        // Try HuggingFace naming
        let hf = format!("model.layers.{block}.block_sparse_moe.gate.weight");
        if self.resolve(&hf).is_some() {
            return Some(self.read_f32(&hf));
        }
        None
    }

    fn block_expert_weights(
        &self,
        block: usize,
        expert_idx: usize,
        suffix: &str,
    ) -> Result<Vec<f32>> {
        let canonical = format!("blk.{block}.experts.{expert_idx}.{suffix}");
        if let Some(resolved) = self.resolve(&canonical) {
            return self.read_raw_f32(&resolved);
        }
        let hf_suffix = match suffix {
            "ffn_gate.weight" => "gate_proj.weight",
            "ffn_up.weight" => "up_proj.weight",
            "ffn_down.weight" => "down_proj.weight",
            _ => suffix,
        };
        let hf = format!(
            "model.layers.{block}.block_sparse_moe.experts.{expert_idx}.{hf_suffix}"
        );
        self.read_f32(&hf)
    }
}

/// Infer hyperparameters from ONNX tensor shapes.
fn infer_params_from_onnx(file: &OnnxFile, tensor_names: &[String]) -> ModelHyperparams {
    let mut hidden_dim = 0;
    let mut block_count = 0;
    let mut ffn_dim = 0;
    let mut n_heads = 0;
    let mut n_kv_heads = 0;
    let mut vocab_size = 0;
    let mut n_experts = 0;

    for name in tensor_names {
        // hidden_dim from token embedding
        if (name == "token_embd.weight"
            || name == "model.embed_tokens.weight"
            || name == "tok_embeddings.weight")
            && let Some(t) = file.tensor(name)
            && t.shape.len() >= 2
        {
            vocab_size = t.shape[0] as usize;
            hidden_dim = t.shape[1] as usize;
        }

        // block count
        if let Some(idx) = naming::hf_to_canonical(name)
            .or_else(|| Some(name.clone()))
            .and_then(|c| {
                c.strip_prefix("blk.")
                    .and_then(|r| r.split('.').next()?.parse::<i32>().ok())
            })
        {
            block_count = block_count.max(idx as usize + 1);
        }

        // Also try ONNX naming
        if let Some(idx) = crate::formats::onnx::block_index_from_name_onnx(name) {
            block_count = block_count.max(idx as usize + 1);
        }

        // ffn_dim
        if (name.contains("ffn_gate.weight") || name.contains("gate_proj.weight"))
            && let Some(t) = file.tensor(name)
            && t.shape.len() >= 2
            && t.shape[0] > ffn_dim as u64
        {
            ffn_dim = t.shape[0] as usize;
        }

        // n_heads
        if (name.contains("attn_q.weight") || name.contains("q_proj.weight"))
            && let Some(t) = file.tensor(name)
            && t.shape.len() >= 2
            && t.shape[0] > n_heads as u64
        {
            n_heads = t.shape[0] as usize;
        }

        // n_kv_heads
        if (name.contains("attn_k.weight") || name.contains("k_proj.weight"))
            && let Some(t) = file.tensor(name)
            && t.shape.len() >= 2
        {
            n_kv_heads = t.shape[0] as usize;
        }

        // experts
        if name.contains("experts.") && name.contains("gate_proj.weight") {
            if let Some(eidx) = extract_expert_idx_onnx(name) {
                n_experts = n_experts.max(eidx + 1);
            }
        }
    }

    if n_kv_heads == 0 {
        n_kv_heads = n_heads;
    }
    let head_dim = if n_heads > 0 {
        hidden_dim / n_heads
    } else {
        128
    };

    ModelHyperparams {
        arch: "llama".into(),
        block_count: block_count.max(1),
        hidden_dim: hidden_dim.max(1),
        n_heads: n_heads.max(1),
        n_kv_heads: n_kv_heads.max(1),
        head_dim,
        ffn_dim: ffn_dim.max(hidden_dim * 4),
        vocab_size: vocab_size.max(1),
        norm_eps: 1e-5,
        n_experts,
        n_experts_per_tok: if n_experts > 0 { 2 } else { 0 },
    }
}

fn extract_expert_idx_onnx(name: &str) -> Option<usize> {
    // patterns: model.layers.N.block_sparse_moe.experts.M.gate_proj.weight
    //           layers.N.mlp.experts.M.gate_proj.weight
    for prefix in &[
        "block_sparse_moe.experts.",
        "mlp.experts.",
        "experts.",
    ] {
        if let Some(rest) = name.strip_prefix(prefix) {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<usize>() {
                    return Some(idx);
                }
            }
        }
    }

    // Fallback: look for "experts" followed by a number anywhere
    if let Some(pos) = name.find(".experts.") {
        let after = &name[pos + 9..];
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }

    None
}
