//! Safetensors inference provider.
//!
//! Loads a single `.safetensors` file or a sharded directory, reads tensor
//! weights as f32, and implements `WeightProvider` for the inference engine.

use crate::error::{Error, Result};
use crate::formats::gguf::dequant::bf16_to_f32;
use crate::formats::gguf::dequant::f16_to_f32;
use crate::formats::safetensors::reader::SafetensorsFile;
use crate::model::{Model, TensorDtype};
use std::collections::HashMap;
use std::path::Path;

use super::config::{hf_config_to_hyperparams, read_hf_config};
use super::naming;
use super::WeightProvider;
use super::ModelHyperparams;

/// Safetensors inference provider (single file or sharded).
pub struct SafetensorsProvider {
    shards: Vec<SafetensorsFile>,
    params: ModelHyperparams,
    name_map: HashMap<String, String>,
    tensor_names: Vec<String>,
    /// Maps tensor name → (shard index, actual name in shard)
    tensor_shard_map: HashMap<String, (usize, String)>,
}

impl SafetensorsProvider {
    /// Open a single `.safetensors` file.
    pub fn open(path: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(path)?;
        let tensor_names: Vec<String> = file.tensors.iter().map(|t| t.name.clone()).collect();
        let name_map = naming::build_name_map(&tensor_names);

        // Build shard map: all tensors in shard 0
        let mut tensor_shard_map = HashMap::new();
        for name in &tensor_names {
            tensor_shard_map.insert(name.clone(), (0, name.clone()));
        }

        // Try config.json
        let hf_config = path.parent().and_then(read_hf_config);
        let params = if let Some(config) = &hf_config {
            hf_config_to_hyperparams(config, None, 0)
        } else {
            None
        };
        let params = params.unwrap_or_else(|| infer_params_from_tensors(&file, &tensor_names));

        Ok(Self {
            shards: vec![file],
            params,
            name_map,
            tensor_names,
            tensor_shard_map,
        })
    }

    /// Open a sharded safetensors directory (with `model.safetensors.index.json`).
    pub fn open_sharded(dir: &Path) -> Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        let index_data = std::fs::read(&index_path).map_err(Error::Io)?;
        let index: serde_json::Value =
            serde_json::from_slice(&index_data).map_err(|e| Error::Infer(format!("index json: {e}")))?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Infer("index missing weight_map".into()))?;

        // Group tensors by shard file
        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        shard_names.sort();

        let mut shards = Vec::new();
        let mut tensor_shard_map = HashMap::new();
        let mut all_tensor_names = Vec::new();

        for (shard_idx, shard_name) in shard_names.iter().enumerate() {
            let shard_path = dir.join(shard_name);
            let shard_file = SafetensorsFile::open(&shard_path)?;

            for t in shard_file.tensors.iter() {
                all_tensor_names.push(t.name.clone());
                tensor_shard_map.insert(t.name.clone(), (shard_idx, t.name.clone()));
            }

            shards.push(shard_file);
        }

        let name_map = naming::build_name_map(&all_tensor_names);

        // Build params
        let params = if let Some(config) = read_hf_config(dir) {
            hf_config_to_hyperparams(&config, None, 0)
        } else {
            None
        };
        let params = params.unwrap_or_else(|| {
            infer_params_from_names(&all_tensor_names)
        });

        Ok(Self {
            shards,
            params,
            name_map,
            tensor_names: all_tensor_names,
            tensor_shard_map,
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
        let (shard_idx, actual_name) = self
            .tensor_shard_map
            .get(name)
            .or_else(|| self.tensor_shard_map.values().find(|(_, n)| n == name))
            .ok_or_else(|| Error::Infer(format!("tensor not found: {name}")))?;

        let shard = &self.shards[*shard_idx];
        let ti = shard
            .tensor(actual_name)
            .ok_or_else(|| Error::Infer(format!("tensor not found in shard: {name}")))?;
        let dtype = ti.dtype;
        let raw = Model::read_tensor_bytes(shard, actual_name)
            .map_err(|e| Error::Infer(format!("read failed: {e}")))?;

        match ti.dtype {
            TensorDtype::F32 => {
                let mut out = Vec::with_capacity(raw.len() / 4);
                for chunk in raw.chunks_exact(4) {
                    out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
                Ok(out)
            }
            TensorDtype::F16 => {
                let mut out = Vec::with_capacity(raw.len() / 2);
                for chunk in raw.chunks_exact(2) {
                    let u16_val = u16::from_le_bytes(chunk.try_into().unwrap());
                    out.push(f16_to_f32(u16_val));
                }
                Ok(out)
            }
            TensorDtype::Bf16 => {
                let mut out = Vec::with_capacity(raw.len() / 2);
                for chunk in raw.chunks_exact(2) {
                    let u16_val = u16::from_le_bytes(chunk.try_into().unwrap());
                    out.push(bf16_to_f32(u16_val));
                }
                Ok(out)
            }
            _ => Err(Error::Infer(format!(
                "unsupported dtype {dtype:?} for tensor {name}"
            ))),
        }
    }
}

impl WeightProvider for SafetensorsProvider {
    fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let resolved = self
            .resolve(name)
            .map(String::from)
            .or_else(|| {
                // Try HuggingFace variants
                for variant in naming::onnx_variants(name) {
                    if self.tensor_shard_map.contains_key(&variant) {
                        return Some(variant);
                    }
                }
                None
            })
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
            && self
                .tensor_shard_map
                .contains_key(&format!("blk.{block}.experts.0.ffn_gate.weight"))
    }

    fn block_router_weight(&self, block: usize) -> Option<Result<Vec<f32>>> {
        let canonical = format!("blk.{block}.experts_weights.weight");
        if self.resolve(&canonical).is_some() {
            return Some(self.read_f32(&canonical));
        }
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
        if self.resolve(&canonical).is_some() {
            return self.read_raw_f32(&canonical);
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

/// Infer hyperparameters from tensor names.
fn infer_params_from_tensors(_file: &SafetensorsFile, tensor_names: &[String]) -> ModelHyperparams {
    infer_params_from_names(tensor_names)
}

fn infer_params_from_names(tensor_names: &[String]) -> ModelHyperparams {
    let hidden_dim = 0;
    let mut block_count = 0;
    let ffn_dim = 0;
    let n_heads = 0;
    let n_kv_heads = 0;
    let vocab_size = 0;
    let mut n_experts = 0;

    for name in tensor_names {
        if name == "token_embd.weight" || name == "model.embed_tokens.weight" || name == "tok_embeddings.weight" {
            // Can't read shapes from names alone, but we know the embedding exists
        }

        if let Some(idx) = naming::hf_to_canonical(name)
            .or_else(|| Some(name.clone()))
            .and_then(|c| {
                c.strip_prefix("blk.")
                    .and_then(|r| r.split('.').next()?.parse::<i32>().ok())
            })
        {
            block_count = block_count.max(idx as usize + 1);
        }

        if name.contains("ffn_gate.weight") || name.contains("gate_proj.weight") {
            // ffn_dim will be inferred from shapes when available
        }

        if name.contains("attn_q.weight") || name.contains("q_proj.weight") {
            // n_heads inferred from shapes
        }

        if name.contains("experts.") && name.contains("gate_proj.weight") {
            if let Some(eidx) = extract_expert_idx(name) {
                n_experts = n_experts.max(eidx + 1);
            }
        }
    }

    ModelHyperparams {
        arch: "llama".into(),
        block_count: block_count.max(1),
        hidden_dim: hidden_dim.max(1),
        n_heads: n_heads.max(1),
        n_kv_heads: n_kv_heads.max(1),
        head_dim: if n_heads > 0 { hidden_dim / n_heads } else { 128 },
        ffn_dim: ffn_dim.max(4096),
        vocab_size: vocab_size.max(1),
        norm_eps: 1e-5,
        n_experts,
        n_experts_per_tok: if n_experts > 0 { 2 } else { 0 },
    }
}

fn extract_expert_idx(name: &str) -> Option<usize> {
    for prefix in &["experts.", "mlp.experts.", "block_sparse_moe.experts."] {
        if let Some(rest) = name.strip_prefix(prefix).or_else(|| {
            name.find('.')
                .map(|i| &name[i + 1..])
                .and_then(|r| r.strip_prefix(prefix))
        }) {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<usize>() {
                    return Some(idx);
                }
            }
        }
    }

    if let Some(pos) = name.find("experts.") {
        let after = &name[pos + 8..];
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }

    None
}
