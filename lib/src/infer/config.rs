//! HuggingFace `config.json` parser for model hyperparameters.
//!
//! When a `config.json` exists alongside a Safetensors file, we read
//! model hyperparameters from it to avoid guessing from tensor shapes.

use serde::Deserialize;
use std::path::Path;

/// Partial HuggingFace config.json — only fields we need.
#[derive(Debug, Deserialize)]
pub struct HfConfig {
    pub model_type: Option<String>,
    pub hidden_size: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub vocab_size: Option<usize>,
    pub rms_norm_eps: Option<f64>,
    pub max_position_embeddings: Option<usize>,

    // MoE fields
    pub num_local_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: Option<usize>,

    // Alternative MoE field names (some models use these)
    pub expert_interval: Option<usize>,
    pub n_routed_experts: Option<usize>,
    pub n_experts_per_tok: Option<usize>,
}

/// Try to read a HuggingFace config.json from the given directory.
pub fn read_hf_config(dir: &Path) -> Option<HfConfig> {
    let config_path = dir.join("config.json");
    if !config_path.exists() {
        return None;
    }
    let data = std::fs::read(&config_path).ok()?;
    serde_json::from_slice(&data).ok()
}

/// Convert HfConfig to our ModelHyperparams (filling in what we can).
pub fn hf_config_to_hyperparams(
    config: &HfConfig,
    arch_override: Option<&str>,
    n_experts_from_detection: usize,
) -> Option<super::ModelHyperparams> {
    let hidden_size = config.hidden_size?;
    let n_heads = config.num_attention_heads?;
    let head_dim = hidden_size / n_heads;

    let arch = arch_override
        .or(config.model_type.as_deref())
        .unwrap_or("llama")
        .to_string();

    Some(super::ModelHyperparams {
        arch,
        block_count: config.num_hidden_layers?,
        hidden_dim: hidden_size,
        n_heads,
        n_kv_heads: config.num_key_value_heads.unwrap_or(n_heads),
        head_dim,
        ffn_dim: config.intermediate_size.unwrap_or(hidden_size * 4),
        vocab_size: config.vocab_size.unwrap_or(32000),
        norm_eps: config.rms_norm_eps.unwrap_or(1e-5) as f32,
        n_experts: config
            .num_local_experts
            .or(config.n_routed_experts)
            .unwrap_or(n_experts_from_detection),
        n_experts_per_tok: config
            .num_experts_per_tok
            .or(config.n_experts_per_tok)
            .unwrap_or(2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_config() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, Some(4096));
        assert_eq!(config.num_attention_heads, Some(32));
        assert_eq!(config.num_local_experts, None);
    }

    #[test]
    fn parse_moe_config() {
        let json = r#"{
            "model_type": "mixtral",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "num_local_experts": 8,
            "num_experts_per_tok": 2
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_local_experts, Some(8));
        assert_eq!(config.num_experts_per_tok, Some(2));
    }

    #[test]
    fn config_to_hyperparams() {
        let config = HfConfig {
            model_type: Some("llama".into()),
            hidden_size: Some(2048),
            num_attention_heads: Some(16),
            num_key_value_heads: Some(8),
            intermediate_size: Some(5632),
            num_hidden_layers: Some(24),
            vocab_size: Some(32000),
            rms_norm_eps: Some(1e-5),
            max_position_embeddings: None,
            num_local_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            expert_interval: None,
            n_routed_experts: None,
            n_experts_per_tok: None,
        };
        let params = hf_config_to_hyperparams(&config, None, 0).unwrap();
        assert_eq!(params.hidden_dim, 2048);
        assert_eq!(params.n_heads, 16);
        assert_eq!(params.n_kv_heads, 8);
        assert_eq!(params.head_dim, 128);
        assert_eq!(params.block_count, 24);
        assert_eq!(params.n_experts, 0);
    }
}
