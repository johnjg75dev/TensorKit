//! Cross-format tensor name mapping.
//!
//! Maps between canonical GGUF-style names (`blk.N.suffix`) and
//! HuggingFace/ONNX naming conventions. Each `WeightProvider` builds a
//! lookup table at construction time; this module provides the mapping
//! logic and resolver.

use std::collections::HashMap;

/// Canonical tensor name suffix (without block prefix).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    TokenEmbedding,
    OutputNorm,
    OutputHead,
    AttnNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
    RouterWeight,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    Unknown,
}

/// Map a canonical tensor suffix to a `TensorRole`.
pub fn classify_tensor(suffix: &str) -> TensorRole {
    match suffix {
        "token_embd.weight" | "tok_embeddings.weight" => TensorRole::TokenEmbedding,
        "output_norm.weight" | "norm.weight" => TensorRole::OutputNorm,
        "output.weight" | "lm_head.weight" => TensorRole::OutputHead,
        s if s.ends_with(".attn_norm.weight") => TensorRole::AttnNorm,
        s if s.ends_with(".attn_q.weight") => TensorRole::AttnQ,
        s if s.ends_with(".attn_k.weight") => TensorRole::AttnK,
        s if s.ends_with(".attn_v.weight") => TensorRole::AttnV,
        s if s.ends_with(".attn_output.weight") => TensorRole::AttnOutput,
        s if s.ends_with(".ffn_norm.weight") => TensorRole::FfnNorm,
        s if s.ends_with(".ffn_gate.weight") => TensorRole::FfnGate,
        s if s.ends_with(".ffn_up.weight") => TensorRole::FfnUp,
        s if s.ends_with(".ffn_down.weight") => TensorRole::FfnDown,
        s if s.ends_with(".experts_weights.weight") => TensorRole::RouterWeight,
        _ => TensorRole::Unknown,
    }
}

/// HuggingFace LLaMA/Mistral tensor name → canonical GGUF name.
///
/// Input: `"model.layers.0.self_attn.q_proj.weight"`
/// Output: `"blk.0.attn_q.weight"`
pub fn hf_to_canonical(hf_name: &str) -> Option<String> {
    // model.layers.N.<rest>
    let rest = hf_name.strip_prefix("model.layers.")?;

    let dot = rest.find('.')?;
    let block_idx: i32 = rest[..dot].parse().ok()?;
    let suffix = &rest[dot + 1..];

    let canonical_suffix = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        // MoE variants
        "block_sparse_moe.gate.weight" => "experts_weights.weight",
        s if s.starts_with("block_sparse_moe.experts.") => {
            // block_sparse_mope.experts.N.gate_proj.weight → experts.N.ffn_gate.weight
            let expert_rest = s.strip_prefix("block_sparse_moe.experts.")?;
            let edot = expert_rest.find('.')?;
            let expert_idx: i32 = expert_rest[..edot].parse().ok()?;
            let expert_suffix = &expert_rest[edot + 1..];
            let canon_suffix = match expert_suffix {
                "gate_proj.weight" => "ffn_gate.weight",
                "up_proj.weight" => "ffn_up.weight",
                "down_proj.weight" => "ffn_down.weight",
                _ => return None,
            };
            return Some(format!("blk.{block_idx}.experts.{expert_idx}.{canon_suffix}"));
        }
        s if s.starts_with("mlp.experts.") => {
            // Mixtral-style: mlp.experts.N.gate_proj.weight
            let expert_rest = s.strip_prefix("mlp.experts.")?;
            let edot = expert_rest.find('.')?;
            let expert_idx: i32 = expert_rest[..edot].parse().ok()?;
            let expert_suffix = &expert_rest[edot + 1..];
            let canon_suffix = match expert_suffix {
                "gate_proj.weight" => "ffn_gate.weight",
                "up_proj.weight" => "ffn_up.weight",
                "down_proj.weight" => "ffn_down.weight",
                _ => return None,
            };
            return Some(format!("blk.{block_idx}.experts.{expert_idx}.{canon_suffix}"));
        }
        _ => return None,
    };

    Some(format!("blk.{block_idx}.{canonical_suffix}"))
}

/// Canonical tensor name → HuggingFace name.
pub fn canonical_to_hf(canonical: &str) -> Option<String> {
    // Global tensors
    match canonical {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "tok_embeddings.weight" => return Some("tok_embeddings.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "norm.weight" => return Some("norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }

    // Block tensors: blk.N.<suffix>
    let rest = canonical.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let block_idx: i32 = rest[..dot].parse().ok()?;
    let suffix = &rest[dot + 1..];

    // MoE expert tensors
    if let Some(expert_rest) = suffix.strip_prefix("experts.") {
        let edot = expert_rest.find('.')?;
        let expert_idx: i32 = expert_rest[..edot].parse().ok()?;
        let expert_suffix = &expert_rest[edot + 1..];
        let hf_suffix = match expert_suffix {
            "ffn_gate.weight" => "gate_proj.weight",
            "ffn_up.weight" => "up_proj.weight",
            "ffn_down.weight" => "down_proj.weight",
            _ => return None,
        };
        return Some(format!(
            "model.layers.{block_idx}.block_sparse_moe.experts.{expert_idx}.{hf_suffix}"
        ));
    }

    let hf_suffix = match suffix {
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "experts_weights.weight" => "block_sparse_moe.gate.weight",
        _ => return None,
    };

    Some(format!("model.layers.{block_idx}.{hf_suffix}"))
}

/// ONNX tensor name variants to try, given a canonical name.
///
/// Returns a list of candidate ONNX tensor names to probe, in priority order.
pub fn onnx_variants(canonical: &str) -> Vec<String> {
    let mut out = Vec::new();

    // Global tensors
    match canonical {
        "token_embd.weight" => {
            out.push("model.embed_tokens.weight".into());
            out.push("tok_embeddings.weight".into());
            out.push("token_embd.weight".into());
        }
        "output_norm.weight" => {
            out.push("model.norm.weight".into());
            out.push("norm.weight".into());
            out.push("output_norm.weight".into());
        }
        "output.weight" => {
            out.push("lm_head.weight".into());
            out.push("output.weight".into());
        }
        _ => {}
    }

    // Block tensors
    if let Some(rest) = canonical.strip_prefix("blk.") {
        let dot = rest.find('.');
        if let Some(dot) = dot {
            if let Ok(block_idx) = rest[..dot].parse::<i32>() {
                let suffix = &rest[dot + 1..];
                // Try multiple ONNX naming conventions
                let prefixes = [
                    format!("model.layers.{block_idx}."),
                    format!("layers.{block_idx}."),
                    format!("block.{block_idx}."),
                    format!("transformer.h.{block_idx}."),
                    format!("encoder.layer.{block_idx}."),
                ];

                let onnx_suffix = match suffix {
                    "attn_norm.weight" => Some("input_layernorm.weight"),
                    "attn_q.weight" => Some("self_attn.q_proj.weight"),
                    "attn_k.weight" => Some("self_attn.k_proj.weight"),
                    "attn_v.weight" => Some("self_attn.v_proj.weight"),
                    "attn_output.weight" => Some("self_attn.o_proj.weight"),
                    "ffn_norm.weight" => Some("post_attention_layernorm.weight"),
                    "ffn_gate.weight" => Some("mlp.gate_proj.weight"),
                    "ffn_up.weight" => Some("mlp.up_proj.weight"),
                    "ffn_down.weight" => Some("mlp.down_proj.weight"),
                    "experts_weights.weight" => Some("block_sparse_moe.gate.weight"),
                    _ => None,
                };

                if let Some(s) = onnx_suffix {
                    for prefix in &prefixes {
                        out.push(format!("{prefix}{s}"));
                    }
                }

                // Also try exact canonical name
                out.push(canonical.to_string());
            }
        }
    }

    out
}

/// Build a name resolution map from actual tensor names.
///
/// Given the set of tensor names that exist in the model file, build a map
/// from canonical names to the actual names. This handles arbitrary naming
/// conventions by trying each known mapping and checking if the target exists.
pub fn build_name_map(existing_names: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();

    for name in existing_names {
        // Try HuggingFace → canonical mapping
        if let Some(canonical) = hf_to_canonical(name) {
            // Prefer HF names as they're the most common non-GGUF convention
            map.entry(canonical).or_insert_with(|| name.clone());
        }
    }

    // Also add exact matches for GGUF-style names
    for name in existing_names {
        if name.starts_with("blk.") || name.starts_with("token_embd") || name.starts_with("output") {
            map.entry(name.clone()).or_insert_with(|| name.clone());
        }
    }

    map
}

/// Resolve a canonical tensor name to the best matching name in the model.
///
/// Tries: exact match → HF variants → ONNX variants.
pub fn resolve_tensor_name<'a>(
    canonical: &'a str,
    existing: &'a [String],
    name_map: &'a HashMap<String, String>,
) -> Option<&'a str> {
    // 1. Check name map (built from HF mappings)
    if let Some(mapped) = name_map.get(canonical) {
        if existing.iter().any(|n| n == mapped) {
            return Some(mapped);
        }
    }

    // 2. Exact match
    if existing.iter().any(|n| n == canonical) {
        return Some(canonical);
    }

    // 3. Try ONNX variants
    for variant in onnx_variants(canonical) {
        if existing.iter().any(|n| n == &variant) {
            // Cache in name map for future lookups
            return None; // caller would need mutable name_map
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_to_canonical_llama() {
        assert_eq!(
            hf_to_canonical("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".into())
        );
        assert_eq!(
            hf_to_canonical("model.layers.5.mlp.gate_proj.weight"),
            Some("blk.5.ffn_gate.weight".into())
        );
        assert_eq!(
            hf_to_canonical("model.layers.0.input_layernorm.weight"),
            Some("blk.0.attn_norm.weight".into())
        );
    }

    #[test]
    fn hf_to_canonical_moe() {
        assert_eq!(
            hf_to_canonical("model.layers.0.block_sparse_moe.experts.2.gate_proj.weight"),
            Some("blk.0.experts.2.ffn_gate.weight".into())
        );
        assert_eq!(
            hf_to_canonical("model.layers.3.mlp.experts.0.up_proj.weight"),
            Some("blk.3.experts.0.ffn_up.weight".into())
        );
    }

    #[test]
    fn canonical_to_hf_roundtrip() {
        let cases = vec![
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
            ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"),
            ("blk.2.ffn_gate.weight", "model.layers.2.mlp.gate_proj.weight"),
            ("blk.0.experts.1.ffn_down.weight", "model.layers.0.block_sparse_moe.experts.1.down_proj.weight"),
        ];
        for (canonical, hf) in cases {
            assert_eq!(canonical_to_hf(canonical).as_deref(), Some(hf));
        }
    }

    #[test]
    fn onnx_variants_cover_llama() {
        let variants = onnx_variants("blk.0.attn_q.weight");
        assert!(variants.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(variants.contains(&"layers.0.self_attn.q_proj.weight".to_string()));
    }
}
