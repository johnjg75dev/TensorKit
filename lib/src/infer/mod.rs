//! Lightweight inference engine for viewing activations at each transformer layer.
//!
//! Loads a model (GGUF, Safetensors, or ONNX), runs a single forward pass,
//! and captures the input/output of every sub-layer (attention, FFN, residual
//! streams, MoE router decisions) for inspection and interpretability.
//!
//! # Architecture support
//! Currently supports LLaMA-style architectures (RMSNorm, SwiGLU FFN, GQA
//! attention, RoPE) and Mixture-of-Expert variants (Mixtral-style router +
//! per-expert FFN). Other architectures will return an error.
//!
//! # Interpretability
//! The `interpret` module provides tools for analyzing captured activations:
//! attention entropy, head importance, logit lens, neuron-level analysis,
//! and MoE expert utilization statistics.

pub mod activations;
pub mod config;
pub mod format;
pub mod interpret;
pub mod kv_cache;
pub mod layer;
pub mod model;
pub mod naming;
pub mod ops;
pub mod provider_onnx;
pub mod provider_safetensors;

use crate::error::Result;

pub use activations::{ActivationSnapshot, InterpretationSnapshot, LayerActivations, LogitLensEntry};
pub use kv_cache::KvCache;
pub use model::{InferenceModel, ModelHyperparams};
pub use interpret::ExpertStats;

/// Format-agnostic weight provider trait.
///
/// Each provider (GGUF, Safetensors, ONNX) implements this trait to supply
/// the inference engine with f32 weight data and model hyperparameters.
/// Tensor names use the canonical GGUF-style `blk.N.suffix` convention;
/// each provider maps these to its format's native naming internally.
pub trait WeightProvider: Send + Sync {
    /// Read a tensor's weights as f32 by canonical name (e.g. `blk.0.attn_q.weight`).
    fn read_f32(&self, name: &str) -> Result<Vec<f32>>;

    /// Read a per-block tensor. `suffix` is e.g. `"attn_q.weight"`.
    fn block_f32(&self, block: usize, suffix: &str) -> Result<Vec<f32>> {
        self.read_f32(&format!("blk.{block}.{suffix}"))
    }

    /// Model hyperparameters.
    fn params(&self) -> &ModelHyperparams;

    /// Number of experts per MoE layer (1 = no MoE, dense model).
    fn n_experts(&self) -> usize {
        0
    }

    /// Number of experts to route to per token (top-k). Only meaningful when n_experts > 1.
    fn n_experts_per_tok(&self) -> usize {
        0
    }

    /// Read MoE router gate weights for a given block: `blk.N.experts_weights.weight`.
    fn block_router_weight(&self, block: usize) -> Option<Result<Vec<f32>>> {
        let _ = block;
        None
    }

    /// Read per-expert FFN weights for a given block.
    /// Returns `Some(Ok(Vec<Vec<f32>>))` where the outer Vec has one entry per expert,
    /// each containing the concatenated gate+up+down weights for that expert.
    fn block_expert_weights(
        &self,
        block: usize,
        expert_idx: usize,
        suffix: &str,
    ) -> Result<Vec<f32>> {
        self.read_f32(&format!("blk.{block}.experts.{expert_idx}.{suffix}"))
    }

    /// Whether the given block uses MoE (has router + experts).
    fn is_moe_block(&self, block: usize) -> bool {
        let _ = block;
        false
    }
}
