//! Lightweight inference engine for viewing activations at each transformer layer.
//!
//! Loads a GGUF model, runs a single forward pass, and captures the input/output
//! of every sub-layer (attention, FFN, residual streams) for inspection.
//!
//! # Architecture support
//! Currently supports LLaMA-style architectures (RMSNorm, SwiGLU FFN, GQA attention,
//! RoPE). Other architectures will return an error.

pub mod activations;
pub mod layer;
pub mod model;
pub mod ops;

pub use activations::{ActivationSnapshot, LayerActivations};
pub use model::InferenceModel;
