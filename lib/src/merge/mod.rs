//! Model "merge" subsystem: layer merging, MoE expert merging, depth
//! expansion, and weight tying.
//!
//! All operations are pure Rust with no I/O — they take `&[f32]` (or
//! `&dyn Model` for metadata) and return owned values. The CLI layer
//! wires them up to actual model files.
//!
//! Quick map of the public surface:
//! ```text
//!   average   — elementwise mean of two tensors
//!   slerp     — spherical linear interpolation of two tensors
//!   moe       — merge N expert weight matrices into one
//!   depth     — insert (duplicate or zero-fill) a new transformer block
//!   tying     — detect / record weight tying between embed and output
//!   strategy  — shared enums (MergeStrategy, WeightFormat)
//! ```

mod average;
mod depth;
mod moe;
mod slerp;
mod strategy;
pub mod task_arith;
mod tying;

#[cfg(test)]
#[path = "../../tests/unit/merge/tests.rs"]
mod tests;

pub use average::{average_into, average_tensors};
pub use depth::{insert_block, InsertPlan, InsertResult, InsertSource};
pub use moe::{merge_experts, MoEMergeStrategy, MoEWeights};
pub use slerp::{slerp_tensors, SlerpT};
pub use strategy::{MergeStrategy, WeightFormat};
pub use task_arith::{
    apply_task_vector, apply_task_vector_owned, compute_task_vector, elect_sign, ties_merge,
    trim_task_vector, MultiplierHub, TaskEntry, TiesConfig,
};
pub use tying::{apply_tying, plan_tying, verify_tying_compatible, TyingPlan, TyingResult};
