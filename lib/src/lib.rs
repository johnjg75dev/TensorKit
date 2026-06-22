//! `tensorkit` — fast AI model analyzer and transformer-block pruner.
//!
//! Supports GGUF (v1–v3), safetensors, and ONNX. Per-block "removability"
//! is scored heuristically by default; the `calibrate` feature uses `candle`
//! to run a forward pass and rank blocks by activation-delta instead.
//!
//! Quick start:
//! ```no_run
//! use tensorkit::{Analyzer, formats::gguf::GgufFile};
//!
//! let model = GgufFile::open("model.gguf")?;
//! let analysis = Analyzer::with_sample_per_tensor(200_000).analyze(&model)?;
//! println!("recommended: {:?}", analysis.recommendation);
//! # Ok::<(), tensorkit::Error>(())
//! ```
//!
//! The CLI binary is `tensorkit`.

#![allow(clippy::needless_range_loop)]

pub mod analysis;
#[cfg(feature = "calibrate")]
pub mod calibrate;
pub mod error;
pub mod ffi;
pub mod formats;
pub mod infer;
pub mod merge;
pub mod model;
pub mod prune;
pub mod quantize;
pub mod report;
pub mod svd;

#[cfg(test)]
#[path = "../tests/unit/tests/mod.rs"]
mod tests;

pub use analysis::{
    tensor_spectrum, Analysis, Analyzer, BlockAnalysis, Chart, ChartSeries, PerChannelStats,
    ReportSection, TensorAnalysis, TensorStats,
};
pub use error::{Error, Result};
pub use merge::{
    apply_task_vector, apply_task_vector_owned, apply_tying as merge_apply_tying, average_into,
    average_tensors, compute_task_vector, elect_sign, insert_block, merge_experts, plan_tying,
    slerp_tensors, ties_merge, trim_task_vector, verify_tying_compatible, InsertPlan,
    InsertResult, InsertSource, MergeStrategy, MoEMergeStrategy, MoEWeights, MultiplierHub,
    SlerpT, TaskEntry, TiesConfig, TyingPlan, TyingResult, WeightFormat,
};
pub use model::{BlockRef, MetadataValue, Model, ModelFormat, Tensor, TensorDtype};
pub use prune::{
    apply_embed_prune, build_plan, gguf_value_type, is_block_count_key, is_tensor_count_key,
    looks_like_per_layer_array, parse_block_key, parse_selection, plan_embed_prune, rename_block,
    rename_metadata_block_key, shrink_array, EmbedPrunePlan, PrunePlan, PruneReport, Selection,
    TokenSelection,
};
pub use quantize::{dispatch_quantize, is_quantizable, quantize};
pub use quantize::apply::{block_index_from_name, max_abs_diff};
pub use report::render_html_report;
pub use svd::{
    apply_to_gguf as svd_apply_to_gguf, apply_to_safetensors as svd_apply_to_safetensors,
    build_plan as build_svd_plan, evd_symmetric, jacobi_2x2, orthonormalize_cols, pack_lowrank,
    rank_for_energy, reconstruct, slice_cols, slice_rows, svd_faer, svd_jacobi, svd_randomized,
    transpose, LayerSelection, OutputDtype, RankClamps, RankSpec, RankSpecWithClamps, SvdApplied,
    SvdBackend, SvdConfig, SvdPlan, SvdReport, SvdTarget, TensorSelection, AlignedVec, Mat, Svd,
};
pub use infer::{
    WeightProvider, ModelHyperparams, InterpretationSnapshot,
    interpret::ExpertStats,
};

pub fn git_version() -> &'static str {
    env!("GIT_VERSION")
}
