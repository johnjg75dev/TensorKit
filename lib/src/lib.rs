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
pub mod formats;
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
    apply_embed_prune, build_plan, parse_selection, plan_embed_prune, EmbedPrunePlan,
    PrunePlan, PruneReport, Selection, TokenSelection,
};
pub use quantize::{is_quantizable, quantize};
pub use report::render_html_report;
pub use svd::{
    apply_to_gguf as svd_apply_to_gguf, apply_to_safetensors as svd_apply_to_safetensors,
    build_plan as build_svd_plan, LayerSelection, OutputDtype, RankClamps, RankSpec,
    RankSpecWithClamps, SvdApplied, SvdConfig, SvdPlan, SvdReport, SvdTarget, TensorSelection,
};

pub fn git_version() -> &'static str {
    env!("GIT_VERSION")
}
