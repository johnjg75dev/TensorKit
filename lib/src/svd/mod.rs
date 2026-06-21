//! SVD-based low-rank compression of 2-D weight matrices.
//!
//! Mirrors the layout of the existing `prune` module: a `config` (selection
//! grammar) feeds a `plan` (which tensors, what rank), which is then
//! materialized to disk by `apply`.
//!
//! Quick start:
//! ```no_run
//! use tensorkit::formats::gguf::GgufFile;
//! use tensorkit::svd::{build_plan, apply_to_gguf, OutputDtype, SvdConfig, LayerSelection, TensorSelection, RankSpec, RankSpecWithClamps, RankClamps};
//!
//! let gg = GgufFile::open("model.gguf")?;
//! let cfg = SvdConfig {
//!     layers: LayerSelection::AllMlp,
//!     tensors: TensorSelection::Mlp,
//!     rank: RankSpecWithClamps { spec: RankSpec::Fraction(0.5), clamps: RankClamps { min: 8, max: None } },
//!     dtype: OutputDtype::F16,
//!     ..Default::default()
//! };
//! let plan = build_plan(&gg, &cfg)?;
//! let report = apply_to_gguf(&gg, &plan, std::path::Path::new("model.svd.gguf"))?;
//! println!("compressed {:.1}% of {:.2} MB target bytes",
//!          report.compression_ratio * 100.0,
//!          report.orig_tensor_bytes as f64 / 1_048_576.0);
//! # Ok::<(), tensorkit::Error>(())
//! ```

pub mod apply;
pub mod config;
pub mod faer_backend;
pub mod linalg;
pub mod plan;

pub use apply::{apply_to_gguf, apply_to_onnx, apply_to_safetensors, SvdApplied, SvdReport};
pub use config::{
    AdjacentEntry, AdjacentRole, AdjacentSelection, LayerSelection, OutputDtype, RankClamps,
    RankSpec, RankSpecWithClamps, SvdBackend, SvdConfig, TensorSelection, ATTN_SUFFIXES, FFN_SUFFIXES,
};
pub use plan::{build_plan, SkippedTensor, SvdPlan, SvdTarget};

pub use linalg::{
    evd_symmetric, jacobi_2x2, orthonormalize_cols, pack_lowrank, rank_for_energy, reconstruct,
    slice_cols, slice_rows, svd_jacobi, svd_randomized, transpose, AlignedVec, Mat, Svd,
};
pub use faer_backend::svd_faer;
