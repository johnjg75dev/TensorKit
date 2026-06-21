//! `tensorkit` — analyze, prune, SVD-compress, merge, quantize, and MoE-merge
//! AI model files.
//!
//! Subcommands:
//! - `analyze` — run heuristic analysis, print recommendation
//! - `prune` — remove specified blocks/layers
//! - `svd` — SVD-compress specified tensors
//! - `quant` — quantize model to specified GGML type
//! - `merge` — merge N models (weighted average, slerp)
//! - `moe` — auto-detect experts, merge or purge experts
//! - `pipeline` — run a pipeline from a JSON config file
//! - `interact` — interactive wizard to build and run a pipeline
//! - `bench` — benchmark operations
//! - `test` — run self-tests

mod pipeline;

use clap::{Parser, Subcommand};
use regex::Regex;
use tensorkit::formats::gguf::dequant as gguf_dequant;
use tensorkit::formats::gguf::writer::GgufWriter;
use tensorkit::formats::gguf::GgufFile;
use tensorkit::formats::gguf::types::{dims_product, GgmlType, MetaValue, MetadataKv};
use tensorkit::formats::onnx::OnnxFile;
use tensorkit::formats::safetensors::SafetensorsFile;
use tensorkit::merge::{slerp_tensors, MoEWeights, MoEMergeStrategy, merge_experts};
use tensorkit::model::ModelFormat;
use tensorkit::prune::{
    apply_to_gguf, apply_to_onnx, apply_to_safetensors, build_plan, parse_selection,
};
use tensorkit::quantize::apply::quantize_gguf as quantize_gguf_apply;
use tensorkit::svd::{
    AdjacentSelection, LayerSelection, OutputDtype, RankSpecWithClamps, SvdBackend, SvdConfig, TensorSelection,
    apply_to_gguf as svd_apply_gguf, apply_to_onnx as svd_apply_onnx, apply_to_safetensors as svd_apply_st,
    build_plan as build_svd_plan,
};
use tensorkit::{Analyzer, Error};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const VERSION: &str = env!("GIT_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "tensorkit",
    version = VERSION,
    about = "tensorkit — these go to eleven.",
    long_about = "Analyze, prune, SVD-compress, merge, and quantize AI model files.\n\n\
                  Run `tensorkit <command> --help` for command-specific options."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Analyze a model and print recommendations
    Analyze {
        /// Path to the model file
        model: PathBuf,

        /// Number of elements to sample per tensor
        #[arg(long, default_value_t = 200_000)]
        sample: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Write HTML report to this path
        #[arg(long, value_name = "PATH")]
        report: Option<PathBuf>,

        /// Path to HTML template file (optional, uses built-in if not provided)
        #[arg(long, value_name = "PATH")]
        template: Option<PathBuf>,

        /// Dry run: analyze and print results without writing any outputs
        #[arg(long)]
        dry_run: bool,
    },

    /// Prune specified blocks/layers from a model
    Prune {
        /// Path to the model file
        model: PathBuf,

        /// Selection grammar (e.g., "auto:5", "keep:0,1,2", "drop:7,8", "pattern:.*ffn.*")
        #[arg(long)]
        selection: String,

        /// Output path
        #[arg(long, short = 'o')]
        out: PathBuf,

        /// Verify output after writing
        #[arg(long)]
        verify: bool,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// SVD-compress specified tensors
    Svd {
        /// Path to the model file
        model: PathBuf,

        /// Layer selection (e.g., "all", "all-attn", "0-5,10", "regex:^blk\\.0\\.")
        #[arg(long)]
        layers: String,

        /// Tensor selection within layers (e.g., "mlp", "attn", "all", "ffn_gate,ffn_up")
        #[arg(long, default_value = "mlp")]
        tensors: String,

        /// Rank specification (e.g., "64", "0.5", "energy:0.99", "frac:0.5,min:4")
        #[arg(long, default_value = "0.5,min:4")]
        rank: String,

        /// Output dtype (f32, f16, bf16, auto)
        #[arg(long, default_value = "f16")]
        dtype: String,

        /// SVD backend: "jacobi" (pure-Rust parallel) or "faer" (LAPACK-quality)
        #[arg(long, default_value = "faer")]
        backend: String,

        /// Minimum dimension to qualify for SVD
        #[arg(long, default_value_t = 16)]
        min_dim: usize,

        /// Adjacent tensor selection (e.g., "attn_v+1,ffn_up-2")
        #[arg(long)]
        adjacent: Option<String>,

        /// Output path
        #[arg(long, short = 'o')]
        out: PathBuf,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,

        /// Disable randomized SVD (forces Jacobi)
        #[arg(long)]
        no_randomized: bool,
    },

    /// Quantize model to a GGML type
    Quant {
        /// Path to the model file (GGUF only for now)
        model: PathBuf,

        /// Target quantization type
        #[arg(long)]
        target: String,

        /// Output path
        #[arg(long, short = 'o')]
        out: PathBuf,

        /// Quantize only specified blocks/layers (comma-separated indices or "all")
        #[arg(long, default_value = "all")]
        blocks: String,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Merge N models (weighted). Same-file merge when only model_a is given.
    Merge {
        /// Path to first model (required)
        model_a: PathBuf,

        /// Path to second model (omit for same-file merge)
        #[arg(long)]
        model_b: Option<PathBuf>,

        /// Merge mode: "average" or "slerp:<t>"
        #[arg(long, default_value = "average")]
        mode: String,

        /// Comma-separated weights (e.g., "0.5,0.3,0.2"; default: equal)
        #[arg(long)]
        weights: Option<String>,

        /// JSON config file (overrides --mode, --weights)
        #[arg(long)]
        config: Option<PathBuf>,

        /// Output path
        #[arg(long, short = 'o')]
        out: PathBuf,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Auto-detect MoE experts, merge or purge
    Moe {
        /// Path to model
        model: PathBuf,

        /// Strategy: "average" or "similarity:k"
        #[arg(long, default_value = "average")]
        strategy: String,

        /// Output path
        #[arg(long, short = 'o')]
        out: PathBuf,

        /// Expert tensor name pattern override (regex, default: auto-detect)
        #[arg(long)]
        expert_pattern: Option<String>,

        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Run a pipeline of operations defined in a JSON config file
    Pipeline {
        /// Path to the JSON pipeline definition
        config: PathBuf,
    },

    /// Interactive wizard: build a pipeline step by step, save/load config,
    /// then approve and execute
    Interact,

    /// Benchmark operations
    Bench {
        /// Operation to benchmark (analyze, svd, quant)
        #[arg(default_value = "analyze")]
        op: String,

        /// Model path
        #[arg(long)]
        model: Option<PathBuf>,

        /// Iterations
        #[arg(long, default_value_t = 3)]
        iterations: usize,
    },

    /// Run self-tests
    Test {
        /// Test category (all, dequant, svd, quant)
        #[arg(default_value = "all")]
        category: String,
    },

    /// Run inference and view activations at each layer
    Infer {
        /// Path to the GGUF model file
        model: PathBuf,

        /// Input token IDs (comma-separated, e.g., "1,2,3,4")
        #[arg(long, default_value = "1")]
        tokens: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), Error> {
    match cli.command {
        Commands::Analyze { model, sample, json, report, template, dry_run } => {
            run_analyze(&model, sample, json, report.as_deref(), template.as_deref(), dry_run)
        }
        Commands::Prune { model, selection, out, verify, yes } => {
            run_prune(&model, &selection, &out, verify, yes)
        }
        Commands::Svd { model, layers, tensors, rank, dtype, backend, min_dim, adjacent, out, yes, no_randomized } => {
            run_svd(&model, &layers, &tensors, &rank, &dtype, &backend, min_dim, adjacent.as_deref(), &out, yes, no_randomized)
        }
        Commands::Quant { model, target, out, blocks, yes } => {
            run_quant(&model, &target, &out, &blocks, yes)
        }
        Commands::Merge { model_a, model_b, mode, weights, config, out, yes } => {
            run_merge(&model_a, model_b.as_deref(), &mode, weights.as_deref(), config.as_deref(), &out, yes)
        }
        Commands::Moe { model, strategy, out, expert_pattern, yes } => {
            run_moe(&model, &strategy, &out, expert_pattern.as_deref(), yes)
        }
        Commands::Pipeline { config } => {
            run_pipeline(&config)
        }
        Commands::Interact => {
            pipeline::run_interactive()
        }
        Commands::Bench { op, model, iterations } => {
            run_bench(&op, model.as_deref(), iterations)
        }
        Commands::Test { category } => {
            run_test(&category)
        }
        Commands::Infer { model, tokens, json } => {
            run_infer(&model, &tokens, json)
        }
    }
}

pub(crate) fn run_analyze(model: &Path, sample: usize, json: bool, report_path: Option<&Path>, template_path: Option<&Path>, dry_run: bool) -> Result<(), Error> {
    let format = ModelFormat::from_path(model);
    eprintln!("[open] {} (format: {})", model.display(), format.as_str());

    if dry_run {
        eprintln!("[dry-run] analyze only — no files will be written");
    }

    match format {
        ModelFormat::Gguf => {
            let gg = GgufFile::open(model)?;
            let analysis = Analyzer::with_sample_per_tensor(sample).analyze(&gg)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&analysis)?);
            } else {
                print_human_report(&gg, &analysis);
            }
            if let Some(path) = report_path {
                if dry_run {
                    eprintln!("[dry-run] would write report to {}", path.display());
                } else {
                    write_html_report(&analysis, path, template_path)?;
                    eprintln!("[report] wrote {}", path.display());
                }
            }
            if dry_run {
                print_dry_run_analysis(&analysis);
            }
        }
        ModelFormat::Safetensors => {
            let st = SafetensorsFile::open(model)?;
            let analysis = Analyzer::with_sample_per_tensor(sample).analyze(&st)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&analysis)?);
            } else {
                print_human_report(&st, &analysis);
            }
            if let Some(path) = report_path {
                if dry_run {
                    eprintln!("[dry-run] would write report to {}", path.display());
                } else {
                    write_html_report(&analysis, path, template_path)?;
                    eprintln!("[report] wrote {}", path.display());
                }
            }
            if dry_run {
                print_dry_run_analysis(&analysis);
            }
        }
        ModelFormat::Onnx => {
            let onnx = OnnxFile::open(model)?;
            let analysis = Analyzer::with_sample_per_tensor(sample).analyze(&onnx)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&analysis)?);
            } else {
                print_human_report(&onnx, &analysis);
            }
            if let Some(path) = report_path {
                if dry_run {
                    eprintln!("[dry-run] would write report to {}", path.display());
                } else {
                    write_html_report(&analysis, path, template_path)?;
                    eprintln!("[report] wrote {}", path.display());
                }
            }
            if dry_run {
                print_dry_run_analysis(&analysis);
            }
        }
        ModelFormat::Unknown => {
            return Err(Error::Gguf(format!(
                "unsupported file extension for '{}'. Supported formats: .gguf, .safetensors/.st, .onnx",
                model.display()
            )));
        }
    }
    Ok(())
}

pub(crate) fn run_prune(model: &Path, selection_str: &str, out: &Path, verify: bool, yes: bool) -> Result<(), Error> {
    let format = ModelFormat::from_path(model);
    eprintln!("[open] {} (format: {})", model.display(), format.as_str());

    let selection = parse_selection(selection_str)?;
    
    match format {
        ModelFormat::Gguf => {
            let gg = GgufFile::open(model)?;
            let analysis = Analyzer::new().analyze(&gg)?;
            let plan = build_plan(&gg, &selection, Some(&analysis.blocks))?;
            confirm_or_exit(&format!("prune {:?}", plan.dropped_blocks), out, yes)?;
            let report = apply_to_gguf(&gg, &plan, out)?;
            print_prune_report(&report);
            if verify {
                verify_gguf(out, &plan)?;
            }
        }
        ModelFormat::Safetensors => {
            let st = SafetensorsFile::open(model)?;
            let analysis = Analyzer::new().analyze(&st)?;
            let plan = build_plan(&st, &selection, Some(&analysis.blocks))?;
            confirm_or_exit(&format!("prune {:?}", plan.dropped_blocks), out, yes)?;
            let report = apply_to_safetensors(&st, &plan, out)?;
            print_prune_report(&report);
        }
        ModelFormat::Onnx => {
            let onnx = OnnxFile::open(model)?;
            let analysis = Analyzer::new().analyze(&onnx)?;
            let plan = build_plan(&onnx, &selection, Some(&analysis.blocks))?;
            confirm_or_exit(&format!("prune {:?}", plan.dropped_blocks), out, yes)?;
            let report = apply_to_onnx(&onnx, &plan, out)?;
            print_prune_report(&report);
        }
        ModelFormat::Unknown => {
            return Err(Error::Gguf(format!(
                "unsupported file extension for '{}'. Supported formats: .gguf, .safetensors/.st, .onnx",
                model.display()
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_svd(
    model: &Path, layers_str: &str, tensors_str: &str, rank_str: &str, dtype_str: &str,
    backend_str: &str, min_dim: usize, adjacent_str: Option<&str>, out: &Path, yes: bool, no_randomized: bool
) -> Result<(), Error> {
    let format = ModelFormat::from_path(model);
    eprintln!("[open] {} (format: {})", model.display(), format.as_str());

    let layers = LayerSelection::parse(layers_str)
        .map_err(|e| Error::Gguf(format!("--layers: {e}")))?;
    let tensors = TensorSelection::parse(tensors_str)
        .map_err(|e| Error::Gguf(format!("--tensors: {e}")))?;
    let rank = RankSpecWithClamps::parse(rank_str)
        .map_err(|e| Error::Gguf(format!("--rank: {e}")))?;
    let dtype = OutputDtype::parse(dtype_str)
        .map_err(|e| Error::Gguf(format!("--dtype: {e}")))?;
    let backend = SvdBackend::parse(backend_str)
        .map_err(|e| Error::Gguf(format!("--backend: {e}")))?;
    
    let adjacent = if let Some(s) = adjacent_str {
        Some(AdjacentSelection::parse(s)
            .map_err(|e| Error::Gguf(format!("--adjacent: {e}")))?)
    } else {
        None
    };

    eprintln!("[svd] backend: {backend}");

    let cfg = SvdConfig {
        layers, tensors, rank, dtype, backend, min_dim,
        randomized: !no_randomized, randomized_oversample: 8, randomized_power_iters: 2, randomized_min_elems: 262_144,
        suffix_a: ".svd_a".into(), suffix_b: ".svd_b".into(),
        per_layer: Default::default(), per_tensor: Vec::new(), adjacent: adjacent.flatten(),
    };

    match format {
        ModelFormat::Gguf => {
            let gg = GgufFile::open(model)?;
            let plan = build_svd_plan(&gg, &cfg)?;
            print_svd_plan_summary(&plan);
            confirm_or_exit(&format!("SVD compress {} targets", plan.targets.len()), out, yes)?;
            let report = svd_apply_gguf(&gg, &plan, out)?;
            print_svd_report(&report);
        }
        ModelFormat::Safetensors => {
            let st = SafetensorsFile::open(model)?;
            let plan = build_svd_plan(&st, &cfg)?;
            print_svd_plan_summary(&plan);
            confirm_or_exit(&format!("SVD compress {} targets", plan.targets.len()), out, yes)?;
            let report = svd_apply_st(&st, &plan, out)?;
            print_svd_report(&report);
        }
        ModelFormat::Onnx => {
            let onnx = OnnxFile::open(model)?;
            let plan = build_svd_plan(&onnx, &cfg)?;
            print_svd_plan_summary(&plan);
            confirm_or_exit(&format!("SVD compress {} targets", plan.targets.len()), out, yes)?;
            let report = svd_apply_onnx(&onnx, &plan, out)?;
            print_svd_report(&report);
        }
        ModelFormat::Unknown => {
            return Err(Error::Gguf(format!(
                "unsupported file extension for '{}'. Supported formats: .gguf, .safetensors/.st, .onnx",
                model.display()
            )));
        }
    }
    Ok(())
}

pub(crate) fn run_quant(model: &Path, target_str: &str, out: &Path, blocks_str: &str, yes: bool) -> Result<(), Error> {
    let format = ModelFormat::from_path(model);
    if format != ModelFormat::Gguf {
        return Err(Error::Gguf("quantize currently only supports GGUF".into()));
    }
    
    let target = parse_quant_type(target_str)?;
    eprintln!("[quant] {} -> {} as {}", model.display(), out.display(), target.as_str());
    
    let blocks = if blocks_str == "all" || blocks_str.is_empty() {
        None
    } else {
        let indices = tensorkit::prune::selection::parse_index_list(blocks_str)
            .map_err(|e| Error::Gguf(format!("invalid --blocks: {e}")))?;
        Some(indices)
    };
    
    confirm_or_exit(&format!("quantize to {}", target.as_str()), out, yes)?;
    let report = quantize_gguf_apply(model, target, out, blocks.as_deref())?;
    print_quant_report(&report);
    Ok(())
}

pub(crate) fn run_merge(
    model_a: &Path, model_b: Option<&Path>, mode: &str,
    weights_str: Option<&str>, config_path: Option<&Path>,
    out: &Path, yes: bool,
) -> Result<(), Error> {
    // Collect models.
    let models: Vec<&Path> = if let Some(mb) = model_b {
        vec![model_a, mb]
    } else {
        vec![model_a]
    };
    let n = models.len();

    // Parse config / weights / mode.
    let (_, per_model_weights) = if let Some(cfg) = config_path {
        let text = std::fs::read_to_string(cfg).map_err(Error::Io)?;
        let cfg: MergeConfig = serde_json::from_str(&text)
            .map_err(|e| Error::Gguf(format!("merge config: {e}")))?;
        (cfg.strategy.clone(), cfg.weights)
    } else {
        let mode = mode.to_string();
        let weights: Option<Vec<f64>> = weights_str.map(|s| {
            s.split(',')
                .map(|w| w.trim().parse::<f64>().map_err(|_| Error::Gguf(format!("bad weight '{w}'"))))
                .collect::<Result<Vec<_>, _>>()
        }).transpose()?;
        (mode, weights)
    };

    let weights: Vec<f64> = per_model_weights.unwrap_or_else(|| {
        let w = 1.0 / n as f64;
        vec![w; n]
    });
    if weights.len() != n {
        return Err(Error::Gguf(format!(
            "expected {n} weights but got {}", weights.len()
        )));
    }
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum.abs() < 1e-12 {
        return Err(Error::Gguf("sum of weights is zero".into()));
    }
    let weights: Vec<f32> = weights.into_iter().map(|w| (w / weight_sum) as f32).collect();

    let slerp_t = if let Some(rest) = mode.strip_prefix("slerp:") {
        Some(rest.parse::<f32>()
            .map_err(|_| Error::Gguf(format!("bad slerp t: '{rest}'")))?)
    } else if mode == "average" { None } else {
        return Err(Error::Gguf(format!("unknown merge mode '{mode}' (use 'average' or 'slerp:<t>')")));
    };

    // Open all models.
    let mut ggs: Vec<GgufFile> = Vec::with_capacity(n);
    for &p in &models {
        eprintln!("[open] {}", p.display());
        ggs.push(GgufFile::open(p)?);
    }

    // Verify tensor count match for cross-file merge.
    if n > 1 {
        let ref_count = ggs[0].tensors.len();
        for (i, gg) in ggs.iter().enumerate().skip(1) {
            if gg.tensors.len() != ref_count {
                return Err(Error::Gguf(format!(
                    "model {} has {} tensors, expected {}", i + 1, gg.tensors.len(), ref_count
                )));
            }
        }
    }

    let writer_version = ggs[0].version;
    let writer_alignment = ggs[0].alignment;

    // For same-file merge (n==1), operate on the single model's tensors
    // (pairs of tensors within the same file). This only makes sense
    // for MoE-style merging; for now just copy.
    if n == 1 {
        // Same-file merge: copy all tensors verbatim (no-op).
        eprintln!("[merge] single-file mode: copying tensors");
        let gg = &ggs[0];
        let mut writer = GgufWriter::new(writer_version, writer_alignment);
        for kv in &gg.metadata {
            writer.add_kv(kv.clone());
        }
        for ti in &gg.tensors {
            let src = gg.tensor_slice(ti).unwrap_or_default();
            writer.add_tensor(ti.name.clone(), ti.n_dims, ti.dims, ti.ggml_type, src);
        }
        confirm_or_exit("merge (same-file copy)", out, yes)?;
        let out_bytes = writer.into_bytes()?;
        let mut f = std::fs::File::create(out)?;
        f.write_all(&out_bytes)?;
        return Ok(());
    }

    // Cross-file merge: iterate tensors in parallel.
    let mode_desc = slerp_t.map(|t| format!("slerp({t})")).unwrap_or_else(|| "average".into());
    let weight_desc: Vec<String> = weights.iter().map(|w| format!("{:.3}", w)).collect();
    confirm_or_exit(
        &format!("merge {} models ({}) weights={}", n, mode_desc, weight_desc.join(",")),
        out, yes,
    )?;

    let ref_gg = &ggs[0];
    let mut writer = GgufWriter::new(writer_version, writer_alignment);
    for kv in &ref_gg.metadata {
        writer.add_kv(kv.clone());
    }
    // Add merge metadata.
    writer.add_kv(MetadataKv {
        key: "tensorkit.merge.source".into(),
        value_type: 8,
        value: MetaValue::String(models.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>().join(", ")),
    });
    writer.add_kv(MetadataKv {
        key: "tensorkit.merge.mode".into(),
        value_type: 8,
        value: MetaValue::String(mode_desc.clone()),
    });

    for (ti_ref, ti_other) in ref_gg.tensors.iter().zip(ggs[1].tensors.iter()) {
        if ti_ref.name != ti_other.name {
            return Err(Error::Gguf(format!(
                "tensor name mismatch: '{}' vs '{}'", ti_ref.name, ti_other.name
            )));
        }
        let name = &ti_ref.name;
        // Dequantize both tensors.
        let a_bytes = ref_gg.tensor_slice(ti_ref).unwrap_or_default();
        let b_bytes = ggs[1].tensor_slice(ti_other).unwrap_or_default();
        let deq_a = gguf_dequant::dequantize(ti_ref.ggml_type, a_bytes, None)
            .unwrap_or_else(|| a_bytes.iter().map(|&b| b as f32).collect());
        let deq_b = gguf_dequant::dequantize(ti_other.ggml_type, b_bytes, None)
            .unwrap_or_else(|| b_bytes.iter().map(|&b| b as f32).collect());

        if deq_a.len() != deq_b.len() {
            return Err(Error::Gguf(format!(
                "tensor '{name}' length mismatch: {} vs {}", deq_a.len(), deq_b.len()
            )));
        }

        let merged = if let Some(t) = slerp_t {
            slerp_tensors(&deq_a, &deq_b, t)
        } else {
            // Weighted average.
            let wa = weights[0];
            let wb = weights[1];
            deq_a.iter().zip(&deq_b).map(|(a, b)| a * wa + b * wb).collect()
        };

        // Keep output in the first model's quant type.
        let out_ty = if ti_ref.ggml_type.block_size() > 1 {
            GgmlType::F32  // Write merged tensors as F32
        } else {
            ti_ref.ggml_type
        };
        let out_bytes: Vec<u8> = if out_ty == ti_ref.ggml_type && ti_ref.ggml_type.block_size() == 1 {
            // Non-quantized: store F32 directly.
            merged.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            merged.iter().flat_map(|v| v.to_le_bytes()).collect()
        };

        let out_ty = GgmlType::F32;
        writer.add_tensor(ti_ref.name.clone(), ti_ref.n_dims, ti_ref.dims, out_ty, &out_bytes);
        eprintln!("  merged {}", ti_ref.name);
    }

    let out_bytes = writer.into_bytes()?;
    let mut f = std::fs::File::create(out)?;
    f.write_all(&out_bytes)?;
    eprintln!("[merge] wrote {}", out.display());
    Ok(())
}

pub(crate) fn run_moe(model: &Path, strategy: &str, out: &Path, expert_pattern: Option<&str>, yes: bool) -> Result<(), Error> {
    eprintln!("[open] {}", model.display());
    let gg = GgufFile::open(model)?;

    // Auto-detect experts or use pattern override.
    let pat = expert_pattern.map(|s| s.to_string()).unwrap_or_else(|| {
        // Common MoE tensor patterns.
        r"(ffn_gate|ffn_up|ffn_down)_exp[s]?\d*|experts?\.\d+\.|moe\.\w+"
            .to_string()
    });
    let re = Regex::new(&pat)
        .map_err(|e| Error::Gguf(format!("bad expert pattern '{pat}': {e}")))?;

    // Detect: group tensors by the part before the expert index suffix.
    // For each matched tensor, extract the base name and expert index.
    // Strategy: find groups of 4+ tensors matching a pattern with a numeric index.
    let mut seen: HashMap<String, Vec<String>> = HashMap::new();

    // Simple heuristic: group tensors ending in _exp0, _exp1, etc.
    let group_re = Regex::new(r"^(.*_exp[s]?)(\d+)$")
        .map_err(|e| Error::Gguf(format!("internal: {e}")))?;

    for t in &gg.tensors {
        let name = &t.name;
        if let Some(caps) = group_re.captures(name) {
            let base = caps[1].to_string();
            seen.entry(base).or_default().push(name.clone());
        } else if re.is_match(name) {
            // Non-grouped expert pattern matched.
        }
    }

    // Filter to groups that look like expert families (2+ tensors).
    let mut groups: Vec<(String, Vec<String>)> = seen.into_iter().filter(|(_, v)| v.len() >= 2).collect();

    if groups.is_empty() {
        return Err(Error::Gguf("no expert tensors detected".into()));
    }

    // For each group, merge experts.
    let mut writer = GgufWriter::new(gg.version, gg.alignment);
    for kv in &gg.metadata {
        writer.add_kv(kv.clone());
    }
    writer.add_kv(MetadataKv {
        key: "tensorkit.moe.strategy".into(),
        value_type: 8,
        value: MetaValue::String(strategy.into()),
    });

    let moe_strat = parse_moe_strategy(strategy)?;

    // Copy tensors that are NOT experts.
    let all_expert: std::collections::HashSet<&str> = groups.iter()
        .flat_map(|(_, v)| v.iter().map(|s| s.as_str()))
        .collect();

    let mut merged_count = 0;
    let mut copy_count = 0;

    for ti in &gg.tensors {
        if all_expert.contains(ti.name.as_str()) {
            continue; // handled below
        }
        // Copy verbatim.
        let src = gg.tensor_slice(ti).unwrap_or_default();
        writer.add_tensor(ti.name.clone(), ti.n_dims, ti.dims, ti.ggml_type, src);
        copy_count += 1;
    }

    for (base, names) in &mut groups {
        // Dequantize and merge each expert.
        names.sort();
        let infos: Vec<_> = names.iter().filter_map(|n| gg.get_tensor(n)).collect();
        if infos.is_empty() {
            continue;
        }
        let ref_info = infos[0];
        let deq_experts: Vec<Vec<f32>> = infos.iter().map(|info| {
            let bytes = gg.tensor_slice(info).unwrap_or_default();
            gguf_dequant::dequantize(info.ggml_type, bytes, None)
                .unwrap_or_else(|| bytes.iter().map(|&b| b as f32).collect())
        }).collect();

        let shape = (dims_product(&ref_info.dims, ref_info.n_dims) as usize, 1usize);
        let moe_w = MoEWeights::new(deq_experts, shape);

        let merged = merge_experts(&moe_w, moe_strat);
        let merged_bytes: Vec<u8> = merged.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Write merged tensor (as F32).
        let merged_name = format!("{base}merged");
        writer.add_tensor(merged_name, ref_info.n_dims, ref_info.dims, GgmlType::F32, &merged_bytes);
        merged_count += 1;
        eprintln!("  merged {}: {} experts -> 1", base, names.len());
    }

    confirm_or_exit(&format!("MoE merge: {} expert groups, {} copied", merged_count, copy_count), out, yes)?;
    let out_bytes = writer.into_bytes()?;
    let mut f = std::fs::File::create(out)?;
    f.write_all(&out_bytes)?;
    eprintln!("[moe] wrote {}", out.display());
    Ok(())
}

fn parse_moe_strategy(s: &str) -> Result<MoEMergeStrategy, Error> {
    if s == "average" {
        Ok(MoEMergeStrategy::Average)
    } else if let Some(k) = s.strip_prefix("similarity:") {
        let k = k.parse::<usize>()
            .map_err(|_| Error::Gguf(format!("bad similarity k '{k}'")))?;
        Ok(MoEMergeStrategy::Similarity { keep_top_k: k })
    } else {
        Err(Error::Gguf(format!(
            "unknown MoE strategy '{s}' (use 'average' or 'similarity:<k>')"
        )))
    }
}

#[derive(serde::Deserialize)]
struct MergeConfig {
    #[serde(default)]
    strategy: String,
    #[serde(default)]
    weights: Option<Vec<f64>>,
}

fn run_pipeline(config_path: &Path) -> Result<(), Error> {
    use pipeline::PipelineStep;
    let cfg = pipeline::PipelineConfig::from_json(config_path)?;
    let mut model = cfg.model;
    for (i, step) in cfg.steps.iter().enumerate() {
        eprintln!("[pipeline] step {}: {:?}", i + 1, step);
        match step {
            PipelineStep::Analyze { sample, json, report, template } => {
                let m = model.as_ref().ok_or_else(|| Error::Gguf("pipeline: no model set for analyze step".into()))?;
                run_analyze(m, sample.unwrap_or(200_000), json.unwrap_or(false), report.as_ref().map(|v| &**v), template.as_ref().map(|v| &**v), false)?;
            }
            PipelineStep::Quant { quant_type, out, verify } => {
                let m = model.as_ref().ok_or_else(|| Error::Gguf("pipeline: no model set for quant step".into()))?;
                run_quant(m, quant_type, out, "", verify.unwrap_or(false))?;
                model = Some(out.clone());
            }
            PipelineStep::Prune { selection, out, verify } => {
                let m = model.as_ref().ok_or_else(|| Error::Gguf("pipeline: no model set for prune step".into()))?;
                run_prune(m, selection, out, verify.unwrap_or(false), true)?;
                model = Some(out.clone());
            }
            _ => eprintln!("[pipeline] step {}: unsupported, skipping", i + 1),
        }
    }
    Ok(())
}

fn run_bench(op: &str, _model: Option<&Path>, _iterations: usize) -> Result<(), Error> {
    eprintln!("[bench] op={} iterations={}", op, _iterations);
    // TODO: implement benchmarking
    Ok(())
}

fn run_test(_category: &str) -> Result<(), Error> {
    eprintln!("[test] category={}", _category);
    // TODO: implement self-tests
    Ok(())
}

// ---- Helpers -------------------------------------------------------------

fn confirm_or_exit(action: &str, out: &Path, yes: bool) -> Result<(), Error> {
    eprintln!("[confirm] about to {} -> {}", action, out.display());
    if yes {
        eprintln!("           --yes set; proceeding.");
        return Ok(());
    }
    eprint!("           continue? [y/N] ");
    std::io::stderr().flush().ok();
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).map_err(Error::Io)?;
    let s = s.trim().to_ascii_lowercase();
    if s == "y" || s == "yes" {
        Ok(())
    } else {
        eprintln!("           cancelled.");
        std::process::exit(0);
    }
}

fn print_human_report<M: tensorkit::Model + ?Sized>(m: &M, a: &tensorkit::Analysis) {
    let total_bytes: u64 = m.tensors().iter().map(|t| t.byte_size).sum();
    println!("\n=== model summary ===");
    if let Some(n) = m.name() { println!("  model.name:     {n}"); }
    if let Some(a_) = m.architecture() { println!("  architecture:   {a_}"); }
    if let Some(bc) = m.block_count() { println!("  block_count:    {bc}"); }
    println!("  tensors:        {}", m.tensors().len());
    println!("  total size:     {:.2} MB", total_bytes as f64 / 1_048_576.0);
    println!("  sample/tensor:  {}", a.sample_per_tensor);

    println!("\n=== block summary ===");
    println!("{:<10}  {:<7}  {:>7}  {:>10}  {:>9}", "label", "role", "tensors", "bytes", "MB");
    let mut sortable: Vec<_> = a.blocks.iter().collect();
    sortable.sort_by_key(|b| (b.index, b.label.clone()));
    for b in &sortable {
        println!("{:<10}  {:<7}  {:>7}  {:>10}  {:>9.2}", b.label, b.role.as_str(), b.tensor_count, b.total_bytes, b.total_bytes as f64 / 1_048_576.0);
    }

    println!("\n=== recommended ===");
    println!("  auto-prune {} blocks: {:?}", a.recommendation_count, a.recommendation);
}

fn print_prune_report(r: &tensorkit::PruneReport) {
    println!("\n=== prune report ===");
    println!("  bytes:   {} -> {} ({:.2} MB saved)", r.bytes_in, r.bytes_out, (r.bytes_in as f64 - r.bytes_out as f64) / 1_048_576.0);
    println!("  tensors: kept={} dropped={}", r.tensors_kept, r.tensors_dropped);
    println!("  blocks:  {} -> {}", r.original_block_count, r.new_block_count);
    println!("  output:  {}", r.output_path);
}

fn print_svd_plan_summary(plan: &tensorkit::SvdPlan) {
    eprintln!("\n[SVD plan]");
    eprintln!("  blocks:           {}", plan.original_block_count);
    eprintln!("  targets:          {}", plan.targets.len());
    eprintln!("  skipped:          {}", plan.skipped.len());
    eprintln!("  orig:             {:.2} MB", plan.orig_bytes() as f64 / 1_048_576.0);
    eprintln!("  est. new:         {:.2} MB", plan.new_bytes() as f64 / 1_048_576.0);
    eprintln!("  est. compression: {:.1}%", plan.compression_ratio() * 100.0);
}

fn print_svd_report(r: &tensorkit::SvdReport) {
    println!("\n=== SVD report ===");
    println!("  output:       {}", r.output_path);
    println!("  file size:    {} -> {} bytes ({:.2} MB -> {:.2} MB)", r.bytes_in, r.bytes_out, r.bytes_in as f64 / 1_048_576.0, r.bytes_out as f64 / 1_048_576.0);
    println!("  compression:  {:.1}%", r.compression_ratio * 100.0);
    println!("  targets:      {}", r.applied.len());
    println!("  skipped:      {}", r.skipped.len());
}

fn print_quant_report(r: &tensorkit::quantize::apply::QuantizeReport) {
    println!("\n=== quantize report ===");
    println!("  output:       {}", r.output_path);
    println!("  target:       {}", r.target);
    println!("  tensors:      {} total, {} quantized", r.tensors_total, r.tensors_quantized);
    println!("  bytes:        {} -> {} ({:.2} MB -> {:.2} MB)", r.bytes_in, r.bytes_out, r.bytes_in as f64 / 1_048_576.0, r.bytes_out as f64 / 1_048_576.0);
    println!("  compression:  {:.1}%", r.compression_ratio * 100.0);
}

fn verify_gguf(path: &Path, plan: &tensorkit::PrunePlan) -> Result<(), Error> {
    use tensorkit::formats::gguf::verify;
    let kept: Vec<String> = plan.keep.iter().filter(|(_, k)| *k).map(|(n, _)| n.clone()).collect();
    let r = verify::verify(path, &kept)?;
    println!("\n=== verify ===");
    if r.ok { println!("  PASS"); } else { println!("  FAIL"); }
    println!("  tensors: {}", r.kept_tensors);
    println!("  bytes:   {} ({:.2} MB)", r.total_bytes, r.total_bytes as f64 / 1_048_576.0);
    for e in &r.errors { println!("  ERROR: {e}"); }
    for w in &r.warnings { println!("  warn:  {w}"); }
    Ok(())
}

fn print_dry_run_analysis(analysis: &tensorkit::Analysis) {
    println!("\n=== DRY RUN: Expected Impact ===");
    let total_mb = analysis.total_bytes as f64 / 1_048_576.0;
    println!("  model size:    {:.2} MB", total_mb);

    if analysis.recommendation_count > 0 {
        let prune_mb = (analysis.total_bytes - analysis.estimated_bytes_after_prune) as f64 / 1_048_576.0;
        let after_mb = analysis.estimated_bytes_after_prune as f64 / 1_048_576.0;
        println!("  prune:         {} blocks recommended", analysis.recommendation_count);
        println!("  savings:       {:.2} MB ({:.1}% reduction)", prune_mb, prune_mb / total_mb * 100.0);
        println!("  estimated:     {:.2} MB after prune", after_mb);
        let scores: Vec<String> = analysis.blocks.iter()
            .filter(|b| b.removable > 0.5)
            .map(|b| format!("{} ({:.2})", b.label, b.removable))
            .collect();
        println!("  candidates:    {}", scores.join(", "));
    } else {
        println!("  prune:         no blocks recommended for pruning");
    }

    let total_tensors: usize = analysis.blocks.iter().map(|b| b.tensors.len()).sum();
    println!("  tensors:       {} total", total_tensors);
    println!("  blocks:        {} total", analysis.blocks.len());

    // Show spectra availability
    let spec_count: usize = analysis.blocks.iter().map(|b| b.spectra.len()).sum();
    if spec_count > 0 {
        println!("  spectra:       {} tensors have SVD data available", spec_count);
    }

    println!("\n  No files were written (--dry-run).");
}

fn parse_quant_type(s: &str) -> Result<GgmlType, Error> {
    match s.to_ascii_lowercase().as_str() {
        "q2_k" => Ok(GgmlType::Q2K),
        "q3_k" => Ok(GgmlType::Q3K),
        "q4_0" => Ok(GgmlType::Q4_0),
        "q4_1" => Ok(GgmlType::Q4_1),
        "q4_k" => Ok(GgmlType::Q4K),
        "q5_0" => Ok(GgmlType::Q5_0),
        "q5_1" => Ok(GgmlType::Q5_1),
        "q5_k" => Ok(GgmlType::Q5K),
        "q6_k" => Ok(GgmlType::Q6K),
        "q8_0" => Ok(GgmlType::Q8_0),
        "q8_1" => Ok(GgmlType::Q8_1),
        "q8_k" => Ok(GgmlType::Q8K),
        "iq1_s" => Ok(GgmlType::Iq1S),
        "iq1_m" => Ok(GgmlType::Iq1M),
        "iq2_s" => Ok(GgmlType::Iq2S),
        "iq2_xxs" => Ok(GgmlType::Iq2Xxs),
        "iq2_xs" => Ok(GgmlType::Iq2Xs),
        "iq3_xxs" => Ok(GgmlType::Iq3Xxs),
        "iq3_s" => Ok(GgmlType::Iq3S),
        "iq4_nl" => Ok(GgmlType::Iq4Nl),
        "iq4_xs" => Ok(GgmlType::Iq4Xs),
        "tq1_0" => Ok(GgmlType::Tq1_0),
        "tq2_0" => Ok(GgmlType::Tq2_0),
        other => Err(Error::Gguf(format!(
            "unsupported quant type {:?} (supported: q2_k, q3_k, q4_0, q4_1, q4_k, q5_0, q5_1, q5_k, q6_k, q8_0, q8_1, q8_k, iq1_s, iq1_m, iq2_s, iq2_xxs, iq2_xs, iq3_xxs, iq3_s, iq4_nl, iq4_xs, tq1_0, tq2_0)", other
        ))),
    }
}

fn write_html_report(analysis: &tensorkit::Analysis, path: &Path, template_path: Option<&Path>) -> Result<(), Error> {
    let html = tensorkit::render_html_report(analysis, template_path)
        .map_err(|e| Error::Gguf(format!("report generation failed: {e}")))?;
    std::fs::write(path, html).map_err(Error::Io)?;
    Ok(())
}

fn run_infer(model: &Path, tokens_str: &str, json: bool) -> Result<(), Error> {
    let token_ids: Vec<u32> = tokens_str
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| Error::Infer(format!("invalid token ID: {e}")))?;

    if token_ids.is_empty() {
        return Err(Error::Infer("no token IDs provided".into()));
    }

    tensorkit::infer::activations::run_inference(model, &token_ids, json)
}