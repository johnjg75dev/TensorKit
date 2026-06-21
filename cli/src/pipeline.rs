//! Pipeline config + interactive wizard.
//! Run `tensorkit interact` for the guided step-through UI.

use dialoguer::{Confirm, Input, Select};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Data model (shared between JSON file and interactive wizard)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub model: Option<PathBuf>,
    pub steps: Vec<PipelineStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum PipelineStep {
    Analyze {
        sample: Option<usize>,
        json: Option<bool>,
        report: Option<PathBuf>,
        template: Option<PathBuf>,
    },
    Quant {
        quant_type: String,
        out: PathBuf,
        verify: Option<bool>,
    },
    Prune {
        selection: String,
        out: PathBuf,
        verify: Option<bool>,
    },
    Svd {
        out: PathBuf,
        selection: Option<String>,
        rank: Option<String>,
        #[serde(rename = "dry-run")]
        dry_run: Option<bool>,
        verify: Option<bool>,
    },
    Merge {
        models: Vec<PathBuf>,
        out: PathBuf,
        weights: Option<Vec<f32>>,
        slerp: Option<f32>,
        verify: Option<bool>,
    },
    MoE {
        selection: Option<String>,
        strategy: Option<String>,
        out: Option<PathBuf>,
        dry_run: Option<bool>,
    },
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

impl PipelineConfig {
    pub fn from_json(path: &std::path::Path) -> Result<Self, crate::Error> {
        let contents = std::fs::read_to_string(path).map_err(crate::Error::Io)?;
        serde_json::from_str(&contents)
            .map_err(|e| crate::Error::Gguf(format!("pipeline config parse error: {e}")))
    }

    #[allow(dead_code)]
    pub fn from_json_str(s: &str) -> Result<Self, crate::Error> {
        serde_json::from_str(s)
            .map_err(|e| crate::Error::Gguf(format!("pipeline config parse error: {e}")))
    }

    pub fn to_json(&self) -> Result<String, crate::Error> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::Gguf(format!("pipeline config serialize error: {e}")))
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), crate::Error> {
        let json = self.to_json()?;
        std::fs::write(path, &json).map_err(crate::Error::Io)?;
        eprintln!("[save] wrote pipeline config to {}", path.display());
        Ok(())
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl PipelineStep {
    /// One-line description shown in menus and summaries.
    pub fn summary(&self) -> String {
        match self {
            PipelineStep::Analyze { report, .. } => {
                let r = report.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|| "(none)".into());
                format!("Analyze → report: {r}")
            }
            PipelineStep::Quant { quant_type, out, .. } => {
                format!("Quantize → {} → {}", quant_type, out.display())
            }
            PipelineStep::Prune { selection, out, .. } => {
                format!("Prune ({selection}) → {}", out.display())
            }
            PipelineStep::Svd { out, .. } => {
                format!("SVD compress → {}", out.display())
            }
            PipelineStep::Merge { models, out, .. } => {
                let names: Vec<_> = models.iter().map(|m| m.to_string_lossy().to_string()).collect();
                format!("Merge {} models → {}", names.join(", "), out.display())
            }
            PipelineStep::MoE { strategy, out, .. } => {
                let s = strategy.as_deref().unwrap_or("average");
                let o = out.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|| "(stdout)".into());
                format!("MoE ({s}) → {o}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Interactive wizard
// ---------------------------------------------------------------------------

/// Available quantisation types (same list as parse_quant_type in main.rs).
const QUANT_TYPES: &[&str] = &[
    "q2_k", "q3_k", "q4_0", "q4_1", "q4_k", "q5_0", "q5_1", "q5_k",
    "q6_k", "q8_0", "q8_1", "q8_k",
    "iq1_s", "iq1_m", "iq2_s", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq3_s",
    "iq4_nl", "iq4_xs",
    "tq1_0", "tq2_0",
];

/// Run the interactive pipeline builder.
pub fn run_interactive() -> Result<(), crate::Error> {
    let mut cfg = PipelineConfig { model: None, steps: Vec::new() };
    let mut dirty = false;

    eprintln!("{}", BANNER);
    eprintln!("Interactive pipeline builder. Build a sequence of operations,");
    eprintln!("save/load config, then execute when ready.\n");

    loop {
        // Show current state
        eprintln!("────────────────────────────────────────────");
        if let Some(ref m) = cfg.model {
            eprintln!("  Model: {}", m.display());
        } else {
            eprintln!("  Model: (not set)");
        }
        if cfg.steps.is_empty() {
            eprintln!("  Steps: (none)");
        } else {
            for (i, step) in cfg.steps.iter().enumerate() {
                eprintln!("  {}. {}", i + 1, step.summary());
            }
        }
        eprintln!("────────────────────────────────────────────\n");

        let items = vec![
            "Set model path",
            "Add step",
            "Remove step",
            "Reorder steps",
            "Save config",
            "Load config",
            "Run pipeline",
            "Exit",
        ];

        let sel = Select::new()
            .with_prompt("Main menu")
            .items(&items)
            .default(0)
            .interact()
            .map_err(|e| crate::Error::Gguf(format!("interactive select error: {e}")))?;

        match sel {
            0 => { // Set model path
                let path: String = Input::new()
                    .with_prompt("Path to model file")
                    .with_initial_text(cfg.model.as_ref().map(|p| p.to_string_lossy().to_string()).unwrap_or_default())
                    .interact_text()
                    .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
                if !path.is_empty() {
                    cfg.model = Some(PathBuf::from(&path));
                    dirty = true;
                }
            }
            1 => add_step(&mut cfg)?,  // Add step
            2 => remove_step(&mut cfg)?,  // Remove step
            3 => reorder_steps(&mut cfg)?,  // Reorder steps
            4 => { // Save config
                let path: String = Input::new()
                    .with_prompt("Save path (JSON)")
                    .interact_text()
                    .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
                if !path.is_empty() {
                    cfg.save(&PathBuf::from(&path))?;
                }
            }
            5 => { // Load config
                let path: String = Input::new()
                    .with_prompt("Load path (JSON)")
                    .interact_text()
                    .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
                if !path.is_empty() {
                    cfg = PipelineConfig::from_json(&PathBuf::from(&path))?;
                    dirty = true;
                    eprintln!("[load] loaded {} steps", cfg.steps.len());
                }
            }
            6 => { // Run pipeline
                if cfg.model.is_none() {
                    eprintln!("[error] no model set. Set one first.");
                    continue;
                }
                if cfg.steps.is_empty() {
                    eprintln!("[error] no steps defined. Add some first.");
                    continue;
                }
                eprintln!("\n=== Pipeline Summary ===");
                eprintln!("  Model: {}", cfg.model.as_ref().unwrap().display());
                for (i, step) in cfg.steps.iter().enumerate() {
                    eprintln!("  {}. {}", i + 1, step.summary());
                }
                let ok = Confirm::new()
                    .with_prompt("Execute this pipeline?")
                    .default(false)
                    .interact()
                    .map_err(|e| crate::Error::Gguf(format!("confirm error: {e}")))?;
                if ok {
                    execute_pipeline(&cfg)?;
                    eprintln!("\n[pipeline] all steps completed.");
                } else {
                    eprintln!("[pipeline] cancelled.");
                }
            }
            7 => { // Exit
                if dirty {
                    let ok = Confirm::new()
                        .with_prompt("Unsaved changes. Exit anyway?")
                        .default(false)
                        .interact()
                        .map_err(|e| crate::Error::Gguf(format!("confirm error: {e}")))?;
                    if !ok {
                        continue;
                    }
                }
                eprintln!("Goodbye.");
                break;
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Step sub-menus
// ---------------------------------------------------------------------------

fn add_step(cfg: &mut PipelineConfig) -> Result<(), crate::Error> {
    let items = vec!["Analyze", "Quantize", "Prune", "SVD", "Merge", "MoE", "(cancel)"];
    let sel = Select::new()
        .with_prompt("Step type")
        .items(&items)
        .default(0)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("select error: {e}")))?;

    let step = match sel {
        0 => prompt_analyze()?,
        1 => prompt_quant()?,
        2 => prompt_prune()?,
        3 => prompt_svd()?,
        4 => prompt_merge()?,
        5 => prompt_moe()?,
        _ => return Ok(()),
    };
    eprintln!("  Added: {}", step.summary());
    cfg.steps.push(step);
    Ok(())
}

fn remove_step(cfg: &mut PipelineConfig) -> Result<(), crate::Error> {
    if cfg.steps.is_empty() {
        eprintln!("[info] no steps to remove.");
        return Ok(());
    }
    let items: Vec<String> = cfg.steps.iter().enumerate()
        .map(|(i, s)| format!("{}. {}", i + 1, s.summary()))
        .chain(std::iter::once("(cancel)".into()))
        .collect();

    let sel = Select::new()
        .with_prompt("Remove which step?")
        .items(&items)
        .default(items.len() - 1)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("select error: {e}")))?;

    if sel < cfg.steps.len() {
        let removed = cfg.steps.remove(sel);
        eprintln!("  Removed: {}", removed.summary());
    }
    Ok(())
}

fn reorder_steps(cfg: &mut PipelineConfig) -> Result<(), crate::Error> {
    if cfg.steps.len() < 2 {
        eprintln!("[info] need at least 2 steps to reorder.");
        return Ok(());
    }
    let items: Vec<String> = cfg.steps.iter()
        .map(|s| s.summary())
        .collect();

    // Use a multi-select sort dialog.
    // dialoguer::Sort presents items and lets the user rearrange them.
    let order = dialoguer::Sort::new()
        .with_prompt("Reorder steps (space to move, enter to confirm)")
        .items(&items)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("sort error: {e}")))?;

    // Interpret the returned indices
    // Sort::interact() returns indices in user-chosen order.
    let old_steps = std::mem::take(&mut cfg.steps);
    for &idx in &order {
        if idx < old_steps.len() {
            cfg.steps.push(old_steps[idx].clone());
        }
    }
    // If Sort didn't return a full permutation, append remaining
    for (i, s) in old_steps.iter().enumerate() {
        if !order.contains(&i) {
            cfg.steps.push(s.clone());
        }
    }
    eprintln!("  Reordered.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-step prompts
// ---------------------------------------------------------------------------

fn prompt_analyze() -> Result<PipelineStep, crate::Error> {
    let sample: String = Input::new()
        .with_prompt("Samples per tensor (default 200000)")
        .default("200000".into())
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
    let sample: Option<usize> = sample.parse().ok();

    let json_out: bool = Confirm::new()
        .with_prompt("Output as JSON?")
        .default(false)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("confirm error: {e}")))?;

    let report: String = Input::new()
        .with_prompt("HTML report path (leave empty to skip)")
        .default("".into())
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
    let report = if report.is_empty() { None } else { Some(PathBuf::from(&report)) };

    let template: String = Input::new()
        .with_prompt("HTML template path (optional)")
        .default("".into())
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
    let template = if template.is_empty() { None } else { Some(PathBuf::from(&template)) };

    Ok(PipelineStep::Analyze { sample, json: Some(json_out), report, template })
}

fn prompt_quant() -> Result<PipelineStep, crate::Error> {
    let quant_idx = Select::new()
        .with_prompt("Quantization type")
        .items(QUANT_TYPES)
        .default(0)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("select error: {e}")))?;
    let quant_type = QUANT_TYPES[quant_idx].to_string();

    let out: String = Input::new()
        .with_prompt("Output path")
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
    let out = PathBuf::from(&out);

    let verify: bool = Confirm::new()
        .with_prompt("Verify output?")
        .default(false)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("confirm error: {e}")))?;

    Ok(PipelineStep::Quant { quant_type, out, verify: Some(verify) })
}

fn prompt_prune() -> Result<PipelineStep, crate::Error> {
    let selection: String = Input::new()
        .with_prompt("Selection (e.g. auto:5, keep:0,1,2, drop:7,8, pattern:.*ffn.*)")
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;

    let out: String = Input::new()
        .with_prompt("Output path")
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
    let out = PathBuf::from(&out);

    let verify: bool = Confirm::new()
        .with_prompt("Verify output?")
        .default(false)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("confirm error: {e}")))?;

    Ok(PipelineStep::Prune { selection, out, verify: Some(verify) })
}

fn prompt_svd() -> Result<PipelineStep, crate::Error> {
    let out: String = Input::new()
        .with_prompt("Output path")
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
    let out = PathBuf::from(&out);

    let selection: String = Input::new()
        .with_prompt("Tensor selection (e.g. mlp, attn, all, ffn_gate,ffn_up)")
        .default("mlp".into())
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;

    let rank: String = Input::new()
        .with_prompt("Rank (e.g. 64, 0.5, energy:0.99, frac:0.5,min:4)")
        .default("0.5,min:4".into())
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;

    let dry_run: bool = Confirm::new()
        .with_prompt("Dry run (show plan only)?")
        .default(false)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("confirm error: {e}")))?;

    let verify: bool = Confirm::new()
        .with_prompt("Verify output?")
        .default(false)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("confirm error: {e}")))?;

    Ok(PipelineStep::Svd {
        out,
        selection: Some(selection),
        rank: Some(rank),
        dry_run: Some(dry_run),
        verify: Some(verify),
    })
}

fn prompt_merge() -> Result<PipelineStep, crate::Error> {
    let out: String = Input::new()
        .with_prompt("Output path")
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
    let out = PathBuf::from(&out);

    let model_count: String = Input::new()
        .with_prompt("Number of models to merge")
        .default("2".into())
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
    let n: usize = model_count.parse().unwrap_or(2);

    let mut models = Vec::with_capacity(n);
    for i in 0..n {
        let path: String = Input::new()
            .with_prompt(format!("Path to model {}", i + 1))
            .interact_text()
            .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
        models.push(PathBuf::from(&path));
    }

    let mode_items = vec!["average", "slerp", "(cancel)"];
    let mode_sel = Select::new()
        .with_prompt("Merge mode")
        .items(&mode_items)
        .default(0)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("select error: {e}")))?;

    let (weights, slerp) = match mode_sel {
        0 => (None, None),
        1 => {
            let t: String = Input::new()
                .with_prompt("Slerp t parameter (0.0–1.0)")
                .default("0.5".into())
                .interact_text()
                .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
            (None, Some(t.parse::<f32>().unwrap_or(0.5)))
        }
        _ => return Ok(PipelineStep::Merge {
            models, out, weights: None, slerp: None, verify: Some(false),
        }),
    };

    let verify: bool = Confirm::new()
        .with_prompt("Verify output?")
        .default(false)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("confirm error: {e}")))?;

    Ok(PipelineStep::Merge { models, out, weights, slerp, verify: Some(verify) })
}

fn prompt_moe() -> Result<PipelineStep, crate::Error> {
    let strategy_items = vec!["average", "similarity:<k>", "(cancel)"];
    let sel = Select::new()
        .with_prompt("MoE strategy")
        .items(&strategy_items)
        .default(0)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("select error: {e}")))?;

    let strategy = match sel {
        0 => "average".to_string(),
        1 => {
            let k: String = Input::new()
                .with_prompt("Keep top-k experts")
                .default("2".into())
                .interact_text()
                .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;
            format!("similarity:{k}")
        }
        _ => return Ok(PipelineStep::MoE {
            selection: None, strategy: None, out: None, dry_run: None,
        }),
    };

    let out: String = Input::new()
        .with_prompt("Output path")
        .interact_text()
        .map_err(|e| crate::Error::Gguf(format!("input error: {e}")))?;

    let dry_run: bool = Confirm::new()
        .with_prompt("Dry run (show plan only)?")
        .default(false)
        .interact()
        .map_err(|e| crate::Error::Gguf(format!("confirm error: {e}")))?;

    Ok(PipelineStep::MoE {
        selection: None,
        strategy: Some(strategy),
        out: if out.is_empty() { None } else { Some(PathBuf::from(&out)) },
        dry_run: Some(dry_run),
    })
}

// ---------------------------------------------------------------------------
// Pipeline execution
// ---------------------------------------------------------------------------

fn execute_pipeline(cfg: &PipelineConfig) -> Result<(), crate::Error> {
    use crate::{run_analyze, run_moe, run_prune, run_quant, run_merge, run_svd};
    let mut model = cfg.model.clone();

    for (i, step) in cfg.steps.iter().enumerate() {
        eprintln!("\n[pipeline] step {}/{}: {}", i + 1, cfg.steps.len(), step.summary());
        match step {
            PipelineStep::Analyze { sample, json, report, template } => {
                let m = model.as_ref()
                    .ok_or_else(|| crate::Error::Gguf("pipeline: no model set for analyze step".into()))?;
                run_analyze(m, sample.unwrap_or(200_000), json.unwrap_or(false),
                    report.as_ref().map(|v| &**v), template.as_ref().map(|v| &**v), false)?;
            }
            PipelineStep::Quant { quant_type, out, verify } => {
                let m = model.as_ref()
                    .ok_or_else(|| crate::Error::Gguf("pipeline: no model set for quant step".into()))?;
                run_quant(m, quant_type, out, "", verify.unwrap_or(false))?;
                model = Some(out.clone());
            }
            PipelineStep::Prune { selection, out, verify } => {
                let m = model.as_ref()
                    .ok_or_else(|| crate::Error::Gguf("pipeline: no model set for prune step".into()))?;
                run_prune(m, selection, out, verify.unwrap_or(false), true)?;
                model = Some(out.clone());
            }
            PipelineStep::Svd { out, selection, rank, dry_run, verify } => {
                let m = model.as_ref()
                    .ok_or_else(|| crate::Error::Gguf("pipeline: no model set for svd step".into()))?;
                let _ = dry_run;
                // Build CLI-style args from pipeline fields
                let layers = "all";
                let tensors = selection.as_deref().unwrap_or("mlp");
                let rank_str = rank.as_deref().unwrap_or("0.5,min:4");
                let dtype = "f16";
                let backend = "faer";
                run_svd(m, layers, tensors, rank_str, dtype, backend, 16, None, out, verify.unwrap_or(false), false)?;
                model = Some(out.clone());
            }
            PipelineStep::Merge { models, out, weights, slerp, verify: _verify } => {
                let first = models.first()
                    .ok_or_else(|| crate::Error::Gguf("pipeline: merge requires at least one model".into()))?;
                if models.len() > 2 {
                    eprintln!("[warn] pipeline merge currently supports at most 2 models; using first two");
                }
                let mode = if let Some(t) = slerp { format!("slerp:{t}") } else { "average".into() };
                let weights_str = weights.as_ref()
                    .map(|w| w.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
                run_merge(first, models.get(1).map(|v| &**v), &mode,
                    weights_str.as_deref(), None, out, true)?;
                model = Some(out.clone());
            }
            PipelineStep::MoE { strategy, out, dry_run, .. } => {
                let m = model.as_ref()
                    .ok_or_else(|| crate::Error::Gguf("pipeline: no model set for moe step".into()))?;
                let s = strategy.as_deref().unwrap_or("average");
                let o = out.as_ref().map(|p| p.as_path()).unwrap_or_else(|| m);
                let _ = dry_run;
                run_moe(m, s, o, None, true)?;
                if out.is_some() {
                    model = out.clone();
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

const BANNER: &str = r#"
╔══════════════════════════════════════════════╗
║         TensorKit Interactive Mode           ║
║   Analyze · Prune · SVD · Quant · Merge · MoE ║
╚══════════════════════════════════════════════╝
"#;
