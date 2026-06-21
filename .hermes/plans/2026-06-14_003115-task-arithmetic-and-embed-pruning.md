# Task Arithmetic + Embedding Pruning — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add two new tensor-surgery operations to TensorKit — task vector arithmetic (with TIES-merge and a multiplier hub) and embedding-row pruning — following the flat directory convention (`merge/task_arith.rs`, `prune/embed.rs`).

**Architecture:** Two independent library modules (no I/O, pure `&[f32]` / `&dyn Model` functions), then CLI + pipeline wiring. All operations are training-free and operate on released model files. The "multiplier hub" is a lightweight registry that pairs task vectors with configurable `α` scalars and composites them in one pass.

**Tech Stack:** Rust 2024 edition, `rayon` (already a dependency) for parallel map, `regex` (already a dependency) for token matching. No new crate dependencies.

---

## Background — What These Algorithms Are

### Task Vectors (Ilharco et al., 2022 — arXiv 2212.04089)

A **task vector** is the elementwise difference between a fine-tuned model and its base:

```
τ = θ_finetuned − θ_base
```

Applying it to a base model:

```
θ_new = θ_base + α · τ        (α scales the "task strength")
```

Key operations:
- **Add** a task vector (apply a skill)
- **Subtract** a task vector ("unlearn" something)
- **Compose** multiple tasks: `θ_base + α_A · τ_A + α_B · τ_B`

### TIES-Merging (Yadav et al., 2023 — arXiv 2306.01708)

When merging multiple task vectors, sign conflicts and small-delta noise degrade quality. TIES fixes this in three steps:

```
1. TRIM:     For each task vector, keep only the top-K% by |τ| (zero the rest)
2. ELECT:    Per-coordinate, pick the majority sign across all task vectors
3. MERGE:    Average only the values whose sign agrees with the elected sign
```

Result: θ_new = θ_base + α · TIES_merge({τ_k})

Typical results: 2–5% absolute accuracy retention vs. naive averaging.

### Embedding Pruning

The token embedding matrix (`token_embd.weight`, shape `[vocab_size, hidden_dim]`) is often the largest single tensor. When the user only needs a subset of the vocabulary (e.g., English-only, code-only, a specific language), unused token rows can be deleted, saving significant memory. The row indices to keep are determined by:

- A vocabulary file (e.g., `tokenizer.json`, a word list)
- A language/pattern specification (regex matching token strings)
- A frequency file (top-N most-used tokens)

After pruning, all other tensors that depend on `vocab_size` (typically `output.weight` / `lm_head.weight`, and sometimes `token_embd.type`) must be resized consistently, or the user must be warned.

---

## Dependency Graph

```
Task 1 (task_arith.rs)  ─┐
                          ├─→ Task 3 (lib.rs wiring) ─→ Task 5 (CLI) ─→ Task 7 (pipeline)
Task 2 (embed.rs)  ──────┘
                                    └─→ Task 6 (unit tests)
```

## Execution Strategy

**Wave 1** (parallel — no dependencies): Tasks 1 and 2.
**Build gate:** `cargo check -p tensorkit`
**Wave 2** (sequential): Task 3 (module wiring), then Tasks 5+6 (CLI + tests) in parallel.
**Build gate:** `cargo test -p tensorkit`
**Wave 3** (sequential): Task 7 (pipeline integration).
**Final:** Full `cargo test --workspace` + `cargo clippy`.

---

## Wave 1 — Core Library Modules (Parallel)

### Task 1: Task Arithmetic Module

**Objective:** Create `lib/src/merge/task_arith.rs` — task vector computation, application, TIES-merge, and a multiplier hub registry.

**Files:**
- Create: `lib/src/merge/task_arith.rs`

**Module surface (all `pub` functions, zero I/O, pure `&[f32]`):**

```rust
// --- Task Vectors ---

/// Compute a single task vector: τ = θ_finetuned − θ_base.
/// Panics if a.len() != b.len().
pub fn compute_task_vector(base: &[f32], finetuned: &[f32]) -> Vec<f32>;

/// Apply a task vector: θ_out = θ_base + α · τ.
/// Writes into `out` (must be same length as `base` and `tau`).
pub fn apply_task_vector(out: &mut [f32], base: &[f32], tau: &[f32], alpha: f32);

/// Apply a task vector, returning a new Vec.
pub fn apply_task_vector_owned(base: &[f32], tau: &[f32], alpha: f32) -> Vec<f32>;

// --- TIES-Merging ---

/// Configuration for the TIES trim/elect/merge pipeline.
#[derive(Debug, Clone)]
pub struct TiesConfig {
    /// Fraction of coordinates to keep per task vector (e.g. 0.2 = top 20%).
    pub density: f64,
    /// If true, resolve sign conflicts by majority vote before merging.
    /// If false, skip elect-sign (equivalent to naive trimmed average).
    pub elect_sign: bool,
}

/// Trim a task vector: keep only the top-`density` fraction by magnitude,
/// zero out the rest.
pub fn trim_task_vector(tau: &[f32], density: f64) -> Vec<f32>;

/// Elect sign per coordinate across N task vectors. Returns a Vec of
/// signs (+1.0 or −1.0) of length `dim`.
///
/// `task_vectors` is a slice of N trimmed task vectors, each of length `dim`.
pub fn elect_sign(task_vectors: &[&[f32]]) -> Vec<f32>;

/// Merge N task vectors using the TIES algorithm (trim → elect → average
/// only agreeing signs). Returns the merged task vector of length `dim`.
pub fn ties_merge(task_vectors: &[&[f32]], config: &TiesConfig) -> Vec<f32>;

// --- Multiplier Hub ---

/// A named task vector with a scalar multiplier.
#[derive(Debug, Clone)]
pub struct TaskEntry {
    pub name: String,
    pub vector: Vec<f32>,
    pub alpha: f32,
}

/// The multiplier hub: a collection of named task vectors that can be
/// composited onto a base model in one pass.
#[derive(Debug, Clone)]
pub struct MultiplierHub {
    pub entries: Vec<TaskEntry>,
}

impl MultiplierHub {
    pub fn new() -> Self;

    /// Register a task vector with a given multiplier.
    pub fn add(&mut self, name: impl Into<String>, vector: Vec<f32>, alpha: f32);

    /// Remove a registered task vector by name.
    pub fn remove(&mut self, name: &str) -> bool;

    /// Update the multiplier for an existing task vector.
    pub fn set_alpha(&mut self, name: &str, alpha: f32) -> bool;

    /// List all registered task vectors and their multipliers.
    pub fn list(&self) -> &[(String, f32)];

    /// Apply all registered task vectors to a base tensor.
    /// θ_out = θ_base + Σ (α_k · τ_k)
    /// Writes into `out` (must be same length as base).
    pub fn apply(&self, out: &mut [f32], base: &[f32]);

    /// Apply all registered task vectors, returning a new Vec.
    pub fn apply_owned(&self, base: &[f32]) -> Vec<f32>;

    /// Composite: first TIES-merge all registered vectors, then apply.
    /// This is equivalent to: θ_out = θ_base + α_total · ties_merge(τ_k)
    /// where α_total is a global scale applied to the merged result.
    pub fn apply_ties(&self, base: &[f32], config: &TiesConfig) -> Vec<f32>;
}
```

**Step 1: Write failing tests**

Create test file `lib/tests/unit/merge/task_arith.rs`:

```rust
use super::*;

#[test]
fn compute_task_vector_basic() {
    let base = [1.0, 2.0, 3.0];
    let ft = [1.5, 2.5, 2.0];
    let tau = compute_task_vector(&base, &ft);
    assert_eq!(tau, vec![0.5, 0.5, -1.0]);
}

#[test]
fn apply_task_vector_zero_alpha() {
    let base = [1.0, 2.0, 3.0];
    let tau = [10.0, 10.0, 10.0];
    let result = apply_task_vector_owned(&base, &tau, 0.0);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn apply_task_vector_alpha_one() {
    let base = [1.0, 2.0, 3.0];
    let tau = [0.5, 0.5, -1.0];
    let result = apply_task_vector_owned(&base, &tau, 1.0);
    assert_eq!(result, vec![1.5, 2.5, 2.0]);
}

#[test]
fn apply_task_vector_alpha_half() {
    let base = [0.0, 0.0, 0.0];
    let tau = [2.0, 4.0, 6.0];
    let result = apply_task_vector_owned(&base, &tau, 0.5);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn trim_top_50_percent() {
    let tau = [1.0, 2.0, 3.0, 4.0]; // |τ|: 1,2,3,4 → keep top 2: values 3,4
    let trimmed = trim_task_vector(&tau, 0.5);
    assert_eq!(trimmed, vec![0.0, 0.0, 3.0, 4.0]);
}

#[test]
fn elect_sign_majority_positive() {
    let a = [1.0, -2.0, 3.0];
    let b = [0.5, -1.0, 4.0];
    let signs = elect_sign(&[&a, &b]);
    assert_eq!(signs, vec![1.0, -1.0, 1.0]);
}

#[test]
fn elect_sign_tiebreak_positive() {
    let a = [1.0, 0.0];
    let b = [-1.0, 0.0];
    let signs = elect_sign(&[&a, &b]);
    // tie on coord 0 (1 vs -1), zero on coord 1 → both resolve to +1.0
    assert_eq!(signs, vec![1.0, 1.0]);
}

#[test]
fn ties_merge_basic() {
    let a = [10.0, 0.1, -5.0];
    let b = [-8.0, 0.2, -4.0];
    let config = TiesConfig { density: 0.67, elect_sign: true };
    let merged = ties_merge(&[&a, &b], &config);
    // density 0.67 → keep top 2 of 3 per vector
    // a trimmed: [10.0, 0.0, -5.0]
    // b trimmed: [-8.0, 0.0, -4.0]
    // elect sign: [+, -, -] (coord 0: + wins, coord 1: both zero → +, coord 2: both − → −)
    // merged (avg of agreeing): [(10 + −8)/2=1, 0.0, (−5 + −4)/2=−4.5]
    assert_eq!(merged.len(), 3);
    assert!((merged[0] - 1.0).abs() < 1e-5);
    assert!(merged[1].abs() < 1e-5);
    assert!((merged[2] - (-4.5)).abs() < 1e-5);
}

#[test]
fn multiplier_hub_basic() {
    let base = [0.0, 0.0, 0.0];
    let tau1 = [1.0, 2.0, 3.0];
    let tau2 = [3.0, 2.0, 1.0];
    let mut hub = MultiplierHub::new();
    hub.add("task_a", tau1, 0.5);
    hub.add("task_b", tau2, 0.5);
    let result = hub.apply_owned(&base);
    assert_eq!(result, vec![2.0, 2.0, 2.0]); // 0.5*1 + 0.5*3, etc.
}

#[test]
fn multiplier_hub_remove() {
    let mut hub = MultiplierHub::new();
    hub.add("a", vec![1.0], 1.0);
    assert!(hub.remove("a"));
    assert!(!hub.remove("nonexistent"));
    assert!(hub.list().is_empty());
}

#[test]
fn multiplier_hub_set_alpha() {
    let mut hub = MultiplierHub::new();
    hub.add("a", vec![10.0], 1.0);
    assert!(hub.set_alpha("a", 2.0));
    assert!(!hub.set_alpha("nonexistent", 1.0));
    let result = hub.apply_owned(&[0.0]);
    assert_eq!(result, vec![20.0]);
}
```

**Step 2: Run tests — expect failures**
```bash
cd "C:\Users\John\Desktop\AI Gens\Rust\TensorKit" && cargo test --lib --test unit_test -- merge::task_arith 2>&1 | tail -20
```
Expected: compilation errors (module not found).

**Step 3: Implement `lib/src/merge/task_arith.rs`**

Complete implementation with all functions above. Key implementation notes:
- `trim_task_vector`: build index list, sort by `|τ|` descending, zero out everything below the top-K threshold.
- `elect_sign`: per-coordinate, count positive vs negative, resolve ties by defaulting to `+1.0`.
- `ties_merge`: call trim on each, elect_sign, then for each coordinate keep only values with matching sign, average them, multiply by `(1/N)` or `(K_used / N_total)` for the merged result.
- `MultiplierHub`: simple `Vec<TaskEntry>` storage. `apply` does one pass: `out[i] = base[i] + Σ(entry.alpha * entry.vector[i])`.

**Step 4: Run tests — expect pass**
```bash
cargo test --lib --test unit_test -- merge::task_arith 2>&1 | tail -15
```
Expected: all tests pass.

**Step 5: Commit**
```bash
git add lib/src/merge/task_arith.rs lib/tests/unit/merge/task_arith.rs
git commit -m "feat: add task arithmetic module (task vectors, TIES-merge, multiplier hub)"
```

---

### Task 2: Embedding Pruning Module

**Objective:** Create `lib/src/prune/embed.rs` — prune rows from the vocab embedding tensor based on token selection criteria.

**Files:**
- Create: `lib/src/prune/embed.rs`

**Module surface:**

```rust
use crate::error::Result;
use crate::model::Model;

/// How to select which token rows to keep.
#[derive(Debug, Clone)]
pub enum TokenSelection {
    /// Keep only tokens whose IDs are in this list.
    ById(Vec<u32>),
    /// Keep tokens whose string representation matches this regex.
    ByPattern(regex::Regex),
    /// Keep the first N token rows (0..N).
    TopN(usize),
    /// Keep rows listed in an external file (one token string per line).
    ByFile(std::path::PathBuf),
}

/// Plan for embedding pruning: describes which rows to keep and the
/// mapping from old row indices to new row indices.
#[derive(Debug, Clone)]
pub struct EmbedPrunePlan {
    /// Sorted list of original row indices to keep.
    pub keep_rows: Vec<u32>,
    /// Mapping: old_index → new_index (only for kept rows).
    pub remap: std::collections::HashMap<u32, u32>,
    /// Original vocab size.
    pub original_vocab_size: u32,
    /// New vocab size.
    pub new_vocab_size: u32,
    /// Name of the embedding tensor (e.g. "token_embd.weight").
    pub embed_tensor_name: String,
    /// Name of the output projection tensor (e.g. "output.weight"), if present.
    pub output_tensor_name: Option<String>,
}

/// Build a plan: analyze the model's vocab tensor and token metadata to
/// determine which rows to keep.
///
/// `vocab_tokens` provides the string representation of each token index.
/// If `None`, the function falls back to the model's tokenizer metadata
/// (if available) or returns an error for pattern/file-based selections.
pub fn plan_embed_prune(
    model: &dyn Model,
    selection: &TokenSelection,
    vocab_tokens: Option<&[String]>,
) -> Result<EmbedPrunePlan>;

/// Apply the plan: read the embedding tensor, write a new one with only
/// the kept rows. Also remaps the output projection tensor if present.
///
/// Returns a list of (tensor_name, new_bytes) pairs for the caller to write.
pub fn apply_embed_prune<M: Model + ?Sized>(
    model: &M,
    plan: &EmbedPrunePlan,
) -> Result<Vec<(String, Vec<u8>)>>;
```

**Step 1: Write failing tests**

Create test file `lib/tests/unit/prune/embed.rs`:

```rust
use super::*;

#[test]
fn plan_by_id_basic() {
    // Mock model with 4-row embedding
    let model = MockEmbedModel::new(4, 8);
    let selection = TokenSelection::ById(vec![0, 2, 3]);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    assert_eq!(plan.keep_rows, vec![0, 2, 3]);
    assert_eq!(plan.original_vocab_size, 4);
    assert_eq!(plan.new_vocab_size, 3);
    assert_eq!(plan.remap.get(&0), Some(&0));
    assert_eq!(plan.remap.get(&2), Some(&1));
    assert_eq!(plan.remap.get(&3), Some(&2));
}

#[test]
fn plan_top_n() {
    let model = MockEmbedModel::new(10, 8);
    let selection = TokenSelection::TopN(3);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    assert_eq!(plan.keep_rows, vec![0, 1, 2]);
    assert_eq!(plan.new_vocab_size, 3);
}

#[test]
fn plan_by_pattern() {
    let model = MockEmbedModel::new(4, 8);
    let vocab = vec![
        "hello".into(), "world".into(), "foo".into(), "bar".into(),
    ];
    let selection = TokenSelection::ByPattern(
        regex::Regex::new("^(hello|bar)$").unwrap(),
    );
    let plan = plan_embed_prune(&model, &selection, Some(&vocab)).unwrap();
    assert_eq!(plan.keep_rows, vec![0, 3]);
    assert_eq!(plan.new_vocab_size, 2);
}

#[test]
fn apply_prune_produces_correct_sizes() {
    let model = MockEmbedModel::new(4, 8);
    let selection = TokenSelection::ById(vec![0, 2]);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    let result = apply_embed_prune(&model, &plan).unwrap();
    // Embedding tensor: 2 rows × 8 cols × 4 bytes (f32) = 64 bytes
    let embed = result.iter().find(|(n, _)| n == "token_embd.weight").unwrap();
    assert_eq!(embed.1.len(), 2 * 8 * 4);
    // Output tensor: 2 rows × 8 cols × 4 bytes = 64 bytes
    let output = result.iter().find(|(n, _)| n == "output.weight").unwrap();
    assert_eq!(output.1.len(), 2 * 8 * 4);
}

// --- Test helper: a minimal mock implementing Model for embed prune tests ---

struct MockEmbedModel {
    vocab_size: u32,
    hidden_dim: u32,
}

impl MockEmbedModel {
    fn new(vocab_size: u32, hidden_dim: u32) -> Self {
        Self { vocab_size, hidden_dim }
    }
}

impl crate::model::Model for MockEmbedModel {
    fn format(&self) -> crate::model::ModelFormat { crate::model::ModelFormat::Unknown }
    fn name(&self) -> Option<&str> { None }
    fn architecture(&self) -> Option<&str> { None }
    fn block_count(&self) -> Option<usize> { None }
    fn tensors(&self) -> &[crate::model::Tensor] {
        // static — can't return self-referencing data; use a leak-based approach
        // or return a pre-built slice. For tests, we'll use a different pattern:
        // see actual implementation note below.
        unimplemented!("mock tensors via separate helper")
    }
    fn tensor(&self, name: &str) -> Option<&crate::model::Tensor> {
        match name {
            "token_embd.weight" | "output.weight" => Some(&crate::model::Tensor {
                name: name.to_string(),
                dtype: crate::model::TensorDtype::F32,
                shape: vec![self.vocab_size as u64, self.hidden_dim as u64],
                byte_size: (self.vocab_size * self.hidden_dim * 4) as u64,
                data_offset: 0,
            }),
            _ => None,
        }
    }
    fn metadata(&self, _key: &str) -> Option<crate::model::MetadataValue<'_>> { None }
    fn read_tensor_bytes(&self, name: &str) -> crate::Result<std::borrow::Cow<'_, [u8]>> {
        let nelem = (self.vocab_size * self.hidden_dim) as usize;
        let data: Vec<u8> = (0..nelem)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        // Leak the data to get a 'static reference — OK for tests
        let data: &'static [u8] = Box::leak(data.into_boxed_slice());
        Ok(std::borrow::Cow::Borrowed(data))
    }
}
```

**Step 2: Run tests — expect failures**
```bash
cargo test --lib --test unit_test -- prune::embed 2>&1 | tail -20
```
Expected: compilation errors.

**Step 3: Implement `lib/src/prune/embed.rs`**

Key implementation notes:
- `plan_embed_prune` takes a `&dyn Model` and resolves `TokenSelection` to a concrete `Vec<u32>` of keep-indices.
  - `ById` → sort and validate
  - `TopN` → `(0..N).collect()`
  - `ByPattern` → iterate `vocab_tokens`, match regex, collect matching indices
  - `ByFile` → read lines, build a `HashSet<String>`, iterate vocab, collect matches
- `apply_embed_prune` reads the embedding tensor bytes via `model.read_tensor_bytes()`, interprets as `&[f32]` (row-major `[vocab, hidden]`), then writes only the kept rows as a new `Vec<u8>`.
  - If the output projection tensor exists, it is also remapped (columns of the output weight = rows of the embedding in most architectures, but verify via tensor shape convention).
  - Returns `Vec<(String, Vec<u8>)>` — same pattern as `prune::apply_to_gguf`.

**Step 4: Run tests — expect pass**
```bash
cargo test --lib --test unit_test -- prune::embed 2>&1 | tail -15
```
Expected: all tests pass.

**Step 5: Commit**
```bash
git add lib/src/prune/embed.rs lib/tests/unit/prune/embed.rs
git commit -m "feat: add embedding pruning module (token selection, row removal)"
```

---

## Wave 2 — Module Wiring + CLI (Sequential)

### Task 3: Wire Modules Into lib.rs

**Objective:** Register the new modules, update re-exports, and update `merge/mod.rs` + `prune/mod.rs`.

**Files:**
- Modify: `lib/src/merge/mod.rs` — add `pub mod task_arith;` and re-exports
- Modify: `lib/src/prune/mod.rs` — add `pub mod embed;` and re-exports
- Modify: `lib/src/lib.rs` — add to the `pub use merge::{...}` and `pub use prune::{...}` blocks

**Step 1: Edit `merge/mod.rs`**

Add after existing module declarations:

```rust
mod task_arith;

pub use task_arith::{
    apply_task_vector, apply_task_vector_owned, compute_task_vector,
    elect_sign, ties_merge, trim_task_vector, MultiplierHub, TaskEntry,
    TiesConfig,
};
```

**Step 2: Edit `prune/mod.rs`**

Add after existing module declarations:

```rust
pub mod embed;

pub use embed::{plan_embed_prune, apply_embed_prune, EmbedPrunePlan, TokenSelection};
```

**Step 3: Edit `lib.rs`**

Add to the `pub use merge::{...}` block:
```
    apply_task_vector, apply_task_vector_owned, compute_task_vector,
    elect_sign, ties_merge, trim_task_vector, MultiplierHub, TaskEntry,
    TiesConfig,
```

Add to the `pub use prune::{...}` block:
```
    plan_embed_prune, apply_embed_prune, EmbedPrunePlan, TokenSelection,
```

**Step 4: Build check**
```bash
cargo check -p tensorkit 2>&1
```
Expected: clean.

**Step 5: Commit**
```bash
git add lib/src/lib.rs lib/src/merge/mod.rs lib/src/prune/mod.rs
git commit -m "feat: wire task_arith and embed prune modules into lib exports"
```

---

### Task 4: CLI Subcommands

**Objective:** Add two new CLI subcommands: `task` (task vector operations) and extend `prune` with an `--embed` flag.

**Files:**
- Modify: `cli/src/main.rs` — add `Commands::Task` variant and `run_task()` function
- Modify: `cli/src/main.rs` — add `--embed` / `--token-file` flags to existing `Commands::Prune`
- Modify: `cli/src/main.rs` — add `run_prune_embed()` function

**New `Commands::Task` variant:**

```rust
/// Task-vector arithmetic: compute, apply, compose, or TIES-merge
Task {
    /// Path to base model
    #[arg(long)]
    base: PathBuf,

    /// Path to fine-tuned model (for computing task vector)
    #[arg(long)]
    finetuned: Option<PathBuf>,

    /// Path to pre-computed task vector file (JSON: { "name": [f32...] })
    #[arg(long)]
    vector: Option<PathBuf>,

    /// Sub-action: "compute", "apply", "ties-merge"
    #[arg(long, default_value = "apply")]
    action: String,

    /// Multiplier / alpha (for "apply" action)
    #[arg(long, default_value_t = 1.0)]
    alpha: f32,

    /// Path to target model to apply vector to (for "apply" action)
    #[arg(long)]
    target: Option<PathBuf>,

    /// Output path
    #[arg(long, short = 'o')]
    out: PathBuf,

    /// TIES density (for "ties-merge" action, fraction to keep)
    #[arg(long, default_value_t = 0.2)]
    ties_density: f64,

    /// Skip confirmation
    #[arg(long, short = 'y')]
    yes: bool,
},
```

**`run_task()` function** (~80 lines): dispatches based on `action`:
- `"compute"`: opens base + finetuned, reads tensors, computes τ per tensor, writes JSON
- `"apply"`: opens target, reads task vector JSON, applies with α, writes output
- `"ties-merge"`: loads N task vector files, runs TIES algorithm, applies result

**Extend `Commands::Prune`:**

```rust
    // Add to existing Prune variant:
    /// Prune embedding rows for unused tokens (use with --token-file)
    #[arg(long)]
    embed: bool,

    /// Path to token file (one token per line) for embedding pruning
    #[arg(long)]
    token_file: Option<PathBuf>,

    /// Regex pattern for token selection in embedding pruning
    #[arg(long)]
    token_pattern: Option<String>,

    /// Keep only top N token rows
    #[arg(long)]
    token_top_n: Option<usize>,
```

**`run_prune_embed()` function** (~60 lines): opens model, builds `TokenSelection` from flags, calls `plan_embed_prune` + `apply_embed_prune`, writes output.

**Step 1: Implement CLI changes**

**Step 2: Build check**
```bash
cargo check -p tensorkit-cli 2>&1
```
Expected: clean.

**Step 3: Smoke test**
```bash
cargo run -- --help 2>&1
```
Expected: `task` subcommand appears in help text.

**Step 4: Commit**
```bash
git add cli/src/main.rs
git commit -m "feat: add 'task' CLI subcommand and embed pruning flags to 'prune'"
```

---

### Task 5: Unit Tests for CLI + Integration

**Objective:** Add integration tests for the new CLI subcommands and verify end-to-end flows.

**Files:**
- Create: `cli/tests/cli_task.rs` — CLI integration test for task vector operations
- Modify: `lib/tests/unit/merge/tests.rs` — add `mod task_arith;` module declaration

**Step 1: Create `cli/tests/cli_task.rs`**

Test structure (RAII cleanup pattern from `rust-refactoring-patterns.md`):
- Test `task compute` with mock GGUF files
- Test `task apply` with pre-computed vector
- Test `prune --embed --token-file` with a small embedding

Note: these tests need small test GGUF fixtures. Use the existing test GGUF infrastructure if present, or create minimal in-memory test models.

**Step 2: Verify module test registration**

Check `lib/tests/unit/merge/tests.rs` includes `mod task_arith;`:
```rust
// existing content should already have:
// mod average; mod depth; mod moe; mod slerp; mod strategy; mod tying;
// Add:
mod task_arith;
```

Check `lib/tests/unit/tests/mod.rs` includes the prune embed module (or register it).

**Step 3: Run all tests**
```bash
cargo test --workspace 2>&1 | tail -30
```
Expected: all pass.

**Step 4: Commit**
```bash
git add cli/tests/cli_task.rs lib/tests/unit/merge/tests.rs
git commit -m "test: add unit + integration tests for task arithmetic and embed pruning"
```

---

## Wave 3 — Pipeline Integration

### Task 6: Pipeline Step Variants

**Objective:** Add `TaskArith` and `PruneEmbed` variants to `PipelineStep` so both operations can be composed in JSON pipeline configs.

**Files:**
- Modify: `cli/src/pipeline.rs` — add enum variants + prompt functions + execution dispatch

**Add to `PipelineStep` enum:**

```rust
TaskArith {
    base: Option<PathBuf>,
    finetuned: Option<PathBuf>,
    vector: Option<PathBuf>,
    action: String,
    alpha: Option<f32>,
    ties_density: Option<f64>,
    out: PathBuf,
    verify: Option<bool>,
},
PruneEmbed {
    token_file: Option<PathBuf>,
    token_pattern: Option<String>,
    token_top_n: Option<usize>,
    out: PathBuf,
    verify: Option<bool>,
},
```

**Add prompt functions** `prompt_task_arith()` and `prompt_prune_embed()` following the pattern of existing `prompt_merge()` / `prompt_moe()`.

**Add to `execute_pipeline()` match block:**

```rust
PipelineStep::TaskArith { .. } => { /* dispatch run_task */ }
PipelineStep::PruneEmbed { .. } => { /* dispatch run_prune_embed */ }
```

**Add to `PipelineStep::summary()`:**

```rust
PipelineStep::TaskArith { action, out, .. } => {
    format!("TaskArith ({action}) → {}", out.display())
}
PipelineStep::PruneEmbed { out, .. } => {
    format!("PruneEmbed → {}", out.display())
}
```

**Step 1: Implement all changes**

**Step 2: Build check**
```bash
cargo check -p tensorkit-cli 2>&1
```

**Step 3: Verify interact mode shows new options**
```bash
cargo run -- interact 2>&1
```
Expected: "Task Arithmetic" and "Embedding Pruning" appear in step menu.

**Step 4: Commit**
```bash
git add cli/src/pipeline.rs
git commit -m "feat: add TaskArith and PruneEmbed pipeline steps"
```

---

### Task 7: Final Verification + Documentation

**Objective:** Run full test suite, clippy, and verify the help text is clean.

**Step 1: Full build**
```bash
cargo build --workspace 2>&1
```

**Step 2: Full test suite**
```bash
cargo test --workspace 2>&1 | tail -40
```
Expected: all pass.

**Step 3: Clippy**
```bash
cargo clippy --workspace 2>&1 | grep -E "^(warning|error)" | head -20
```
Expected: no warnings or errors.

**Step 4: Verify CLI help text**
```bash
cargo run -- task --help 2>&1
cargo run -- prune --help 2>&1
```

**Step 5: Commit**
```bash
git add -A
git commit -m "chore: final verification, clippy clean, help text confirmed"
```

---

## Error Workarounds — Complete Failure-Mode Matrix

Every error below has: what triggers it, what the user sees, and the code-level workaround. Error variants follow the existing convention (`Error::TaskArith(String)`, `Error::EmbedPrune(String)` — add these to `lib/src/error.rs`).

### New Error Variants (add to `error.rs`)

```rust
#[error("task arithmetic error: {0}")]
TaskArith(String),

#[error("embedding prune error: {0}")]
EmbedPrune(String),
```

---

### TASK ARITHMETIC — Failure Modes

#### TA-1: Length mismatch between base and finetuned tensors

**Trigger:** `compute_task_vector(base, ft)` where `base.len() != ft.len()`.
**Cause:** Models have different architectures, or tensor was resized between saves.
**Current behavior:** `assert!` panics (like `average_tensors`).
**Workaround:** Replace panic with `Result`. Return `Error::TaskArith`:
```rust
pub fn compute_task_vector(base: &[f32], finetuned: &[f32]) -> Result<Vec<f32>> {
    if base.len() != finetuned.len() {
        return Err(Error::TaskArith(format!(
            "length mismatch: base has {} elements, finetuned has {}",
            base.len(), finetuned.len()
        )));
    }
    // ... proceed
}
```
**CLI impact:** `run_task` catches this and prints `error: task arithmetic error: length mismatch...` then exits 1.

#### TA-2: Tensor exists in one model but not the other

**Trigger:** `task compute --base A.gguf --finetuned B.gguf` where B has a tensor A doesn't (or vice versa).
**Cause:** Different architectures, or one model was already pruned.
**Workaround:** Skip missing tensors with a warning, continue with the intersection. At the end, report:
```
[warn] skipped 3 tensors not found in both models: lm_head.weight, embed_out.weight, tok_embeddings.bias
[ok] computed task vectors for 197 tensors (2.1 GB total)
```
**Implementation:** In `run_task` (compute action), build a `HashSet<&str>` intersection of tensor names from both models. Only iterate the intersection. Emit `eprintln!` for each skipped tensor.

#### TA-3: Non-f32 tensor dtype (quantized, f16, bf16)

**Trigger:** Either model has quantized tensors (Q4_0, Q8_0, etc.) or f16/bf16.
**Cause:** GGUF models are often quantized; safetensors may use f16.
**Workaround:** Always dequantize to f32 before computing the task vector. The existing `dequantize(ty, bytes, max_elems)` function handles all GGML types. For f16/bf16, use `scan_f16`/`scan_bf16` from `formats::gguf::dequant::scalar`.
```rust
fn to_f32_vec(tensor: &Tensor, model: &dyn Model) -> Result<Vec<f32>> {
    let raw = model.read_tensor_bytes(&tensor.name)?;
    match tensor.dtype {
        TensorDtype::F32 => Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
        TensorDtype::F16 => Ok(scan_f16(&raw)),
        TensorDtype::Bf16 => Ok(scan_bf16(&raw)),
        other => {
            // Try GGML dequantize
            let ggml_ty = tensor_dtype_to_ggml(other)?;
            dequantize(ggml_ty, &raw, None)
                .ok_or_else(|| Error::TaskArith(format!(
                    "tensor '{}': cannot dequantize type {}", tensor.name, other.as_str()
                )))
        }
    }
}
```
**Output dtype:** The resulting task vector is always `Vec<f32>`. When applied to the target model, the output is written as f32 (or re-quantized if the user chains with `--quant` in a pipeline).

#### TA-4: NaN / Inf in task vectors

**Trigger:** `θ_ft - θ_base` produces NaN or Inf (overflow, corrupted weights, or diverged fine-tune).
**Cause:** Fine-tuning instability, weight corruption, or very large deltas.
**Workaround:** Two-pronged:
1. **Detection:** After computing each task vector, scan for NaN/Inf. If found, report and offer `--clamp` flag:
   ```
   [warn] tensor 'blk.3.attn_q.weight': task vector contains 12 NaN values (of 262144)
   [hint] use --clamp to replace NaN/Inf with 0.0 before applying
   ```
2. **Clamp mode:** `--clamp` replaces NaN with 0.0 and Inf with ±1e6 before writing the task vector file. This is a lossy rescue — the user is warned.
3. **Without --clamp:** If NaN/Inf detected, return `Error::TaskArith(format!("tensor '{}': {} NaN/Inf values in task vector (use --clamp to proceed)", ...))`.

**Implementation:**
```rust
fn scan_nan_inf(v: &[f32]) -> (usize, usize) {
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let inf = v.iter().filter(|x| x.is_infinite()).count();
    (nan, inf)
}

fn clamp_task_vector(v: &mut [f32]) {
    for x in v.iter_mut() {
        if x.is_nan() { *x = 0.0; }
        else if *x > 1e6 { *x = 1e6; }
        else if *x < -1e6 { *x = -1e6; }
    }
}
```

#### TA-5: Density parameter out of range

**Trigger:** `ties_merge` called with `density <= 0.0` or `density > 1.0`.
**Workaround:** Validate in `TiesConfig::new()` or at the top of `ties_merge`:
```rust
if !(0.0 < config.density && config.density <= 1.0) {
    return Err(Error::TaskArith(format!(
        "TIES density must be in (0, 1], got {}", config.density
    )));
}
```

#### TA-6: All-zero task vector after trim

**Trigger:** Density is very low (e.g., 0.01) and the task vector has mostly small values.
**Cause:** User set density too aggressive, or fine-tune barely changed the model.
**Workaround:** Not an error — warn and proceed. The merged result is effectively the base model. `eprintln!("[warn] task vector '{}' is all zeros after trim at density={}", name, density)`.

#### TA-7: Task vector JSON parse error

**Trigger:** Malformed JSON, wrong structure, missing keys.
**Workaround:** Already covered by `Error::Json`. Add a helpful hint in `run_task`:
```rust
let hub: MultiplierHub = serde_json::from_str(&json)
    .map_err(|e| Error::TaskArith(format!(
        "failed to parse task vector file: {}\n[hint] expected format: {{ \"vectors\": {{ \"name\": [f32...] }} }}",
        e
    )))?;
```

#### TA-8: Task vector tensors not in target model

**Trigger:** Task vector file references tensors that don't exist in the target model (e.g., computed from a different architecture).
**Workaround:** Skip with warning, continue with intersection:
```
[warn] task vector 'embed_out.weight' not found in target model — skipping
[ok] applied 195 of 198 task vectors to target
```

#### TA-9: Empty MultiplierHub

**Trigger:** `hub.apply()` or `hub.apply_ties()` called with zero entries.
**Workaround:** Not an error — return a copy of the base tensor. This is a valid no-op. `eprintln!("[info] hub has no registered vectors — output is identical to base")`.

#### TA-10: f32 overflow when scaling

**Trigger:** `alpha * tau[i]` overflows f32 (e.g., alpha=1000.0, tau=1e30).
**Workaround:** After each multiply, check `is_finite()`. If overflow:
```rust
let scaled = alpha * tau[i];
if !scaled.is_finite() {
    return Err(Error::TaskArith(format!(
        "overflow at element {}: alpha={} * tau={} = {}",
        i, alpha, tau[i], scaled
    )));
}
```
**Alternative (softer):** Clamp to ±f32::MAX instead of erroring. Make this configurable via `--clamp-overflow`.

#### TA-11: CLI flag conflicts

**Trigger:** `task --action compute` without `--finetuned`, or `task --action apply` without `--vector`.
**Workaround:** Validate in `run_task()` before doing any work:
```rust
match action.as_str() {
    "compute" => {
        let ft = finetuned.as_ref().ok_or_else(|| Error::TaskArith(
            "--finetuned is required for 'compute' action".into()
        ))?;
        // ...
    }
    "apply" => {
        let vec_path = vector.as_ref().ok_or_else(|| Error::TaskArith(
            "--vector is required for 'apply' action".into()
        ))?;
        // ...
    }
    other => return Err(Error::TaskArith(format!(
        "unknown action '{}': expected 'compute', 'apply', or 'ties-merge'", other
    ))),
}
```

#### TA-12: Output path equals input path

**Trigger:** `task --action apply --target model.gguf -o model.gguf`
**Workaround:** Check before writing:
```rust
if out == target {
    return Err(Error::TaskArith("output path must differ from input path".into()));
}
```

---

### EMBEDDING PRUNING — Failure Modes

#### EP-1: No embedding tensor found

**Trigger:** Model doesn't have `token_embd.weight`, `tok_embeddings.weight`, or `embed.weight`.
**Cause:** Non-standard architecture, or model uses a different naming convention.
**Workaround:** Try the same candidate list as `tying.rs` (`EMBED_CANDIDATES`). If none found, also scan all tensors for any with shape `[vocab, hidden]` where `vocab > 1000` (heuristic). If still nothing:
```
error: embedding prune error: no embedding tensor found (tried: token_embd.weight, tok_embeddings.weight, embed.weight)
[hint] use --embed-tensor-name to specify the tensor name manually
```
Add `--embed-tensor-name` flag to the CLI for manual override.

#### EP-2: Embedding tensor is quantized (Q4_0, Q8_0, etc.)

**Trigger:** GGUF model with quantized embeddings.
**Workaround:** Dequantize to f32, prune rows, write output as f32. Warn the user:
```
[warn] embedding tensor is Q4_0 — output will be f32 (larger file)
[hint] chain with --quant to re-quantize after pruning
```
The output is always f32 because row removal changes the tensor layout and re-quantization requires the full quantization pipeline.

#### EP-3: Embedding tensor is f16/bf16

**Trigger:** Safetensors model with f16 embeddings.
**Workaround:** Convert to f32 via `scan_f16`/`scan_f16`, prune, write as f32. Same warning as EP-2. Future: offer `--output-dtype f16` to convert back.

#### EP-4: Vocab size mismatch between embedding and output projection

**Trigger:** `token_embd.weight` has shape `[V1, H]` but `output.weight` has shape `[V2, H]` or `[H, V2]` where `V1 ≠ V2`.
**Cause:** Corrupted model, or the two tensors use different vocab sizes (e.g., after a partial edit).
**Workaround:** Two strategies:
1. **Strict (default):** Error if mismatched:
   ```
   error: embedding prune error: vocab size mismatch: token_embd.weight has 32000 rows, output.weight has 32001 columns
   [hint] use --force to prune only the embedding tensor
   ```
2. **--force:** Prune only the embedding tensor, leave output projection untouched. The model will be in an inconsistent state, but this is useful for debugging or when the user knows what they're doing.

**Implementation:**
```rust
fn validate_vocab_size(embed: &Tensor, output: Option<&Tensor>) -> Result<()> {
    let embed_vocab = embed.shape.first()
        .ok_or_else(|| Error::EmbedPrune("embedding tensor has no dimensions".into()))?;
    if let Some(out) = output {
        // output.weight is typically [hidden, vocab] — check last dim
        let out_vocab = out.shape.last()
            .ok_or_else(|| Error::EmbedPrune("output tensor has no dimensions".into()))?;
        if embed_vocab != out_vocab {
            return Err(Error::EmbedPrune(format!(
                "vocab size mismatch: embedding '{}' has {} rows, output '{}' has {} cols",
                embed.name, embed_vocab, out.name, out_vocab
            )));
        }
    }
    Ok(())
}
```

#### EP-5: Token file doesn't exist or is unreadable

**Trigger:** `--token-file nonexistent.txt`
**Workaround:** Already covered by `Error::Io`. Add a hint:
```
error: I/O error: No such file or directory (os error 2)
[hint] check the path to your token file
```

#### EP-6: Token file is empty

**Trigger:** `--token-file empty.txt` (0 lines).
**Workaround:** Error with a clear message:
```
error: embedding prune error: token file is empty (0 tokens)
[hint] provide a file with one token per line, or use --token-top-n to keep the first N tokens
```
Do NOT silently prune everything — that's almost never what the user wants.

#### EP-7: No tokens match the regex pattern

**Trigger:** `--token-pattern "^(xyz)$"` where no token matches.
**Workaround:** Error:
```
error: embedding prune error: pattern '^(xyz)$' matched 0 tokens out of 32000
[hint] check your regex, or use --token-file for exact token lists
```
No `--allow-empty` for v1 — pruning all tokens is never useful.

#### EP-8: TopN > vocab_size

**Trigger:** `--token-top-n 50000` on a 32000-token vocab.
**Workaround:** Clamp to vocab_size with a warning:
```
[warn] --token-top-n 50000 exceeds vocab size 32000 — clamping to 32000 (no pruning)
```
This is a no-op, which is correct — the user asked for more tokens than exist.

#### EP-9: ById contains indices >= vocab_size

**Trigger:** `--token-id 99999` on a 32000-token vocab.
**Workaround:** Filter out invalid indices with warnings:
```
[warn] token index 99999 >= vocab_size 32000 — skipping 3 invalid indices
[ok] keeping 29997 of 30000 requested tokens
```
Don't error — partial pruning is still useful.

#### EP-10: Token strings not available for pattern matching

**Trigger:** `--token-pattern ".*"` but the model has no tokenizer metadata and no `--token-file` was provided.
**Workaround:** Error with actionable guidance:
```
error: embedding prune error: pattern matching requires token strings
[hint] provide --token-file with one token per line, or use --token-id / --token-top-n instead
```

#### EP-11: Output projection tensor has different hidden_dim than embedding

**Trigger:** `token_embd.weight` has shape `[V, H1]` but `output.weight` has shape `[V, H2]` where `H1 ≠ H2`.
**Cause:** Corrupted model or unusual architecture.
**Workaround:** Error:
```
error: embedding prune error: hidden dim mismatch: embedding has H1=4096, output has H2=2048
[hint] these tensors should share the hidden dimension — model may be corrupted
```

#### EP-12: Memory pressure from large embedding matrices

**Trigger:** 128K vocab × 4096 hidden × 4 bytes = 512MB. Pruning requires loading the full embedding into memory.
**Workaround for v1:** Document the limitation. The dequantize + row-select + write pipeline is inherently streaming-friendly, but the current implementation loads the full tensor.
**Future:** Chunked processing (read/write in row batches). Not needed for v1 since 512MB fits in any machine that can run these models.

#### EP-13: Embedding tensor has more than 2 dimensions

**Trigger:** Shape `[V, H, 1]` or `[1, V, H]` (unusual but possible).
**Workaround:** If the tensor has exactly 2 dimensions, proceed normally. If it has 1 dimension, error. If it has 3+ dimensions, warn and attempt to interpret:
```
[warn] embedding tensor has 3 dimensions [V, H, 1] — treating as 2D [V, H]
```
Flatten the extra dims and proceed.

---

### CLI / PIPELINE — Failure Modes

#### CP-1: Task vector JSON file too large for memory

**Trigger:** Model with 100B params → task vector JSON could be ~800MB.
**Workaround for v1:** Let `serde_json::from_str` handle it — it will fail with `Error::Json` if memory is exhausted. Document: "task vector files for models >10B params may exceed system memory; use `task compute` on a machine with sufficient RAM".
**Future:** Binary format with memory-mapped reading.

#### CP-2: Pipeline step references non-existent model file

**Trigger:** Pipeline JSON with `"model": "nonexistent.gguf"`.
**Workaround:** Already handled — `GgufFile::open()` returns `Error::Io`, pipeline executor catches and propagates.

#### CP-3: Pipeline TaskArith step with missing required fields

**Trigger:** Pipeline JSON with `{"action": "TaskArith", "out": "out.gguf"}` but no `vector` or `finetuned`.
**Workaround:** Validate in `execute_pipeline`:
```rust
PipelineStep::TaskArith { action, vector, finetuned, .. } => {
    match action.as_str() {
        "compute" => {
            if finetuned.is_none() {
                return Err(Error::TaskArith("pipeline: 'finetuned' is required for 'compute'".into()));
            }
        }
        "apply" | "ties-merge" => {
            if vector.is_none() {
                return Err(Error::TaskArith("pipeline: 'vector' is required for 'apply'/'ties-merge'".into()));
            }
        }
        _ => return Err(Error::TaskArith(format!("pipeline: unknown action '{}'", action)))
    }
    // ... proceed
}
```

#### CP-4: PruneEmbed step with conflicting selection flags

**Trigger:** `--token-file tokens.txt --token-pattern ".*"` (both provided).
**Workaround:** Priority order: `token_file` > `token_pattern` > `token_top_n` > `token_id`. If multiple are provided, use the first and warn:
```
[warn] multiple token selection methods specified — using --token-file (highest priority)
```

#### CP-5: Pipeline step output path already exists

**Trigger:** Re-running a pipeline that already produced output files.
**Workaround:** The existing `confirm_or_exit` pattern handles this for CLI commands. For pipeline execution, the pipeline executor already overwrites (no confirmation in pipeline mode). Document: "pipeline steps overwrite existing output files without confirmation".

---

### DEQUANTIZATION — Failure Modes (shared by TA and EP)

#### DQ-1: Unknown quantization type

**Trigger:** Model uses a quantization type not yet implemented in `dequantize()`.
**Workaround:** `dequantize()` returns `None` for unknown types. Map to error:
```rust
dequantize(ggml_ty, &raw, None)
    .ok_or_else(|| Error::TaskArith(format!(
        "tensor '{}': unsupported quantization type {} — cannot dequantize",
        tensor.name, tensor.dtype.as_str()
    )))?
```

#### DQ-2: Corrupt quantized data (wrong byte count)

**Trigger:** Quantized tensor has fewer bytes than expected for its shape.
**Workaround:** `dequantize()` will return `None` or produce fewer elements than expected. Check output length:
```rust
let f32s = dequantize(ty, &raw, Some(expected_elems))
    .ok_or_else(|| Error::TaskArith(format!(
        "tensor '{}': dequantization failed (type={}, bytes={})",
        tensor.name, tensor.dtype.as_str(), raw.len()
    )))?;
if f32s.len() != expected_elems {
    return Err(Error::TaskArith(format!(
        "tensor '{}': dequantized to {} elements, expected {}",
        tensor.name, f32s.len(), expected_elems
    )));
}
```

---

### TEST-ONLY — Failure Modes

#### T-1: Mock model memory leak (`Box::leak`)

**Trigger:** Every test run leaks the mock tensor data.
**Workaround:** Acceptable for test-sized tensors (a few KB). The OS reclaims on process exit. If this becomes a problem (e.g., in CI with leak detection), switch to `static` arrays or `once_cell::Lazy`.

#### T-2: Test GGUF fixtures may not exist

**Trigger:** Integration tests need real GGUF files to test end-to-end.
**Workaround:** Create minimal GGUF test fixtures using the existing `GgufWriter` in `lib/src/formats/gguf/writer.rs`. Build a helper:
```rust
fn create_test_gguf(path: &Path, vocab_size: u32, hidden_dim: u32) {
    // Use GgufWriter to write a minimal GGUF with:
    // - token_embd.weight: [vocab_size, hidden_dim] f32
    // - output.weight: [hidden_dim, vocab_size] f32
    // - metadata: "tokenizer.ggml.tokens" = list of token strings
}
```
This is self-contained and doesn't depend on external model files.

---

### Summary: Error Variant Mapping

| Error Variant | Module | Covers |
|---|---|---|
| `Error::TaskArith(String)` | `merge/task_arith.rs` | TA-1 through TA-12 |
| `Error::EmbedPrune(String)` | `prune/embed.rs` | EP-1 through EP-13 |
| `Error::UnsupportedType(String)` | (existing) | DQ-1 fallback |
| `Error::TensorNotFound(String)` | (existing) | EP-1 fallback |
| `Error::Io(std::io::Error)` | (existing) | EP-5, CP-2 |
| `Error::Json(serde_json::Error)` | (existing) | TA-7, CP-1 |
| `Error::Regex(regex::Error)` | (existing) | EP-7 compilation |

---

## Open Questions

1. **Task vector file format**: JSON `{ "tensor_name": [f32...] }` per tensor is simple but big. Should we support a binary header version? (Recommend: JSON for v1, binary later.)

2. **Embedding pruning + output projection alignment**: In most architectures, `output.weight` has shape `[hidden_dim, vocab_size]` (transposed relative to embedding). The code must handle this. Verify the convention in the GGUF files being used. **Decision for v1:** Check both `[V, H]` and `[H, V]` layouts; validate that the vocab dimension matches; error if neither layout matches.

3. **Multi-file task vectors**: Should `task compute` emit one JSON per tensor or one big JSON? (Recommend: one big JSON, keyed by tensor name — matches the Hub pattern.)

4. **Token frequency data**: Where does the user get token frequency data? A separate tool (like `tokenizer --dump-freq`) might be needed. For v1, `--token-file` with a manual word list is sufficient.

5. **Re-quantization after task vector apply**: The output of `task apply` is always f32. Should the CLI offer an `--output-dtype` flag to re-quantize? (Recommend: no for v1 — chain with `tensorkit quant` in a pipeline.)

---

## Files Summary

| File | Action | Wave |
|---|---|---|
| `lib/src/error.rs` | Modify (add `TaskArith`, `EmbedPrune` variants) | 1 |
| `lib/src/merge/task_arith.rs` | Create | 1 |
| `lib/src/prune/embed.rs` | Create | 1 |
| `lib/src/merge/mod.rs` | Modify | 2 |
| `lib/src/prune/mod.rs` | Modify | 2 |
| `lib/src/lib.rs` | Modify | 2 |
| `cli/src/main.rs` | Modify | 2 |
| `lib/tests/unit/merge/task_arith.rs` | Create | 1 |
| `lib/tests/unit/prune/embed.rs` | Create | 1 |
| `lib/tests/unit/merge/tests.rs` | Modify | 2 |
| `cli/tests/cli_task.rs` | Create | 2 |
| `cli/src/pipeline.rs` | Modify | 3 |
