//! SVD-compression configuration.
//!
//! ## Layer selection grammar
//!
//! ```text
//!   all                          â€” every block
//!   0-23                         â€” inclusive range
//!   0,1,2                        â€” explicit list
//!   0-5,10,20-22                 â€” combinations
//!   regex:^blk\.(0|1|2)\.       â€” by regex (matched against tensor name)
//!   all-attn                     â€” alias for all 2D attention projections
//!   all-ffn                      â€” alias for all 2D FFN projections
//!   all-mlp                      â€” alias for all 2D attention + FFN projections
//! ```
//!
//! ## Tensor selection grammar (per selected layer)
//!
//! ```text
//!   attn                         â€” attn_q, attn_k, attn_v, attn_output
//!   ffn                          â€” ffn_up, ffn_down, ffn_gate, ffn_gate_up
//!   mlp                          â€” same as attn+ffn
//!   attn_q,attn_v                â€” explicit list of suffixes
//!   regex:^.*\.weight$           â€” by regex (matched against tensor name suffix after `blk.N.`)
//!   all                          â€” any 2D weight
//! ```
//!
//! ## Rank specification grammar
//!
//! ```text
//!   64                           â€” absolute rank for every selected tensor
//!   0.5                          â€” fraction of min(m, n) (50% in this example)
//!   energy:0.99                  â€” keep enough singular values to retain 99% of squared-singular-value sum
//!   abs:64,min:8,max:512         â€” absolute rank, with floor/ceiling clamps
//!   frac:0.5,min:8,max:512       â€” fractional with clamps
//! ```
//!
//! ## Output dtype
//!
//! ```text
//!   f32, f16, bf16               â€” element type for the packed (A, B) factors
//! ```

use crate::error::{Error, Result};
use regex::Regex;
use std::collections::BTreeMap;

/// Tensor-family token used by `AdjacentSelection`. Each variant maps to a
/// 2-D weight suffix like `.attn_q.weight` (see `ATTN_SUFFIXES` / `FFN_SUFFIXES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdjacentRole {
    AttnQ,
    AttnK,
    AttnV,
    AttnOutput,
    FfnUp,
    FfnDown,
    FfnGate,
    FfnGateUp,
}

impl AdjacentRole {
    /// Lowercase token used in the selection grammar.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AttnQ => "attn_q",
            Self::AttnK => "attn_k",
            Self::AttnV => "attn_v",
            Self::AttnOutput => "attn_output",
            Self::FfnUp => "ffn_up",
            Self::FfnDown => "ffn_down",
            Self::FfnGate => "ffn_gate",
            Self::FfnGateUp => "ffn_gate_up",
        }
    }

    /// Parse a single role token. Returns `None` for anything else.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "attn_q" => Some(Self::AttnQ),
            "attn_k" => Some(Self::AttnK),
            "attn_v" => Some(Self::AttnV),
            "attn_output" => Some(Self::AttnOutput),
            "ffn_up" => Some(Self::FfnUp),
            "ffn_down" => Some(Self::FfnDown),
            "ffn_gate" => Some(Self::FfnGate),
            "ffn_gate_up" => Some(Self::FfnGateUp),
            _ => None,
        }
    }
}

/// One `(role, block_offset)` entry in an `AdjacentSelection`.
///
/// `offset` is added to the *primary* block index to compute the
/// adjacent target's block. `0` means "same block", `1` means "next block",
/// `-1` means "previous block".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AdjacentEntry {
    pub role: AdjacentRole,
    pub offset: i32,
}

/// User-supplied list of "adjacent" tensor targets to compress alongside
/// the primary selection.
///
/// Grammar:
/// ```text
///   <role>[+<role>...][+N|-N]
///
///   role  := attn_q | attn_k | attn_v | attn_output
///         |  ffn_up | ffn_down | ffn_gate | ffn_gate_up
///   N     := signed integer offset applied to the block index
///           of the most recent role (default 0).
///
///   ""    â€” no adjacent selection (returns Ok(None) from parse).
/// ```
///
/// Examples:
///   `attn_v`          â€” same-block attn_v
///   `attn_v+1`        â€” next-block attn_v
///   `attn_v-1`        â€” previous-block attn_v
///   `ffn_gate-2+ffn_up+1` â€” ffn_gate two blocks back, ffn_up one ahead
#[derive(Debug, Clone, Default)]
pub struct AdjacentSelection {
    pub entries: Vec<AdjacentEntry>,
}

impl AdjacentSelection {
    /// Parse a string into an `AdjacentSelection`. Returns `Ok(None)` for
    /// the empty string (which is the "no adjacent" default).
    pub fn parse(s: &str) -> Result<Option<Self>> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(None);
        }
        let mut entries: Vec<AdjacentEntry> = Vec::new();
        for token in s.split('+') {
            if token.is_empty() {
                return Err(Error::InvalidSvdConfig(format!(
                    "empty role in adjacent selection '{s}' (stray '+'?)"
                )));
            }
            // Bare integer: an offset applied to the most recent role.
            if let Ok(offset) = token.parse::<i32>() {
                let last = entries.last_mut().ok_or_else(|| {
                    Error::InvalidSvdConfig(format!(
                        "offset '{token}' without preceding role in adjacent selection '{s}'"
                    ))
                })?;
                last.offset = offset;
                continue;
            }
            // Role token, possibly with a negative offset suffix
            // (e.g. `attn_v-1`, `ffn_gate_up-2`). Positive offsets are
            // expressed via the `+` separator, so they never appear here.
            // We keep the leading `-` so `offset_str` parses as a signed int.
            let (role_name, offset_str) = match token.find('-') {
                Some(idx) => (&token[..idx], &token[idx..]),
                None => (token, ""),
            };
            let role = AdjacentRole::parse(role_name).ok_or_else(|| {
                Error::InvalidSvdConfig(format!(
                    "unknown role '{role_name}' in adjacent selection '{s}' \
                     (want attn_q, attn_k, attn_v, attn_output, \
                     ffn_up, ffn_down, ffn_gate, ffn_gate_up)"
                ))
            })?;
            if role_name.is_empty() {
                return Err(Error::InvalidSvdConfig(format!(
                    "missing role before '-' in adjacent selection '{s}'"
                )));
            }
            let offset: i32 = if offset_str.is_empty() {
                0
            } else {
                offset_str.parse().map_err(|_| {
                    Error::InvalidSvdConfig(format!(
                        "bad offset '{offset_str}' for role '{role_name}' \
                         in adjacent selection '{s}'"
                    ))
                })?
            };
            entries.push(AdjacentEntry { role, offset });
        }
        if entries.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self { entries }))
    }
}

/// Which layers (transformer blocks) the SVD should target.
#[derive(Debug, Clone)]
pub enum LayerSelection {
    /// Every block index found in the model.
    All,
    /// A list of explicit block indices.
    Indices(Vec<i32>),
    /// A regex matched against each tensor's full name.
    Pattern(Regex),
    /// Convenience: all attention projections (attn_q, attn_k, attn_v, attn_output) in every block.
    AllAttn,
    /// Convenience: all FFN projections (ffn_up, ffn_down, ffn_gate, ffn_gate_up) in every block.
    AllFfn,
    /// Convenience: all attention + FFN projections in every block.
    AllMlp,
}

impl LayerSelection {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s == "all" {
            return Ok(Self::All);
        }
        if s == "all-attn" {
            return Ok(Self::AllAttn);
        }
        if s == "all-ffn" {
            return Ok(Self::AllFfn);
        }
        if s == "all-mlp" {
            return Ok(Self::AllMlp);
        }
        if let Some(rest) = s.strip_prefix("regex:") {
            let re = Regex::new(rest)
                .map_err(|e| Error::InvalidSvdConfig(format!("bad layer regex: {e}")))?;
            return Ok(Self::Pattern(re));
        }
        // Default: list / range of indices.
        let idx = crate::prune::selection::parse_index_list(s)?;
        if idx.is_empty() {
            return Err(Error::InvalidSvdConfig(format!(
                "empty layer list in '{s}'"
            )));
        }
        Ok(Self::Indices(idx))
    }
}

/// Which tensors within each selected layer should be compressed.
#[derive(Debug, Clone)]
pub enum TensorSelection {
    /// Any 2D `.weight` tensor (skips 1D vectors and quantization blocks that aren't 2D matrices).
    All,
    /// Suffix list (matched against the part after `blk.N.`).
    Named(Vec<String>),
    /// Regex matched against the full tensor name.
    Pattern(Regex),
    /// Convenience: attention projections.
    Attn,
    /// Convenience: FFN projections.
    Ffn,
    /// Convenience: attention + FFN.
    Mlp,
}

impl TensorSelection {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s == "all" {
            return Ok(Self::All);
        }
        if s == "attn" {
            return Ok(Self::Attn);
        }
        if s == "ffn" {
            return Ok(Self::Ffn);
        }
        if s == "mlp" {
            return Ok(Self::Mlp);
        }
        if let Some(rest) = s.strip_prefix("regex:") {
            let re = Regex::new(rest)
                .map_err(|e| Error::InvalidSvdConfig(format!("bad tensor regex: {e}")))?;
            return Ok(Self::Pattern(re));
        }
        // Comma-separated list of suffixes.
        let mut v = Vec::new();
        for part in s.split(',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            v.push(p.to_string());
        }
        if v.is_empty() {
            return Err(Error::InvalidSvdConfig(format!(
                "empty tensor list in '{s}'"
            )));
        }
        Ok(Self::Named(v))
    }

    /// Returns true if a tensor with full name `full` (e.g. `blk.3.attn_q.weight`)
    /// matches this selection.
    pub fn matches(&self, full: &str) -> bool {
        match self {
            Self::All => is_2d_weight(full),
            Self::Attn => suffix_in(full, ATTN_SUFFIXES),
            Self::Ffn => suffix_in(full, FFN_SUFFIXES),
            Self::Mlp => suffix_in(full, ATTN_SUFFIXES) || suffix_in(full, FFN_SUFFIXES),
            Self::Named(suffixes) => suffixes.iter().any(|s| full.contains(s)),
            Self::Pattern(re) => re.is_match(full),
        }
    }
}

/// Rank specification. May carry optional floor/ceiling clamps.
#[derive(Debug, Clone)]
pub enum RankSpec {
    /// Absolute rank `k` for every selected tensor.
    Absolute(usize),
    /// Fraction of `min(m, n)` for every selected tensor.
    Fraction(f64),
    /// Keep the smallest `k` such that `sum_{i<k} s_i^2 >= energy * total`.
    Energy(f64),
}

#[derive(Debug, Clone)]
pub struct RankClamps {
    pub min: usize,
    pub max: Option<usize>,
}

impl Default for RankClamps {
    fn default() -> Self {
        Self { min: 1, max: None }
    }
}

#[derive(Debug, Clone)]
pub struct RankSpecWithClamps {
    pub spec: RankSpec,
    pub clamps: RankClamps,
}

impl RankSpecWithClamps {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        // Parse the leading rank spec; collect trailing ,key:val clamps.
        let mut parts = s.split(',').map(str::trim).filter(|p| !p.is_empty());
        let head = parts
            .next()
            .ok_or_else(|| Error::InvalidSvdConfig("empty rank spec".into()))?;
        let mut clamps = RankClamps::default();
        let spec: Option<RankSpec>;
        if let Some(rest) = head.strip_prefix("energy:") {
            let e: f64 = rest
                .parse()
                .map_err(|e| Error::InvalidSvdConfig(format!("bad energy '{rest}': {e}")))?;
            if !(0.0..=1.0).contains(&e) {
                return Err(Error::InvalidSvdConfig(format!(
                    "energy must be in [0,1], got {e}"
                )));
            }
            spec = Some(RankSpec::Energy(e));
        } else if let Some(rest) = head.strip_prefix("abs:") {
            let n: usize = rest
                .parse()
                .map_err(|e| Error::InvalidSvdConfig(format!("bad abs rank '{rest}': {e}")))?;
            spec = Some(RankSpec::Absolute(n));
        } else if let Some(rest) = head.strip_prefix("frac:") {
            let n: f64 = rest
                .parse()
                .map_err(|e| Error::InvalidSvdConfig(format!("bad frac rank '{rest}': {e}")))?;
            if !(0.0..=1.0).contains(&n) {
                return Err(Error::InvalidSvdConfig(format!(
                    "frac must be in [0,1], got {n}"
                )));
            }
            spec = Some(RankSpec::Fraction(n));
        } else {
            if let Ok(n) = head.parse::<usize>() {
                spec = Some(RankSpec::Absolute(n));
            } else if let Ok(f) = head.parse::<f64>() {
                if !(0.0..=1.0).contains(&f) {
                    return Err(Error::InvalidSvdConfig(format!(
                        "rank must be int or fraction in [0,1], got {f}"
                    )));
                }
                spec = Some(RankSpec::Fraction(f));
            } else {
                return Err(Error::InvalidSvdConfig(format!(
                    "unrecognized rank '{head}'"
                )));
            }
        }
        for p in parts {
            if let Some(rest) = p.strip_prefix("min:") {
                clamps.min = rest
                    .parse()
                    .map_err(|e| Error::InvalidSvdConfig(format!("bad min: {e}")))?;
            } else if let Some(rest) = p.strip_prefix("max:") {
                clamps.max = Some(
                    rest.parse()
                        .map_err(|e| Error::InvalidSvdConfig(format!("bad max: {e}")))?,
                );
            } else {
                return Err(Error::InvalidSvdConfig(format!(
                    "unknown rank option '{p}'"
                )));
            }
        }
        Ok(Self {
            spec: spec.expect("rank spec must be set above"),
            clamps,
        })
    }

    /// Apply the spec to a tensor of shape `m x n`, given a precomputed
    /// spectrum `s` (required for `Energy`, ignored otherwise).
    pub fn resolve(&self, m: usize, n: usize, s: Option<&[f32]>) -> usize {
        let max_possible = m.min(n).max(1);
        let raw = match &self.spec {
            RankSpec::Absolute(k) => *k,
            RankSpec::Fraction(f) => ((max_possible as f64) * f).floor() as usize,
            RankSpec::Energy(e) => {
                let s = s.unwrap_or(&[]);
                super::linalg::rank_for_energy(s, *e, 1, max_possible)
            }
        };
        let lo = self.clamps.min.max(1);
        let hi = self.clamps.max.unwrap_or(max_possible).min(max_possible);
        raw.clamp(lo, hi.max(lo))
    }
}

/// Output element type for the packed (A, B) factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDtype {
    F32,
    F16,
    Bf16,
    /// Auto-select a quantization format that matches the input tensor's
    /// on-disk precision. Currently picks Q8_0 (best accuracy/size ratio for
    /// factors) when the input is quantized, otherwise falls back to F16.
    AutoQuant,
    /// Explicit GGUF block quant type. The string form is one of
    /// `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q4_k`, `q5_k`, `q6_k`,
    /// `q2_k`, `q3_k`, `q8_k` (matching the `--quantize` flags).
    Ggml(crate::formats::gguf::types::GgmlType),
}

/// SVD computation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdBackend {
    /// Pure-Rust parallel Jacobi (no external dependencies, good for small matrices).
    Jacobi,
    /// LAPACK-quality divide-and-conquer via `faer` (fast for large matrices).
    Faer,
}

impl SvdBackend {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "jacobi" | "parallel" | "rust" => Ok(Self::Jacobi),
            "faer" | "lapack" | "fast" => Ok(Self::Faer),
            other => Err(Error::InvalidSvdConfig(format!(
                "unknown SVD backend '{other}' (supported: jacobi, faer)"
            ))),
        }
    }
}

impl std::fmt::Display for SvdBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Jacobi => write!(f, "jacobi"),
            Self::Faer => write!(f, "faer"),
        }
    }
}

impl OutputDtype {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        match s.to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(Self::F32),
            "f16" | "float16" | "fp16" | "half" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "auto" | "autoquant" | "auto-quant" => Ok(Self::AutoQuant),
            other => {
                use crate::formats::gguf::types::GgmlType::*;
                let ty = match other {
                    "f32" => F32,
                    "f16" => F16,
                    "bf16" => Bf16,
                    "q4_0" => Q4_0,
                    "q4_1" => Q4_1,
                    "q5_0" => Q5_0,
                    "q5_1" => Q5_1,
                    "q8_0" => Q8_0,
                    "q8_1" => Q8_1,
                    "q2_k" => Q2K,
                    "q3_k" => Q3K,
                    "q4_k" => Q4K,
                    "q5_k" => Q5K,
                    "q6_k" => Q6K,
                    "q8_k" => Q8K,
                    _ => {
                        return Err(Error::InvalidSvdConfig(format!(
                            "unknown dtype '{other}' (want f32/f16/bf16/auto/<qtype>)"
                        )))
                    }
                };
                Ok(Self::Ggml(ty))
            }
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::AutoQuant => "AUTO-QUANT",
            Self::Ggml(t) => t.as_str(),
        }
    }
    pub fn is_supported_for_ggml(self) -> bool {
        // All variants produce a valid GGML tensor type (raw bytes for F*,
        // GgmlType for the rest).
        true
    }
}

/// Top-level SVD compression configuration.
#[derive(Debug, Clone)]
pub struct SvdConfig {
    pub layers: LayerSelection,
    pub tensors: TensorSelection,
    pub rank: RankSpecWithClamps,
    pub dtype: OutputDtype,
    /// SVD computation backend (Jacobi or Faer).
    pub backend: SvdBackend,
    /// Minimum size of a tensor (min(m, n)) to be eligible. Smaller tensors are skipped.
    pub min_dim: usize,
    /// Random-seeded randomized SVD for large matrices.
    pub randomized: bool,
    /// Randomized SVD oversampling (extra columns in the test matrix).
    pub randomized_oversample: usize,
    /// Randomized SVD power iterations.
    pub randomized_power_iters: usize,
    /// Threshold (in elements) above which randomized SVD is used (when enabled).
    pub randomized_min_elems: usize,
    /// Suffix appended to the original name to form the "A" (tall) factor.
    pub suffix_a: String,
    /// Suffix appended to the original name to form the "B" (wide) factor.
    pub suffix_b: String,
    /// Per-layer rank overrides (block index -> rank spec).
    pub per_layer: BTreeMap<i32, RankSpecWithClamps>,
    /// Per-tensor-suffix rank overrides (matched substring -> rank spec).
    pub per_tensor: Vec<(String, RankSpecWithClamps)>,
    /// Optional list of "adjacent" tensor targets (different role and/or
    /// different block) to compress alongside the primary selection.
    /// `None` (the default) preserves the original behavior of only
    /// targeting the layer + tensor filters.
    pub adjacent: Option<AdjacentSelection>,
}

impl Default for SvdConfig {
    fn default() -> Self {
        Self {
            layers: LayerSelection::All,
            tensors: TensorSelection::Mlp,
            rank: RankSpecWithClamps {
                spec: RankSpec::Fraction(0.5),
                clamps: RankClamps { min: 4, max: None },
            },
            dtype: OutputDtype::F16,
            backend: SvdBackend::Faer,
            min_dim: 16,
            randomized: true,
            randomized_oversample: 8,
            randomized_power_iters: 2,
            randomized_min_elems: 1 << 18, // 256K elems
            suffix_a: ".svd_a".into(),
            suffix_b: ".svd_b".into(),
            per_layer: BTreeMap::new(),
            per_tensor: Vec::new(),
            adjacent: None,
        }
    }
}

impl SvdConfig {
    /// Resolve the effective rank for a single tensor.
    pub fn resolve_rank(
        &self,
        name: &str,
        block_idx: i32,
        m: usize,
        n: usize,
        s: Option<&[f32]>,
    ) -> usize {
        // 1) per-tensor override (first match wins)
        for (needle, spec) in &self.per_tensor {
            if name.contains(needle) {
                return spec.resolve(m, n, s);
            }
        }
        // 2) per-layer override
        if let Some(spec) = self.per_layer.get(&block_idx) {
            return spec.resolve(m, n, s);
        }
        // 3) global spec
        self.rank.resolve(m, n, s)
    }

    /// Build the `(a, b)` factor names from an original tensor name.
    pub fn factor_names(&self, original: &str) -> (String, String) {
        (
            format!("{original}{}", self.suffix_a),
            format!("{original}{}", self.suffix_b),
        )
    }
}

// -- Helpers used by TensorSelection / apply --------------------------------

/// Common attention projection suffixes (lowercase, after `blk.N.`).
pub const ATTN_SUFFIXES: &[&str] = &[
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_output.weight",
    ".attn_qkv.weight",
];

/// Common FFN / MLP projection suffixes.
pub const FFN_SUFFIXES: &[&str] = &[
    ".ffn_up.weight",
    ".ffn_down.weight",
    ".ffn_gate.weight",
    ".ffn_gate_up.weight",
    ".ffn_up_exps.weight",
    ".ffn_down_exps.weight",
    ".ffn_gate_exps.weight",
];

#[inline]
pub fn is_2d_weight(name: &str) -> bool {
    name.ends_with(".weight") && !name.contains("norm") && !name.contains("rope")
}

#[inline]
pub(crate) fn suffix_in(name: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|s| name.ends_with(s))
}

#[cfg(test)]
#[path = "../../tests/unit/svd/config.rs"]
mod tests;
