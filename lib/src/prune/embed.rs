//! Embedding-row pruning: remove token embedding rows for unused tokens.
//!
//! The token embedding matrix (`token_embd.weight`, shape `[vocab_size, hidden_dim]`)
//! is often the largest single tensor. When the user only needs a subset of the
//! vocabulary (e.g., English-only, a specific language), unused token rows can be
//! deleted, saving significant memory.
//!
//! After pruning, the output projection tensor (`output.weight`) is also remapped
//! to stay consistent with the reduced vocab size.

use crate::error::{Error, Result};
use crate::formats::gguf::dequant::scalar::{scan_bf16, scan_f16};
use crate::formats::gguf::dequant::dequantize;
use crate::formats::gguf::types::GgmlType;
use crate::model::{Model, TensorDtype};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

/// Candidate names for the embedding tensor (same as `tying.rs`).
const EMBED_CANDIDATES: &[&str] = &[
    "token_embd.weight",
    "tok_embeddings.weight",
    "embed.weight",
];

/// Candidate names for the output projection tensor.
const OUTPUT_CANDIDATES: &[&str] = &[
    "output.weight",
    "lm_head.weight",
    "embed_out.weight",
];

/// How to select which token rows to keep.
#[derive(Debug, Clone)]
pub enum TokenSelection {
    /// Keep only tokens whose IDs are in this list.
    ById(Vec<u32>),
    /// Keep tokens whose string representation matches this regex.
    ByPattern(regex::Regex),
    /// Keep the first N token rows (`0..N`).
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
    pub remap: HashMap<u32, u32>,
    /// Original vocab size.
    pub original_vocab_size: u32,
    /// New vocab size.
    pub new_vocab_size: u32,
    /// Name of the embedding tensor.
    pub embed_tensor_name: String,
    /// Name of the output projection tensor, if present.
    pub output_tensor_name: Option<String>,
}

/// Build a plan: analyze the model's vocab tensor and token metadata to
/// determine which rows to keep.
///
/// `vocab_tokens` provides the string representation of each token index.
/// Required for `ByPattern` and `ByFile` selections; ignored for `ById`
/// and `TopN`.
pub fn plan_embed_prune(
    model: &dyn Model,
    selection: &TokenSelection,
    vocab_tokens: Option<&[String]>,
) -> Result<EmbedPrunePlan> {
    // 1. Find embedding tensor.
    let embed_name = EMBED_CANDIDATES
        .iter()
        .find(|name| model.tensor(name).is_some())
        .ok_or_else(|| {
            Error::EmbedPrune(format!(
                "no embedding tensor found (tried: {})",
                EMBED_CANDIDATES.join(", ")
            ))
        })?;

    let embed_tensor = model.tensor(embed_name).unwrap();
    let vocab_size = *embed_tensor
        .shape
        .first()
        .ok_or_else(|| Error::EmbedPrune("embedding tensor has no dimensions".into()))?
        as u32;

    // 2. Find output tensor (optional).
    let output_name = OUTPUT_CANDIDATES
        .iter()
        .find(|name| model.tensor(name).is_some())
        .copied()
        .map(String::from);

    // 3. Validate vocab consistency.
    if let Some(ref out_name) = output_name {
        let out_tensor = model.tensor(out_name).unwrap();
        // output.weight is typically [hidden, vocab] — check last dim
        let out_vocab = *out_tensor.shape.last().unwrap_or(&0);
        if vocab_size != out_vocab as u32 {
            return Err(Error::EmbedPrune(format!(
                "vocab size mismatch: embedding '{}' has {} rows, output '{}' has {} cols",
                embed_name, vocab_size, out_name, out_vocab
            )));
        }
    }

    // 4. Resolve selection to keep-indices.
    let keep_rows = resolve_selection(selection, vocab_size, vocab_tokens)?;

    if keep_rows.is_empty() {
        return Err(Error::EmbedPrune(
            "selection resulted in 0 tokens to keep".into(),
        ));
    }

    // 5. Build remap.
    let remap: HashMap<u32, u32> = keep_rows
        .iter()
        .enumerate()
        .map(|(new_idx, &old_idx)| (old_idx, new_idx as u32))
        .collect();

    let new_vocab_size = keep_rows.len() as u32;

    Ok(EmbedPrunePlan {
        keep_rows,
        remap,
        original_vocab_size: vocab_size,
        new_vocab_size,
        embed_tensor_name: embed_name.to_string(),
        output_tensor_name: output_name,
    })
}

/// Apply the plan: read embedding tensor bytes, return new `(name, bytes)`
/// pairs with only the kept rows.
///
/// The output is always f32 (even if the input was quantized or f16/bf16).
pub fn apply_embed_prune<M: Model + ?Sized>(
    model: &M,
    plan: &EmbedPrunePlan,
) -> Result<Vec<(String, Vec<u8>)>> {
    let mut result = Vec::with_capacity(2);

    // --- Embedding tensor ---
    let embed_tensor = model
        .tensor(&plan.embed_tensor_name)
        .ok_or_else(|| Error::TensorNotFound(plan.embed_tensor_name.clone()))?;

    let embed_f32s = read_as_f32s(model, embed_tensor)?;
    let embed_vocab = embed_tensor.shape.first().copied().unwrap_or(0) as usize;
    let hidden_dim = if embed_vocab > 0 {
        embed_f32s.len() / embed_vocab
    } else {
        return Err(Error::EmbedPrune("embedding vocab size is 0".into()));
    };

    // Extract kept rows from the embedding.
    let new_embed_bytes = extract_rows(&embed_f32s, embed_vocab, hidden_dim, &plan.keep_rows);
    result.push((plan.embed_tensor_name.clone(), new_embed_bytes));

    // --- Output projection tensor ---
    if let Some(ref out_name) = plan.output_tensor_name {
        let out_tensor = model
            .tensor(out_name)
            .ok_or_else(|| Error::TensorNotFound(out_name.clone()))?;

        let out_f32s = read_as_f32s(model, out_tensor)?;
        let out_shape = &out_tensor.shape;

        // Determine layout: [hidden, vocab] or [vocab, hidden]?
        if out_shape.len() == 2 {
            let dim0 = out_shape[0] as usize;
            let dim1 = out_shape[1] as usize;

            if dim1 == embed_vocab {
                // Layout is [hidden, vocab] — prune columns.
                let new_out_bytes = extract_cols(&out_f32s, dim0, dim1, &plan.keep_rows);
                result.push((out_name.clone(), new_out_bytes));
            } else if dim0 == embed_vocab {
                // Layout is [vocab, hidden] — prune rows.
                let new_out_bytes = extract_rows(&out_f32s, dim0, dim1, &plan.keep_rows);
                result.push((out_name.clone(), new_out_bytes));
            } else {
                eprintln!(
                    "[warn] output tensor '{}' shape {:?} doesn't match vocab {} — skipping",
                    out_name, out_shape, embed_vocab
                );
            }
        } else {
            eprintln!(
                "[warn] output tensor '{}' has {} dims (expected 2) — skipping",
                out_name,
                out_shape.len()
            );
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Resolve a `TokenSelection` into a sorted `Vec<u32>` of row indices to keep.
fn resolve_selection(
    selection: &TokenSelection,
    vocab_size: u32,
    vocab_tokens: Option<&[String]>,
) -> Result<Vec<u32>> {
    let mut keep: Vec<u32> = match selection {
        TokenSelection::ById(ids) => {
            let mut filtered: Vec<u32> = ids
                .iter()
                .copied()
                .filter(|&id| id < vocab_size)
                .collect();
            let skipped = ids.len() - filtered.len();
            if skipped > 0 {
                eprintln!(
                    "[warn] {} token index(es) >= vocab_size {} — skipped",
                    skipped, vocab_size
                );
            }
            filtered.sort_unstable();
            filtered.dedup();
            filtered
        }

        TokenSelection::TopN(n) => {
            let count = (*n).min(vocab_size as usize);
            if *n > vocab_size as usize {
                eprintln!(
                    "[warn] --token-top-n {} exceeds vocab size {} — clamping",
                    n, vocab_size
                );
            }
            (0..count as u32).collect()
        }

        TokenSelection::ByPattern(re) => {
            let tokens = vocab_tokens.ok_or_else(|| {
                Error::EmbedPrune(
                    "pattern matching requires token strings; provide --token-file \
                     or use --token-id / --token-top-n instead"
                        .into(),
                )
            })?;
            let mut matched: Vec<u32> = tokens
                .iter()
                .enumerate()
                .filter(|(_, tok)| re.is_match(tok))
                .map(|(i, _)| i as u32)
                .collect();
            if matched.is_empty() {
                return Err(Error::EmbedPrune(format!(
                    "pattern '{}' matched 0 tokens out of {}",
                    re.as_str(),
                    vocab_size
                )));
            }
            matched.sort_unstable();
            matched
        }

        TokenSelection::ByFile(path) => {
            let file = std::fs::File::open(path).map_err(Error::Io)?;
            let reader = std::io::BufReader::new(file);
            let token_set: HashSet<String> = reader
                .lines()
                .map(|line| line.map(|l| l.trim().to_string()))
                .collect::<std::result::Result<HashSet<_>, _>>()
                .map_err(Error::Io)?;

            if token_set.is_empty() {
                return Err(Error::EmbedPrune(
                    "token file is empty (0 tokens)".into(),
                ));
            }

            let tokens = vocab_tokens.ok_or_else(|| {
                Error::EmbedPrune(
                    "file-based selection requires token strings; model must have \
                     tokenizer metadata"
                        .into(),
                )
            })?;

            let mut matched: Vec<u32> = tokens
                .iter()
                .enumerate()
                .filter(|(_, tok)| token_set.contains(*tok))
                .map(|(i, _)| i as u32)
                .collect();

            if matched.is_empty() {
                return Err(Error::EmbedPrune(format!(
                    "token file matched 0 tokens out of {} (file had {} entries)",
                    vocab_size,
                    token_set.len()
                )));
            }

            matched.sort_unstable();
            matched
        }
    };

    keep.sort_unstable();
    keep.dedup();
    Ok(keep)
}

/// Read a tensor's raw bytes and convert to `Vec<f32>`, handling any dtype.
fn read_as_f32s<M: Model + ?Sized>(model: &M, tensor: &crate::model::Tensor) -> Result<Vec<f32>> {
    let raw = model.read_tensor_bytes(&tensor.name)?;
    tensor_to_f32s(tensor, &raw)
}

/// Convert raw tensor bytes to `Vec<f32>` based on dtype.
fn tensor_to_f32s(tensor: &crate::model::Tensor, raw: &[u8]) -> Result<Vec<f32>> {
    match tensor.dtype {
        TensorDtype::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        TensorDtype::F16 => Ok(scan_f16(raw)),
        TensorDtype::Bf16 => Ok(scan_bf16(raw)),
        _ => {
            let ggml_ty = dtype_to_ggml(tensor.dtype)?;
            dequantize(ggml_ty, raw, None).ok_or_else(|| {
                Error::EmbedPrune(format!(
                    "cannot dequantize tensor '{}' (type={})",
                    tensor.name,
                    tensor.dtype.as_str()
                ))
            })
        }
    }
}

/// Map `TensorDtype` to `GgmlType` for dequantization.
fn dtype_to_ggml(dt: TensorDtype) -> Result<GgmlType> {
    match dt {
        TensorDtype::Q4_0 => Ok(GgmlType::Q4_0),
        TensorDtype::Q4_1 => Ok(GgmlType::Q4_1),
        TensorDtype::Q5_0 => Ok(GgmlType::Q5_0),
        TensorDtype::Q5_1 => Ok(GgmlType::Q5_1),
        TensorDtype::Q8_0 => Ok(GgmlType::Q8_0),
        TensorDtype::Q8_1 => Ok(GgmlType::Q8_1),
        TensorDtype::Q2K => Ok(GgmlType::Q2K),
        TensorDtype::Q3K => Ok(GgmlType::Q3K),
        TensorDtype::Q4K => Ok(GgmlType::Q4K),
        TensorDtype::Q5K => Ok(GgmlType::Q5K),
        TensorDtype::Q6K => Ok(GgmlType::Q6K),
        TensorDtype::Q8K => Ok(GgmlType::Q8K),
        TensorDtype::Iq2Xxs => Ok(GgmlType::Iq2Xxs),
        TensorDtype::Iq2Xs => Ok(GgmlType::Iq2Xs),
        TensorDtype::Iq3Xxs => Ok(GgmlType::Iq3Xxs),
        TensorDtype::Iq1S => Ok(GgmlType::Iq1S),
        TensorDtype::Iq4Nl => Ok(GgmlType::Iq4Nl),
        TensorDtype::Iq3S => Ok(GgmlType::Iq3S),
        TensorDtype::Iq2S => Ok(GgmlType::Iq2S),
        TensorDtype::Iq4Xs => Ok(GgmlType::Iq4Xs),
        TensorDtype::Iq1M => Ok(GgmlType::Iq1M),
        TensorDtype::Tq1_0 => Ok(GgmlType::Tq1_0),
        TensorDtype::Tq2_0 => Ok(GgmlType::Tq2_0),
        TensorDtype::I8 => Ok(GgmlType::I8),
        TensorDtype::I16 => Ok(GgmlType::I16),
        TensorDtype::I32 => Ok(GgmlType::I32),
        TensorDtype::I64 => Ok(GgmlType::I64),
        _ => Err(Error::UnsupportedType(format!(
            "cannot convert {} to GgmlType",
            dt.as_str()
        ))),
    }
}

/// Extract specific rows from a row-major `[rows, cols]` f32 slice.
/// Returns the bytes of the new tensor (f32, little-endian).
fn extract_rows(data: &[f32], rows: usize, cols: usize, keep: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(keep.len() * cols * 4);
    for &row_idx in keep {
        let start = row_idx as usize * cols;
        let end = start + cols;
        if end <= data.len() {
            for &val in &data[start..end] {
                out.extend_from_slice(&val.to_le_bytes());
            }
        }
    }
    out
}

/// Extract specific columns from a row-major `[rows, cols]` f32 slice.
/// Returns the bytes of the new tensor (f32, little-endian).
fn extract_cols(data: &[f32], rows: usize, cols: usize, keep: &[u32]) -> Vec<u8> {
    let new_cols = keep.len();
    let mut out = Vec::with_capacity(rows * new_cols * 4);
    for row in 0..rows {
        for &col_idx in keep {
            let idx = row * cols + col_idx as usize;
            if idx < data.len() {
                out.extend_from_slice(&data[idx].to_le_bytes());
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/prune/embed.rs"]
mod tests;
