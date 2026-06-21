//! Apply a `PrunePlan` to a model and write the pruned file to disk.
//!
//! Metadata handling:
//! - Scalar `*.block_count` and `*.tensor_count` keys are updated.
//! - Array-typed metadata whose length equals the original block count is
//!   shrunk by removing elements at the dropped block indices.
//! - Per-block metadata keys (`{arch}.blk.{N}.*`) are removed for dropped
//!   blocks and re-indexed for kept blocks.
//! - `tensorkit.prune.*` traceability metadata is written.

use crate::error::{Error, Result};
use crate::formats::gguf::reader::GgufFile;
use crate::formats::gguf::types::{ArrayValue, MetaValue};
use crate::formats::gguf::writer::GgufWriter;
use crate::formats::onnx::{OnnxFile, OnnxWriter};
use crate::formats::safetensors::reader::SafetensorsFile;
use crate::formats::safetensors::writer::SafetensorsWriter;
use crate::model::{Model, TensorDtype};
use crate::prune::plan::PrunePlan;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

#[derive(Debug, serde::Serialize)]
pub struct PruneReport {
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub tensors_kept: usize,
    pub tensors_dropped: usize,
    pub blocks_dropped: Vec<i32>,
    pub original_block_count: i32,
    pub new_block_count: i32,
    pub output_path: String,
}

/// Known per-layer metadata keys (stored as arrays of length `block_count`).
/// These patterns cover all major GGUF architectures.
static PER_LAYER_ARRAY_PATTERNS: &[&str] = &[
    ".attention.head_count",
    ".attention.head_count_kv",
    ".attention.alibi_bias_max",
    ".attention.causal",
    ".attention.sliding_window",
    ".attention.key_length",
    ".attention.value_length",
    ".feed_forward_length",
    ".expert_count",
    ".expert_used_count",
    ".expert_feed_forward_length",
    ".rope.dimension_count",
    ".rope.freq_base",
    ".decoder_start_token_id",
    ".ssm.conv_kernel",
    ".ssm.inner_size",
    ".ssm.state_size",
    ".ssm.time_step_rank",
    ".layer_norm_epsilon",
    ".layer_norm_rms_epsilon",
    ".residual_scale",
    ".embedding_length",
];

/// Check if a metadata key is a block-count key.
#[inline]
pub fn is_block_count_key(key: &str) -> bool {
    key == "block_count" || key.ends_with(".block_count")
}

#[inline]
pub fn is_tensor_count_key(key: &str) -> bool {
    key == "tensor_count" || key.ends_with(".tensor_count")
}

/// Check if a metadata key looks like a per-layer array based on known patterns.
pub fn looks_like_per_layer_array(key: &str, arch: &str) -> bool {
    let rest = key.strip_prefix(&format!("{arch}.")).unwrap_or(key);
    PER_LAYER_ARRAY_PATTERNS
        .iter()
        .any(|p| rest == *p || rest.starts_with(&p[1..]))
}

/// Check if a metadata key contains a block index reference (`blk.{N}.`).
pub fn parse_block_key(key: &str) -> Option<(String, i32, String)> {
    // Match: <prefix>blk.<N>.<suffix>
    let blk_pos = key.find("blk.")?;
    let prefix = &key[..blk_pos];
    let after_blk = &key[blk_pos + 4..]; // skip "blk."
    let dot_pos = after_blk.find('.')?;
    let idx_str = &after_blk[..dot_pos];
    let idx: i32 = idx_str.parse().ok()?;
    let suffix = &after_blk[dot_pos..]; // includes the dot
    Some((prefix.to_string(), idx, suffix.to_string()))
}

/// Remove array elements at `dropped` indices (which must be sorted ascending).
pub fn shrink_array(arr: &ArrayValue, dropped: &[i32]) -> ArrayValue {
    let drop_set: HashSet<i32> = dropped.iter().cloned().collect();
    let mut new_elements = Vec::with_capacity(arr.elements.len().saturating_sub(dropped.len()));
    for (i, elem) in arr.elements.iter().enumerate() {
        if !drop_set.contains(&(i as i32)) {
            new_elements.push(elem.clone());
        }
    }
    ArrayValue {
        elem_type: arr.elem_type,
        elements: new_elements,
    }
}

pub fn gguf_value_type(v: &MetaValue) -> u32 {
    match v {
        MetaValue::U8(_) => 0,
        MetaValue::I8(_) => 1,
        MetaValue::U16(_) => 2,
        MetaValue::I16(_) => 3,
        MetaValue::U32(_) => 4,
        MetaValue::I32(_) => 5,
        MetaValue::F32(_) => 6,
        MetaValue::Bool(_) => 7,
        MetaValue::String(_) => 8,
        MetaValue::Array(_) => 9,
        MetaValue::U64(_) => 10,
        MetaValue::I64(_) => 11,
        MetaValue::F64(_) => 12,
    }
}

pub fn apply_to_gguf(gg: &GgufFile, plan: &PrunePlan, dst: &Path) -> Result<PruneReport> {
    let mut writer = GgufWriter::new(gg.version, gg.alignment);

    let arch = gg.architecture().unwrap_or("");
    let kept_tensor_count: u64 = plan.keep.iter().filter(|(_, k)| *k).count() as u64;

    // Build a set of dropped block indices for fast lookup.
    let dropped_set: HashSet<i32> = plan.dropped_blocks.iter().cloned().collect();

    for kv in &gg.metadata {
        let mut new_kv = kv.clone();

        // 1. Update block_count / tensor_count scalars.
        if is_block_count_key(&kv.key) {
            new_kv.value = match kv.value {
                MetaValue::U32(_) => MetaValue::U32(plan.new_block_count as u32),
                MetaValue::U64(_) => MetaValue::U64(plan.new_block_count as u64),
                _ => kv.value.clone(),
            };
            new_kv.value_type = gguf_value_type(&new_kv.value);
        } else if is_tensor_count_key(&kv.key) {
            new_kv.value = match kv.value {
                MetaValue::U32(_) => MetaValue::U32(kept_tensor_count as u32),
                MetaValue::U64(_) => MetaValue::U64(kept_tensor_count),
                _ => kv.value.clone(),
            };
            new_kv.value_type = gguf_value_type(&new_kv.value);
        }

        // 2. Shrink array-typed metadata where length matches original block count,
        //    or where the key matches a known per-layer pattern.
        if let MetaValue::Array(ref arr) = kv.value {
            let len_matches = arr.elements.len() as i32 == plan.original_block_count;
            let is_per_layer = looks_like_per_layer_array(&kv.key, arch);
            if (len_matches || is_per_layer) && !plan.dropped_blocks.is_empty() {
                new_kv.value = MetaValue::Array(shrink_array(arr, &plan.dropped_blocks));
                new_kv.value_type = 9; // Array
            }
        }

        // 3. Handle per-block metadata keys (`{prefix}blk.{N}.{suffix}`).
        if let Some((_, blk_idx, _)) = parse_block_key(&kv.key) {
            if dropped_set.contains(&blk_idx) {
                // Drop this metadata key entirely.
                continue;
            }
            if let Some(&new_idx) = plan.remap.get(&blk_idx) {
                // Re-index the block number in the key.
                new_kv.key = rename_metadata_block_key(&kv.key, blk_idx, new_idx);
            }
        }

        writer.add_kv(new_kv);
    }

    // 4. Add traceability metadata.
    let drop_str: Vec<String> = plan
        .dropped_blocks
        .iter()
        .map(|i| i.to_string())
        .collect();
    let mut trace_kvs = vec![
        (
            "timestamp",
            MetaValue::String(chrono_now_for_metadata()),
        ),
        (
            "dropped_blocks",
            MetaValue::String(drop_str.join(",")),
        ),
        (
            "original_block_count",
            MetaValue::U32(plan.original_block_count as u32),
        ),
        (
            "new_block_count",
            MetaValue::U32(plan.new_block_count as u32),
        ),
        ("method", MetaValue::String("prune".into())),
    ];
    // Add kept/dropped tensor counts
    let kept = plan.keep.iter().filter(|(_, k)| *k).count();
    let dropped = plan.keep.len() - kept;
    trace_kvs.push(("tensors_kept", MetaValue::U32(kept as u32)));
    trace_kvs.push(("tensors_dropped", MetaValue::U32(dropped as u32)));

    for (k, v) in trace_kvs {
        let vt = gguf_value_type(&v);
        writer.add_kv(crate::formats::gguf::types::MetadataKv {
            key: format!("tensorkit.prune.{k}"),
            value_type: vt,
            value: v,
        });
    }

    let name_to_idx: std::collections::HashMap<&str, &crate::formats::gguf::types::TensorInfo> =
        gg.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut kept_count = 0usize;
    let mut dropped_count = 0usize;
    for (name, k) in &plan.keep {
        if !*k {
            dropped_count += 1;
            continue;
        }
        let ti = name_to_idx
            .get(name.as_str())
            .copied()
            .ok_or_else(|| Error::TensorNotFound(name.clone()))?;
        let bytes = gg
            .tensor_slice(ti)
            .ok_or_else(|| Error::Gguf("tensor not in mmap".into()))?;
        let new_name = rename_block(name, &plan.remap);
        writer.add_tensor(new_name, ti.n_dims, ti.dims, ti.ggml_type, bytes);
        kept_count += 1;
    }

    let bytes_in: u64 = gg.tensors.iter().map(|t| t.byte_size).sum();
    let bytes_out: u64 = writer.tensors.iter().map(|t| t.byte_size).sum();

    let out_bytes = writer.into_bytes()?;
    let mut out_file = std::fs::File::create(dst)?;
    out_file.write_all(&out_bytes)?;
    out_file.sync_all()?;

    Ok(PruneReport {
        bytes_in,
        bytes_out,
        tensors_kept: kept_count,
        tensors_dropped: dropped_count,
        blocks_dropped: plan.dropped_blocks.clone(),
        original_block_count: plan.original_block_count,
        new_block_count: plan.new_block_count,
        output_path: dst.display().to_string(),
    })
}

pub fn apply_to_safetensors(
    st: &SafetensorsFile,
    plan: &PrunePlan,
    dst: &Path,
) -> Result<PruneReport> {
    // Carry forward __metadata__ from source, updated for the prune.
    let mut writer = if let Some(ref meta) = st.metadata {
        SafetensorsWriter::with_metadata(meta.clone())
    } else {
        SafetensorsWriter::new()
    };

    // Add/update prune traceability in __metadata__.
    let drop_str: Vec<String> = plan
        .dropped_blocks
        .iter()
        .map(|i| i.to_string())
        .collect();
    writer.set_metadata(
        "tensorkit.prune.dropped_blocks",
        serde_json::Value::String(drop_str.join(",")),
    );
    writer.set_metadata(
        "tensorkit.prune.original_block_count",
        serde_json::json!(plan.original_block_count),
    );
    writer.set_metadata(
        "tensorkit.prune.new_block_count",
        serde_json::json!(plan.new_block_count),
    );
    writer.set_metadata(
        "tensorkit.prune.method",
        serde_json::Value::String("prune".into()),
    );
    writer.set_metadata(
        "tensorkit.prune.timestamp",
        serde_json::Value::String(chrono_now_for_metadata()),
    );

    let kept = plan.keep.iter().filter(|(_, k)| *k).count();
    let dropped = plan.keep.len() - kept;
    writer.set_metadata("tensorkit.prune.tensors_kept", serde_json::json!(kept));
    writer.set_metadata(
        "tensorkit.prune.tensors_dropped",
        serde_json::json!(dropped),
    );

    // Update block_count and tensor_count in metadata if present.
    if writer.metadata.is_some()
        && let Some(m) = writer.metadata.as_mut()
            && let Some(arch) = st.architecture() {
                let bc_key = format!("{arch}.block_count");
                m.insert(bc_key, serde_json::json!(plan.new_block_count));
                let tc_key = format!("{arch}.tensor_count");
                m.insert(tc_key, serde_json::json!(kept));
            }

    let mut kept_count = 0usize;
    let mut dropped_count = 0usize;
    for (name, k) in &plan.keep {
        if !*k {
            dropped_count += 1;
            continue;
        }
        let t = st
            .tensor(name)
            .ok_or_else(|| Error::TensorNotFound(name.clone()))?;
        let bytes = st.read_tensor_bytes(name)?;
        let new_name = rename_block(name, &plan.remap);
        writer.add_raw(new_name, t.dtype, t.shape.clone(), &bytes);
        kept_count += 1;
    }
    let bytes_in: u64 = st.tensors.iter().map(|t| t.byte_size).sum();
    let out_file = std::fs::File::create(dst)?;
    writer.write_to(&out_file)?;
    out_file.sync_all()?;

    let bytes_out = std::fs::metadata(dst)?.len();
    Ok(PruneReport {
        bytes_in,
        bytes_out,
        tensors_kept: kept_count,
        tensors_dropped: dropped_count,
        blocks_dropped: plan.dropped_blocks.clone(),
        original_block_count: plan.original_block_count,
        new_block_count: plan.new_block_count,
        output_path: dst.display().to_string(),
    })
}

pub fn apply_to_onnx(onnx: &OnnxFile, plan: &PrunePlan, dst: &Path) -> Result<PruneReport> {
    let mut writer = OnnxWriter::new();

    // Carry forward producer/graph metadata from source.
    if let Some(name) = onnx.name() {
        writer = writer.producer(name, "");
    }
    if let Some(graph_name) = onnx.proto.graph.as_ref().map(|g| g.name.clone())
        && !graph_name.is_empty() {
            writer = writer.graph_name(&graph_name);
        }
    // Carry forward ONNX metadata properties.
    for prop in &onnx.proto.metadata_props {
        writer.add_metadata(&prop.key, &prop.value);
    }
    // Add prune traceability.
    let drop_str: Vec<String> = plan
        .dropped_blocks
        .iter()
        .map(|i| i.to_string())
        .collect();
    writer.add_metadata("tensorkit.prune.dropped_blocks", &drop_str.join(","));
    writer.add_metadata(
        "tensorkit.prune.original_block_count",
        &plan.original_block_count.to_string(),
    );
    writer.add_metadata(
        "tensorkit.prune.new_block_count",
        &plan.new_block_count.to_string(),
    );
    writer.add_metadata("tensorkit.prune.method", "prune");

    let mut kept_count = 0usize;
    let mut dropped_count = 0usize;

    // For ONNX, block naming is ONNX-style (layer.N.*), not GGUF-style (blk.N.*).
    // The rename_block function handles both by using ONNX naming preserved
    // in the tensor names.
    for (name, keep) in &plan.keep {
        if !*keep {
            dropped_count += 1;
            continue;
        }
        let t = onnx
            .tensor(name)
            .ok_or_else(|| Error::TensorNotFound(name.clone()))?;
        let bytes = onnx.read_tensor_bytes(name)?;
        let new_name = rename_block(name, &plan.remap);

        // Map TensorDtype to ONNX data type int
        let data_type = match t.dtype {
            TensorDtype::F32 => 1,
            TensorDtype::F16 => 10,
            TensorDtype::Bf16 => 16,
            TensorDtype::F64 => 11,
            TensorDtype::I8 => 3,
            TensorDtype::I16 => 5,
            TensorDtype::I32 => 6,
            TensorDtype::I64 => 7,
            TensorDtype::Unknown(0) => 9,
            TensorDtype::Unknown(8) => 2,
            TensorDtype::Unknown(16) => 4,
            TensorDtype::Unknown(32) => 12,
            TensorDtype::Unknown(64) => 13,
            _ => {
                return Err(Error::Onnx(format!(
                    "unsupported dtype {:?} for ONNX output tensor '{}'",
                    t.dtype, name
                )));
            }
        };
        let shape_i64: Vec<i64> = t.shape.iter().map(|&d| d as i64).collect();
        writer.add_raw(&new_name, data_type, &shape_i64, &bytes);
        kept_count += 1;
    }

    let bytes_in: u64 = onnx.tensors.iter().map(|t| t.byte_size).sum();
    let out_file = std::fs::File::create(dst)?;
    writer.write_to(&out_file)?;
    out_file.sync_all()?;

    let bytes_out = std::fs::metadata(dst)?.len();
    Ok(PruneReport {
        bytes_in,
        bytes_out,
        tensors_kept: kept_count,
        tensors_dropped: dropped_count,
        blocks_dropped: plan.dropped_blocks.clone(),
        original_block_count: plan.original_block_count,
        new_block_count: plan.new_block_count,
        output_path: dst.display().to_string(),
    })
}

pub fn rename_block(name: &str, remap: &std::collections::HashMap<i32, i32>) -> String {
    if !name.starts_with("blk.") {
        return name.to_string();
    }
    let rest = &name[4..];
    let mut parts = rest.splitn(2, '.');
    let idx_str = match parts.next() {
        Some(s) => s,
        None => return name.to_string(),
    };
    let rest2 = parts.next().unwrap_or("");
    let idx: i32 = match idx_str.parse() {
        Ok(i) => i,
        Err(_) => return name.to_string(),
    };
    match remap.get(&idx) {
        Some(new_idx) => format!("blk.{new_idx}.{rest2}"),
        None => name.to_string(),
    }
}

/// Re-index a metadata key containing `blk.{N}.` to use the new block index.
pub fn rename_metadata_block_key(key: &str, old_idx: i32, new_idx: i32) -> String {
    let blk_tag = format!("blk.{old_idx}.");
    if let Some(pos) = key.find(&blk_tag) {
        let prefix = &key[..pos];
        let suffix = &key[pos + blk_tag.len()..];
        format!("{prefix}blk.{new_idx}.{suffix}")
    } else {
        key.to_string()
    }
}

/// Generate a UTC timestamp string for metadata (without pulling in `chrono`).
fn chrono_now_for_metadata() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_block_key_basic() {
        let r = parse_block_key("llama.blk.3.attn_q.weight").unwrap();
        assert_eq!(r.0, "llama.");
        assert_eq!(r.1, 3);
        assert_eq!(r.2, ".attn_q.weight");
    }

    #[test]
    fn parse_block_key_no_block() {
        assert!(parse_block_key("output.weight").is_none());
    }

    #[test]
    fn parse_block_key_large_index() {
        let r = parse_block_key("blk.127.ffn_up.weight").unwrap();
        assert_eq!(r.0, "");
        assert_eq!(r.1, 127);
        assert_eq!(r.2, ".ffn_up.weight");
    }

    #[test]
    fn shrink_array_correctness() {
        let arr = ArrayValue {
            elem_type: 4, // U32
            elements: (0..10)
                .map(|i| MetaValue::U32(i as u32))
                .collect(),
        };
        let result = shrink_array(&arr, &[2, 5, 8]);
        assert_eq!(result.elements.len(), 7);
        // Check remaining values: 0,1,3,4,6,7,9
        let vals: Vec<u32> = result
            .elements
            .iter()
            .map(|e| match e {
                MetaValue::U32(v) => *v,
                _ => panic!("expected U32"),
            })
            .collect();
        assert_eq!(vals, vec![0, 1, 3, 4, 6, 7, 9]);
    }

    #[test]
    fn shrink_array_empty_drop() {
        let arr = ArrayValue {
            elem_type: 4,
            elements: vec![MetaValue::U32(1), MetaValue::U32(2)],
        };
        let result = shrink_array(&arr, &[]);
        assert_eq!(result.elements.len(), 2);
    }

    #[test]
    fn shrink_array_drop_all() {
        let arr = ArrayValue {
            elem_type: 4,
            elements: vec![MetaValue::U32(1), MetaValue::U32(2), MetaValue::U32(3)],
        };
        let result = shrink_array(&arr, &[0, 1, 2]);
        assert!(result.elements.is_empty());
    }

    #[test]
    fn rename_block_tensor_name() {
        let mut remap = std::collections::HashMap::new();
        remap.insert(7, 3);
        assert_eq!(rename_block("blk.7.ffn_up.weight", &remap), "blk.3.ffn_up.weight");
    }

    #[test]
    fn rename_block_no_blk_prefix() {
        let remap = std::collections::HashMap::new();
        assert_eq!(rename_block("token_embd.weight", &remap), "token_embd.weight");
    }

    #[test]
    fn rename_block_no_remap_entry() {
        let remap = std::collections::HashMap::new();
        assert_eq!(rename_block("blk.5.attn_q.weight", &remap), "blk.5.attn_q.weight");
    }

    #[test]
    fn rename_metadata_block_key_basic() {
        let result = rename_metadata_block_key("llama.blk.5.rope_freqs", 5, 3);
        assert_eq!(result, "llama.blk.3.rope_freqs");
    }

    #[test]
    fn rename_metadata_block_key_no_match() {
        let result = rename_metadata_block_key("llama.output.weight", 5, 3);
        assert_eq!(result, "llama.output.weight");
    }
}