//! Apply quantization to an existing model file (GGUF only in the first cut).
//!
//! For each tensor in the input file:
//!   - If the tensor is already in the target type, copy verbatim.
//!   - Otherwise, dequantize to `f32` (or decode directly from the source
//!     dtype for non-quantized types) and re-quantize to the target type.
//!
//! The output is written to `dst`, preserving the source file's GGUF version
//! and alignment, and adding a small set of `tensorkit.quantize.*` metadata
//! entries for traceability.

use crate::error::{Error, Result};
use crate::formats::gguf::dequant as gguf_dequant;
use crate::formats::gguf::reader::GgufFile;
use crate::formats::gguf::types::{byte_size_for, dims_product, GgmlType, MetaValue, MetadataKv};
use crate::formats::gguf::writer::GgufWriter;
use crate::quantize;
use std::io::Write;
use std::path::Path;

/// Stats per quantized tensor.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorQuantized {
    pub name: String,
    pub from: String,
    pub to: String,
    pub n_elements: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub max_abs_err: f32,
}

#[derive(Debug, serde::Serialize)]
pub struct QuantizeReport {
    pub input_path: String,
    pub output_path: String,
    pub target: String,
    pub tensors_total: usize,
    pub tensors_quantized: usize,
    pub tensors_passthrough: usize,
    pub tensors_skipped: Vec<(String, String)>,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub compression_ratio: f64,
    pub per_tensor: Vec<TensorQuantized>,
}

/// Quantize every tensor in `src` to `target` and write the result to `dst`.
/// If `blocks` is `Some`, only tensors whose block index (parsed from `blk.N.`)
/// is in the set are quantized; all others are passed through verbatim.
pub fn quantize_gguf(
    src: &Path,
    target: GgmlType,
    dst: &Path,
    blocks: Option<&[i32]>,
) -> Result<QuantizeReport> {
    if !quantize::is_quantizable(target) {
        return Err(Error::Quantize(format!(
            "target type {:?} is not a supported quant type",
            target
        )));
    }
    let gg = GgufFile::open(src)?;
    let mut writer = GgufWriter::new(gg.version, gg.alignment);

    for kv in &gg.metadata {
        writer.add_kv(kv.clone());
    }
    for kv in build_meta(target) {
        writer.add_kv(kv);
    }

    let mut per_tensor = Vec::with_capacity(gg.tensors.len());
    let mut skipped = Vec::new();
    let mut total_in: u64 = 0;
    let mut total_out: u64 = 0;
    let mut n_q = 0;
    let mut n_p = 0;

    for ti in &gg.tensors {
        let src_bytes = gg
            .tensor_slice(ti)
            .ok_or_else(|| Error::Gguf(format!("tensor '{}' not in mmap", ti.name)))?;
        // Block selection: skip tensors not in the selected blocks.
        if let Some(allowed) = blocks
            && let Some(idx) = block_index_from_name(&ti.name)
                && !allowed.contains(&idx) {
                    writer.add_tensor(ti.name.clone(), ti.n_dims, ti.dims, ti.ggml_type, src_bytes);
                    n_p += 1;
                    total_in += ti.byte_size;
                    total_out += ti.byte_size;
                    continue;
                }
        let n_elems = dims_product(&ti.dims, ti.n_dims);
        if ti.ggml_type == target {
            // Pass-through.
            writer.add_tensor(ti.name.clone(), ti.n_dims, ti.dims, ti.ggml_type, src_bytes);
            per_tensor.push(TensorQuantized {
                name: ti.name.clone(),
                from: ti.ggml_type.as_str().to_string(),
                to: target.as_str().to_string(),
                n_elements: n_elems,
                bytes_in: ti.byte_size,
                bytes_out: ti.byte_size,
                max_abs_err: 0.0,
            });
            total_in += ti.byte_size;
            total_out += ti.byte_size;
            n_p += 1;
            continue;
        }
        let deq = match gguf_dequant::dequantize(ti.ggml_type, src_bytes, None) {
            Some(v) => v,
            None => {
                // Source dtype isn't dequantizable (e.g. raw integer tensors).
                // Pass through unchanged.
                writer.add_tensor(ti.name.clone(), ti.n_dims, ti.dims, ti.ggml_type, src_bytes);
                skipped.push((ti.name.clone(), ti.ggml_type.as_str().to_string()));
                total_in += ti.byte_size;
                total_out += ti.byte_size;
                continue;
            }
        };
        if (deq.len() as u64) != n_elems {
            return Err(Error::Quantize(format!(
                "tensor '{}': dequantized {} elems, expected {}",
                ti.name,
                deq.len(),
                n_elems
            )));
        }
        let new_bytes = quantize::quantize(&deq, target);
        let new_bz = new_bytes.len() as u64;
        if new_bz != byte_size_for(n_elems, target) {
            return Err(Error::Quantize(format!(
                "tensor '{}': quantized to {} bytes, expected {} for {} elements at {:?}",
                ti.name,
                new_bz,
                byte_size_for(n_elems, target),
                n_elems,
                target
            )));
        }

        // Round-trip error: dequantize what we just wrote and compare.
        let err = if let Some(recon) = gguf_dequant::dequantize(target, &new_bytes, None) {
            max_abs_diff(&deq, &recon)
        } else {
            0.0
        };

        writer.add_tensor(ti.name.clone(), ti.n_dims, ti.dims, target, &new_bytes);
        per_tensor.push(TensorQuantized {
            name: ti.name.clone(),
            from: ti.ggml_type.as_str().to_string(),
            to: target.as_str().to_string(),
            n_elements: n_elems,
            bytes_in: ti.byte_size,
            bytes_out: new_bz,
            max_abs_err: err,
        });
        total_in += ti.byte_size;
        total_out += new_bz;
        n_q += 1;
    }

    let out_bytes = writer.into_bytes()?;
    let mut f = std::fs::File::create(dst)?;
    f.write_all(&out_bytes)?;
    f.sync_all()?;

    Ok(QuantizeReport {
        input_path: src.display().to_string(),
        output_path: dst.display().to_string(),
        target: target.as_str().to_string(),
        tensors_total: gg.tensors.len(),
        tensors_quantized: n_q,
        tensors_passthrough: n_p,
        tensors_skipped: skipped,
        bytes_in: total_in,
        bytes_out: total_out,
        compression_ratio: if total_out > 0 {
            total_in as f64 / total_out as f64
        } else {
            1.0
        },
        per_tensor,
    })
}

fn build_meta(target: GgmlType) -> Vec<MetadataKv> {
    vec![
        MetadataKv {
            key: "tensorkit.quantize.applied".into(),
            value_type: 7, // GGUF value type for Bool
            value: MetaValue::Bool(true),
        },
        MetadataKv {
            key: "tensorkit.quantize.target".into(),
            value_type: 8, // GGUF value type for String
            value: MetaValue::String(target.as_str().into()),
        },
        MetadataKv {
            key: "tensorkit.quantize.method".into(),
            value_type: 8,
            value: MetaValue::String("simple-per-block".into()),
        },
    ]
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// If `name` has the form `blk.<idx>.<rest>`, returns `Some(idx)`.
pub fn block_index_from_name(name: &str) -> Option<i32> {
    let rest = name.strip_prefix("blk.")?;
    let idx_str = rest.split('.').next()?;
    idx_str.parse().ok()
}
