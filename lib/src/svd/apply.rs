//! Apply an `SvdPlan` to a model file (GGUF or safetensors) and write the
//! compressed model to disk.
//!
//! ## Output structure
//!
//! For each tensor `blk.N.<role>.weight` of shape `m x n` targeted by the plan:
//!
//! ```text
//!   blk.N.<role>.weight          — REMOVED
//!   blk.N.<role>.weight.svd_a    — NEW, shape [m, k], dtype in {F32,F16,BF16}
//!   blk.N.<role>.weight.svd_b    — NEW, shape [k, n], dtype in {F32,F16,BF16}
//! ```
//!
//! Reconstruction: `original ≈ svd_a @ svd_b` (with the chosen dtype).
//!
//! ## Metadata (GGUF)
//!
//! We add the following `MetaValue` pairs to the output file:
//!
//! * `tensorkit.svd.applied`             — bool true
//! * `tensorkit.svd.method`              — string (jacobi / randomized)
//! * `tensorkit.svd.output_dtype`        — string
//! * `tensorkit.svd.targets`             — u32
//! * `tensorkit.svd.orig_bytes`          — u64
//! * `tensorkit.svd.new_bytes`           — u64
//! * `tensorkit.svd.compression_ratio`   — f32
//! * `tensorkit.svd.<name>.rank`         — u32
//! * `tensorkit.svd.<name>.shape`        — array[f32] of [m, n, k]
//! * `tensorkit.svd.<name>.approx_error` — f32 (residual Frobenius ratio)
//!
//! ## Metadata (safetensors)
//!
//! Stored as a `__metadata__` JSON object in the header, with the same keys
//! expressed as a nested JSON tree.

use crate::error::{Error, Result};
use crate::formats::gguf::dequant as gguf_dequant;
use crate::formats::gguf::reader::GgufFile;
use crate::formats::gguf::types::{
    byte_size_for, dims_product, ArrayValue, GgmlType, MetaValue, MetadataKv,
};
use crate::formats::gguf::writer::{GgufWriter, WriterTensor};
use crate::formats::onnx::{OnnxFile, OnnxWriter};
use crate::formats::safetensors::reader::SafetensorsFile;
use crate::formats::safetensors::writer::SafetensorsWriter;
use crate::model::{Model, TensorDtype};
use crate::svd::config::OutputDtype;
use crate::svd::linalg::{pack_lowrank, slice_cols, slice_rows, svd_jacobi, svd_randomized, Mat};
use crate::svd::plan::{SvdPlan, SvdTarget};
use std::io::Write;
use std::path::Path;

/// Per-target stat, reported back to the CLI / API.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SvdApplied {
    pub name: String,
    pub name_a: String,
    pub name_b: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub method: String,
    pub orig_bytes: u64,
    pub new_bytes: u64,
    pub approx_error: f32,
}

#[derive(Debug, serde::Serialize)]
pub struct SvdReport {
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub targets: usize,
    pub applied: Vec<SvdApplied>,
    pub skipped: Vec<(String, String)>,
    pub output_path: String,
    pub orig_tensor_bytes: u64,
    pub new_tensor_bytes: u64,
    pub compression_ratio: f64,
}

// ---- GGUF ----------------------------------------------------------------

pub fn apply_to_gguf(gg: &GgufFile, plan: &SvdPlan, dst: &Path) -> Result<SvdReport> {
    let mut writer = GgufWriter::new(gg.version, gg.alignment);
    for kv in &gg.metadata {
        writer.add_kv(kv.clone());
    }
    for kv in build_svd_kvs(plan) {
        writer.add_kv(kv);
    }

    let targets: std::collections::HashMap<&str, &SvdTarget> =
        plan.targets.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut applied: Vec<SvdApplied> = Vec::with_capacity(plan.targets.len());
    let mut total_orig: u64 = 0;
    let mut total_new: u64 = 0;

    for ti in &gg.tensors {
        if let Some(target) = targets.get(ti.name.as_str()) {
            let bytes = gg
                .tensor_slice(ti)
                .ok_or_else(|| Error::Gguf(format!("tensor '{}' not in mmap", ti.name)))?;
            let deq = dequant_to_f32(ti.ggml_type, bytes).ok_or_else(|| {
                Error::Svd(format!(
                    "unsupported dtype for SVD: {}",
                    ti.ggml_type.as_str()
                ))
            })?;
            if deq.len() != target.m * target.n {
                return Err(Error::Svd(format!(
                    "tensor '{}': dequantized {} elems, expected {}",
                    ti.name,
                    deq.len(),
                    target.m * target.n
                )));
            }
            let a_mat = Mat::from_vec(target.m, target.n, deq);
            let k = if target.k == 0 {
                // Defer to global resolution; the spectrum isn't known yet for Energy.
                plan.config.resolve_rank(
                    &ti.name,
                    crate::analysis::score::classify(&ti.name).1,
                    target.m,
                    target.n,
                    None,
                )
            } else {
                target.k
            };
            let (svd, effective_k) = run_svd(&a_mat, k, &plan.config, &ti.name, target)?;
            let (a_packed, b_packed) = pack_lowrank(&svd);
            // a_packed is m x k_full, b_packed is k_full x n. Truncate to the
            // first `effective_k` columns/rows respectively.
            let a_pack = slice_cols(&a_packed, 0, effective_k);
            let b_pack = slice_rows(&b_packed, 0, effective_k);
            let approx_err = approx_error(&a_mat, &a_pack, &b_pack);

            let a_bytes = encode_factors(&a_pack, plan.config.dtype, target.m, effective_k, ti.ggml_type);
            let b_bytes = encode_factors(&b_pack, plan.config.dtype, effective_k, target.n, ti.ggml_type);
            let a_ty = dtype_to_ggml(plan.config.dtype, ti.ggml_type);
            let a_offset = writer.data.len() as u64;
            let a_n = (target.m * effective_k) as u64;
            let a_bz = a_bytes.len() as u64;
            debug_assert_eq!(a_bz, byte_size_for(a_n, a_ty));
            writer.data.extend_from_slice(&a_bytes);
            writer.tensors.push(WriterTensor {
                name: target.name_a.clone(),
                n_dims: 2,
                dims: [target.m as u64, effective_k as u64, 1, 1],
                ggml_type: a_ty,
                offset: a_offset,
                n_elements: a_n,
                byte_size: a_bz,
            });
            let b_offset = writer.data.len() as u64;
            let b_n = (effective_k * target.n) as u64;
            let b_bz = b_bytes.len() as u64;
            debug_assert_eq!(b_bz, byte_size_for(b_n, a_ty));
            writer.data.extend_from_slice(&b_bytes);
            writer.tensors.push(WriterTensor {
                name: target.name_b.clone(),
                n_dims: 2,
                dims: [effective_k as u64, target.n as u64, 1, 1],
                ggml_type: a_ty,
                offset: b_offset,
                n_elements: b_n,
                byte_size: b_bz,
            });

            let new_bytes = a_bz + b_bz;
            total_orig += target.orig_bytes;
            total_new += new_bytes;
            let method = if plan.config.randomized
                && (target.m * target.n) >= plan.config.randomized_min_elems
            {
                "randomized".to_string()
            } else {
                "jacobi".to_string()
            };
            applied.push(SvdApplied {
                name: ti.name.clone(),
                name_a: target.name_a.clone(),
                name_b: target.name_b.clone(),
                m: target.m,
                n: target.n,
                k: effective_k,
                method,
                orig_bytes: target.orig_bytes,
                new_bytes,
                approx_error: approx_err,
            });
        } else {
            // Pass-through: rewrite the tensor verbatim.
            let bytes = gg
                .tensor_slice(ti)
                .ok_or_else(|| Error::Gguf(format!("tensor '{}' not in mmap", ti.name)))?;
            writer.add_tensor(ti.name.clone(), ti.n_dims, ti.dims, ti.ggml_type, bytes);
        }
    }

    let bytes_in: u64 = gg.tensors.iter().map(|t| t.byte_size).sum();
    let out_bytes = writer.into_bytes()?;
    let mut f = std::fs::File::create(dst)?;
    f.write_all(&out_bytes)?;
    f.sync_all()?;
    let bytes_out = out_bytes.len() as u64;

    Ok(SvdReport {
        bytes_in,
        bytes_out,
        targets: plan.targets.len(),
        applied,
        skipped: plan
            .skipped
            .iter()
            .map(|s| (s.name.clone(), s.reason.clone()))
            .collect(),
        output_path: dst.display().to_string(),
        orig_tensor_bytes: total_orig,
        new_tensor_bytes: total_new,
        compression_ratio: if total_orig == 0 {
            0.0
        } else {
            1.0 - (total_new as f64 / total_orig as f64)
        },
    })
}

// ---- Safetensors ---------------------------------------------------------

pub fn apply_to_safetensors(st: &SafetensorsFile, plan: &SvdPlan, dst: &Path) -> Result<SvdReport> {
    let targets: std::collections::HashMap<&str, &SvdTarget> =
        plan.targets.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut writer = SafetensorsWriter::new();
    let mut applied: Vec<SvdApplied> = Vec::with_capacity(plan.targets.len());
    let mut total_orig: u64 = 0;
    let mut total_new: u64 = 0;

    for t in &st.tensors {
        if let Some(target) = targets.get(t.name.as_str()) {
            let bytes = st.read_tensor_bytes(&t.name)?;
            let f32_vals = match t.dtype {
                TensorDtype::F32 => bytes_to_f32(&bytes),
                TensorDtype::F16 => bytes_to_f32_from_f16(&bytes),
                TensorDtype::Bf16 => bytes_to_f32_from_bf16(&bytes),
                TensorDtype::F64 => {
                    let mut out = Vec::with_capacity(bytes.len() / 8);
                    for c in bytes.chunks_exact(8) {
                        out.push(f64::from_le_bytes([
                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                        ]) as f32);
                    }
                    out
                }
                other => {
                    return Err(Error::Svd(format!(
                        "unsupported safetensors dtype for SVD: {:?}",
                        other
                    )))
                }
            };
            if f32_vals.len() != target.m * target.n {
                return Err(Error::Svd(format!(
                    "tensor '{}': decoded {} elems, expected {}",
                    t.name,
                    f32_vals.len(),
                    target.m * target.n
                )));
            }
            let a_mat = Mat::from_vec(target.m, target.n, f32_vals);
            let k = if target.k == 0 {
                plan.config
                    .resolve_rank(&t.name, 0, target.m, target.n, None)
            } else {
                target.k
            };
            let (svd, effective_k) = run_svd(&a_mat, k, &plan.config, &t.name, target)?;
            let (a_packed, b_packed) = pack_lowrank(&svd);
            let a_pack = slice_cols(&a_packed, 0, effective_k);
            let b_pack = slice_rows(&b_packed, 0, effective_k);
            let approx_err = approx_error(&a_mat, &a_pack, &b_pack);

            let src_ggml = tensordtype_to_ggml(t.dtype);
            let a_bytes = encode_factors(&a_pack, plan.config.dtype, target.m, effective_k, src_ggml);
            let b_bytes = encode_factors(&b_pack, plan.config.dtype, effective_k, target.n, src_ggml);
            let dt = plan.config.dtype;
            writer.add_raw(
                target.name_a.clone(),
                dtype_to_tensordtype(dt, src_ggml),
                vec![target.m as u64, effective_k as u64],
                &a_bytes,
            );
            writer.add_raw(
                target.name_b.clone(),
                dtype_to_tensordtype(dt, src_ggml),
                vec![effective_k as u64, target.n as u64],
                &b_bytes,
            );

            let new_bytes = (a_bytes.len() + b_bytes.len()) as u64;
            total_orig += target.orig_bytes;
            total_new += new_bytes;
            let method = if plan.config.randomized
                && (target.m * target.n) >= plan.config.randomized_min_elems
            {
                "randomized".to_string()
            } else {
                "jacobi".to_string()
            };
            applied.push(SvdApplied {
                name: t.name.clone(),
                name_a: target.name_a.clone(),
                name_b: target.name_b.clone(),
                m: target.m,
                n: target.n,
                k: effective_k,
                method,
                orig_bytes: target.orig_bytes,
                new_bytes,
                approx_error: approx_err,
            });
        } else {
            let bytes = st.read_tensor_bytes(&t.name)?;
            writer.add_raw(t.name.clone(), t.dtype, t.shape.clone(), &bytes);
        }
    }

    // Build the metadata block.
    let mut header_meta = serde_json::Map::new();
    header_meta.insert("applied".into(), serde_json::json!(applied.len()));
    header_meta.insert(
        "output_dtype".into(),
        serde_json::json!(plan.config.dtype.as_str()),
    );
    header_meta.insert("targets".into(), serde_json::json!(plan.targets.len()));
    for a in &applied {
        header_meta.insert(
            format!("tensor.{}.rank", a.name),
            serde_json::json!({"k": a.k, "m": a.m, "n": a.n, "approx_error": a.approx_error, "method": a.method}),
        );
    }

    let bytes_in: u64 = st.tensors.iter().map(|t| t.byte_size).sum();
    write_safetensors_with_metadata(&writer, &header_meta, dst)?;
    let bytes_out = std::fs::metadata(dst)?.len();

    Ok(SvdReport {
        bytes_in,
        bytes_out,
        targets: plan.targets.len(),
        applied,
        skipped: plan
            .skipped
            .iter()
            .map(|s| (s.name.clone(), s.reason.clone()))
            .collect(),
        output_path: dst.display().to_string(),
        orig_tensor_bytes: total_orig,
        new_tensor_bytes: total_new,
        compression_ratio: if total_orig == 0 {
            0.0
        } else {
            1.0 - (total_new as f64 / total_orig as f64)
        },
    })
}

// ---- ONNX -----------------------------------------------------------------

pub fn apply_to_onnx(onnx: &OnnxFile, plan: &SvdPlan, dst: &Path) -> Result<SvdReport> {
    let targets: std::collections::HashMap<&str, &SvdTarget> =
        plan.targets.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut writer = OnnxWriter::new();

    // Carry forward metadata.
    if let Some(name) = onnx.name() {
        writer = writer.producer(name, "");
    }
    if let Some(graph_name) = onnx.proto.graph.as_ref().map(|g| g.name.clone()) {
        if !graph_name.is_empty() {
            writer = writer.graph_name(&graph_name);
        }
    }
    for prop in &onnx.proto.metadata_props {
        writer.add_metadata(&prop.key, &prop.value);
    }
    writer.add_metadata("tensorkit.svd.applied", "true");
    writer.add_metadata("tensorkit.svd.output_dtype", plan.config.dtype.as_str());
    writer.add_metadata("tensorkit.svd.targets", &plan.targets.len().to_string());

    let mut applied: Vec<SvdApplied> = Vec::with_capacity(plan.targets.len());
    let mut total_orig: u64 = 0;
    let mut total_new: u64 = 0;

    for t in &onnx.tensors {
        if let Some(target) = targets.get(t.name.as_str()) {
            let bytes = onnx.read_tensor_bytes(&t.name)?;
            let f32_vals = match t.dtype {
                TensorDtype::F32 => bytes_to_f32(&bytes),
                TensorDtype::F16 => bytes_to_f32_from_f16(&bytes),
                TensorDtype::Bf16 => bytes_to_f32_from_bf16(&bytes),
                TensorDtype::F64 => {
                    let mut out = Vec::with_capacity(bytes.len() / 8);
                    for c in bytes.chunks_exact(8) {
                        out.push(f64::from_le_bytes([
                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                        ]) as f32);
                    }
                    out
                }
                other => {
                    return Err(Error::Svd(format!(
                        "unsupported ONNX dtype for SVD: {:?}",
                        other
                    )))
                }
            };
            if f32_vals.len() != target.m * target.n {
                return Err(Error::Svd(format!(
                    "tensor '{}': decoded {} elems, expected {}",
                    t.name,
                    f32_vals.len(),
                    target.m * target.n
                )));
            }
            let a_mat = Mat::from_vec(target.m, target.n, f32_vals);
            let k = if target.k == 0 {
                plan.config
                    .resolve_rank(&t.name, 0, target.m, target.n, None)
            } else {
                target.k
            };
            let (svd, effective_k) = run_svd(&a_mat, k, &plan.config, &t.name, target)?;
            let (a_packed, b_packed) = pack_lowrank(&svd);
            let a_pack = slice_cols(&a_packed, 0, effective_k);
            let b_pack = slice_rows(&b_packed, 0, effective_k);
            let approx_err = approx_error(&a_mat, &a_pack, &b_pack);

            let src_ggml = tensordtype_to_ggml(t.dtype);
            let a_bytes = encode_factors(&a_pack, plan.config.dtype, target.m, effective_k, src_ggml);
            let b_bytes = encode_factors(&b_pack, plan.config.dtype, effective_k, target.n, src_ggml);
            let out_dt = dtype_to_tensordtype(plan.config.dtype, src_ggml);

            // Write SVD factors via add_tensor to get proper ONNX dtype.
            let a_shape: Vec<u64> = vec![target.m as u64, effective_k as u64];
            writer.add_tensor(&target.name_a, out_dt, &a_shape, &a_bytes)?;
            let b_shape: Vec<u64> = vec![effective_k as u64, target.n as u64];
            writer.add_tensor(&target.name_b, out_dt, &b_shape, &b_bytes)?;

            let new_bytes = (a_bytes.len() + b_bytes.len()) as u64;
            total_orig += target.orig_bytes;
            total_new += new_bytes;
            let method = if plan.config.randomized
                && (target.m * target.n) >= plan.config.randomized_min_elems
            {
                "randomized".to_string()
            } else {
                "jacobi".to_string()
            };
            applied.push(SvdApplied {
                name: t.name.clone(),
                name_a: target.name_a.clone(),
                name_b: target.name_b.clone(),
                m: target.m,
                n: target.n,
                k: effective_k,
                method,
                orig_bytes: target.orig_bytes,
                new_bytes,
                approx_error: approx_err,
            });
        } else {
            let bytes = onnx.read_tensor_bytes(&t.name)?;
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
                    return Err(Error::Svd(format!(
                        "unsupported ONNX dtype for passthrough: {:?}",
                        t.dtype
                    )))
                }
            };
            let shape_i64: Vec<i64> = t.shape.iter().map(|&d| d as i64).collect();
            writer.add_raw(&t.name, data_type, &shape_i64, &bytes);
        }
    }

    let bytes_in: u64 = onnx.tensors.iter().map(|t| t.byte_size).sum();
    let out_file = std::fs::File::create(dst)?;
    writer.write_to(&out_file)?;
    out_file.sync_all()?;
    let bytes_out = std::fs::metadata(dst)?.len();

    Ok(SvdReport {
        bytes_in,
        bytes_out,
        targets: plan.targets.len(),
        applied,
        skipped: plan
            .skipped
            .iter()
            .map(|s| (s.name.clone(), s.reason.clone()))
            .collect(),
        output_path: dst.display().to_string(),
        orig_tensor_bytes: total_orig,
        new_tensor_bytes: total_new,
        compression_ratio: if total_orig == 0 {
            0.0
        } else {
            1.0 - (total_new as f64 / total_orig as f64)
        },
    })
}

// ---- shared helpers ------------------------------------------------------

fn run_svd(
    a_mat: &Mat,
    requested_k: usize,
    cfg: &crate::svd::config::SvdConfig,
    name: &str,
    target: &SvdTarget,
) -> Result<(crate::svd::linalg::Svd, usize)> {
    use crate::svd::config::RankSpec;
    let use_randomized = cfg.randomized && (a_mat.rows * a_mat.cols) >= cfg.randomized_min_elems;
    match &cfg.rank.spec {
        RankSpec::Energy(_) => {
            // For Energy, run SVD at full rank to get the spectrum, then resolve k.
            let full_k = a_mat.rows.min(a_mat.cols);
            let svd_full = if use_randomized {
                svd_randomized(
                    a_mat,
                    full_k,
                    cfg.randomized_oversample,
                    cfg.randomized_power_iters,
                    hash_seed(name, target),
                )?
            } else {
                svd_jacobi(a_mat, 20, 1e-6)?
            };
            let k = cfg.resolve_rank(
                name,
                crate::analysis::score::classify(name).1,
                a_mat.rows,
                a_mat.cols,
                Some(&svd_full.s),
            );
            Ok((svd_full, k))
        }
        _ => {
            let k = requested_k.max(1).min(a_mat.rows).min(a_mat.cols);
            let svd = if use_randomized {
                svd_randomized(
                    a_mat,
                    k,
                    cfg.randomized_oversample,
                    cfg.randomized_power_iters,
                    hash_seed(name, target),
                )?
            } else {
                svd_jacobi(a_mat, 20, 1e-6)?
            };
            Ok((svd, k))
        }
    }
}

fn dequant_to_f32(ty: GgmlType, bytes: &[u8]) -> Option<Vec<f32>> {
    gguf_dequant::dequantize(ty, bytes, None)
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    out
}

fn bytes_to_f32_from_f16(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        out.push(gguf_dequant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
    }
    out
}

fn bytes_to_f32_from_bf16(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for c in bytes.chunks_exact(2) {
        out.push(gguf_dequant::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
    }
    out
}

/// Encode a row-major f32 matrix into the requested on-disk element dtype.
fn encode_factors(
    m: &Mat,
    dtype: OutputDtype,
    rows: usize,
    cols: usize,
    src_ggml: GgmlType,
) -> Vec<u8> {
    debug_assert_eq!(m.rows, rows);
    debug_assert_eq!(m.cols, cols);
    let resolved = match dtype {
        OutputDtype::F32 => return encode_f32(m),
        OutputDtype::F16 => return encode_f16(m),
        OutputDtype::Bf16 => return encode_bf16(m),
        OutputDtype::AutoQuant => auto_pick_quant(src_ggml),
        OutputDtype::Ggml(t) => t,
    };
    if !crate::quantize::is_quantizable(resolved) {
        // Not a quantizable block type — fall back to F16.
        return encode_f16(m);
    }
    crate::quantize::quantize_par(&m.data, resolved)
}

fn encode_f32(m: &Mat) -> Vec<u8> {
    let mut out = Vec::with_capacity(m.data.len() * 4);
    for &v in &m.data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
fn encode_f16(m: &Mat) -> Vec<u8> {
    let mut out = Vec::with_capacity(m.data.len() * 2);
    for &v in &m.data {
        out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
    }
    out
}
fn encode_bf16(m: &Mat) -> Vec<u8> {
    let mut out = Vec::with_capacity(m.data.len() * 2);
    for &v in &m.data {
        out.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
    }
    out
}

#[inline]
fn f32_to_f16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    if v.is_infinite() {
        return sign | 0x7c00;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = (mant | 0x400) >> (1 - exp);
        return sign | (mant as u16);
    }
    sign | ((exp as u16) << 10) | (mant as u16)
}

#[inline]
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

fn dtype_to_ggml(d: OutputDtype, src_ggml: GgmlType) -> GgmlType {
    match d {
        OutputDtype::F32 => GgmlType::F32,
        OutputDtype::F16 => GgmlType::F16,
        OutputDtype::Bf16 => GgmlType::Bf16,
        OutputDtype::AutoQuant => auto_pick_quant(src_ggml),
        OutputDtype::Ggml(t) => t,
    }
}

/// Pick a sensible quantization format for a quantized-output SVD factor.
/// For float source tensors the SVD factors are continuous-valued, so we
/// keep the high precision in F16 (4-byte-per-factor with 5× compression
/// would be a worse trade than the rank reduction itself). For already-
/// quantized sources we re-use the source precision to keep the
/// reconstruction error in the same band.
fn auto_pick_quant(src: GgmlType) -> GgmlType {
    match src {
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 | GgmlType::F64
        | GgmlType::I8 | GgmlType::I16 | GgmlType::I32 | GgmlType::I64 => GgmlType::F16,
        _ => GgmlType::Q8_0, // matches the source's accuracy tier for the common case
    }
}

fn dtype_to_tensordtype(d: OutputDtype, src_ggml: GgmlType) -> TensorDtype {
    use crate::model::TensorDtype;
    match dtype_to_ggml(d, src_ggml) {
        GgmlType::F32 => TensorDtype::F32,
        GgmlType::F16 => TensorDtype::F16,
        GgmlType::Bf16 => TensorDtype::Bf16,
        GgmlType::F64 => TensorDtype::F64,
        GgmlType::I8 => TensorDtype::I8,
        GgmlType::I16 => TensorDtype::I16,
        GgmlType::I32 => TensorDtype::I32,
        GgmlType::I64 => TensorDtype::I64,
        // Safetensors doesn't support GGUF block types; callers are expected
        // to have already routed GGML-quant requests through the GGUF path.
        other => TensorDtype::Unknown(other.as_str().parse().unwrap_or(0)),
    }
}

/// Map a `TensorDtype` (safetensors) to the equivalent `GgmlType` so the
/// auto-quant heuristic and `dtype_to_tensordtype` work uniformly across
/// both formats. Unknown / unsupported tensor dtypes fall back to `F16`.
fn tensordtype_to_ggml(t: TensorDtype) -> GgmlType {
    match t {
        TensorDtype::F32 => GgmlType::F32,
        TensorDtype::F16 => GgmlType::F16,
        TensorDtype::Bf16 => GgmlType::Bf16,
        TensorDtype::F64 => GgmlType::F64,
        TensorDtype::I8 => GgmlType::I8,
        TensorDtype::I16 => GgmlType::I16,
        TensorDtype::I32 => GgmlType::I32,
        TensorDtype::I64 => GgmlType::I64,
        TensorDtype::Q4_0 => GgmlType::Q4_0,
        TensorDtype::Q4_1 => GgmlType::Q4_1,
        TensorDtype::Q5_0 => GgmlType::Q5_0,
        TensorDtype::Q5_1 => GgmlType::Q5_1,
        TensorDtype::Q8_0 => GgmlType::Q8_0,
        TensorDtype::Q8_1 => GgmlType::Q8_1,
        TensorDtype::Q2K => GgmlType::Q2K,
        TensorDtype::Q3K => GgmlType::Q3K,
        TensorDtype::Q4K => GgmlType::Q4K,
        TensorDtype::Q5K => GgmlType::Q5K,
        TensorDtype::Q6K => GgmlType::Q6K,
        TensorDtype::Q8K => GgmlType::Q8K,
        TensorDtype::Iq2Xxs => GgmlType::Iq2Xxs,
        TensorDtype::Iq2Xs => GgmlType::Iq2Xs,
        TensorDtype::Iq3Xxs => GgmlType::Iq3Xxs,
        TensorDtype::Iq3S => GgmlType::Iq3S,
        TensorDtype::Iq4Nl => GgmlType::Iq4Nl,
        TensorDtype::Iq4Xs => GgmlType::Iq4Xs,
        TensorDtype::Iq1S => GgmlType::Iq1S,
        TensorDtype::Iq2S => GgmlType::Iq2S,
        TensorDtype::Iq1M => GgmlType::Iq1M,
        TensorDtype::Tq1_0 => GgmlType::Tq1_0,
        TensorDtype::Tq2_0 => GgmlType::Tq2_0,
        TensorDtype::Unknown(_) => GgmlType::F16,
    }
}

fn approx_error(original: &Mat, a: &Mat, b: &Mat) -> f32 {
    if original.data.is_empty() {
        return 0.0;
    }
    let mut recon = Mat::new(original.rows, original.cols);
    Mat::matmul_into(a, b, &mut recon);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..original.data.len() {
        let d = original.data[i] as f64 - recon.data[i] as f64;
        num += d * d;
        den += (original.data[i] as f64) * (original.data[i] as f64);
    }
    if den == 0.0 {
        0.0
    } else {
        (num / den).sqrt() as f32
    }
}

fn hash_seed(name: &str, target: &SvdTarget) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    for &x in &[target.m, target.n, target.k] {
        for b in x.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

fn build_svd_kvs(plan: &SvdPlan) -> Vec<MetadataKv> {
    let mut kvs = Vec::new();
    let method_str = match &plan.config.rank.spec {
        crate::svd::config::RankSpec::Absolute(_) => "jacobi",
        crate::svd::config::RankSpec::Fraction(_) => "jacobi",
        crate::svd::config::RankSpec::Energy(_) => "jacobi+energy",
    };
    kvs.push(MetadataKv {
        key: "tensorkit.svd.applied".into(),
        value_type: 7,
        value: MetaValue::Bool(true),
    });
    kvs.push(MetadataKv {
        key: "tensorkit.svd.method".into(),
        value_type: 8,
        value: MetaValue::String(method_str.into()),
    });
    kvs.push(MetadataKv {
        key: "tensorkit.svd.output_dtype".into(),
        value_type: 8,
        value: MetaValue::String(plan.config.dtype.as_str().into()),
    });
    kvs.push(MetadataKv {
        key: "tensorkit.svd.targets".into(),
        value_type: 4,
        value: MetaValue::U32(plan.targets.len() as u32),
    });
    let total_orig: u64 = plan.targets.iter().map(|t| t.orig_bytes).sum();
    let total_new: u64 = plan.targets.iter().map(|t| t.new_bytes).sum();
    kvs.push(MetadataKv {
        key: "tensorkit.svd.orig_bytes".into(),
        value_type: 10,
        value: MetaValue::U64(total_orig),
    });
    kvs.push(MetadataKv {
        key: "tensorkit.svd.new_bytes".into(),
        value_type: 10,
        value: MetaValue::U64(total_new),
    });
    let ratio = if total_orig == 0 {
        0.0
    } else {
        1.0 - (total_new as f64 / total_orig as f64)
    };
    kvs.push(MetadataKv {
        key: "tensorkit.svd.compression_ratio".into(),
        value_type: 6,
        value: MetaValue::F32(ratio as f32),
    });

    for t in &plan.targets {
        let key = format!("tensorkit.svd.{}.rank", t.name);
        kvs.push(MetadataKv {
            key,
            value_type: 4,
            value: MetaValue::U32(t.k as u32),
        });
        let key = format!("tensorkit.svd.{}.shape", t.name);
        let elems = vec![
            MetaValue::F32(t.m as f32),
            MetaValue::F32(t.n as f32),
            MetaValue::F32(t.k as f32),
        ];
        kvs.push(MetadataKv {
            key,
            value_type: 9,
            value: MetaValue::Array(ArrayValue {
                elem_type: 6,
                elements: elems,
            }),
        });
    }
    kvs
}

/// Write a safetensors file with a `__metadata__` block in the JSON header.
fn write_safetensors_with_metadata(
    w: &SafetensorsWriter,
    meta: &serde_json::Map<String, serde_json::Value>,
    dst: &Path,
) -> Result<()> {
    let mut offset: u64 = 0;
    let mut header_obj: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for t in &w.tensors {
        let start = offset;
        let end = start + t.bytes.len() as u64;
        header_obj.insert(
            t.name.clone(),
            serde_json::json!({
                "dtype": dt_str(t.dtype),
                "shape": t.shape,
                "data_offsets": [start, end],
            }),
        );
        offset = end;
    }
    if !meta.is_empty() {
        header_obj.insert(
            "__metadata__".into(),
            serde_json::Value::Object(meta.clone()),
        );
    }
    let header_json = serde_json::to_vec(&header_obj)?;
    let mut f = std::fs::File::create(dst)?;
    f.write_all(&(header_json.len() as u64).to_le_bytes())?;
    f.write_all(&header_json)?;
    for t in &w.tensors {
        f.write_all(&t.bytes)?;
    }
    f.sync_all()?;
    Ok(())
}

fn dt_str(d: TensorDtype) -> &'static str {
    match d {
        TensorDtype::F32 => "F32",
        TensorDtype::F16 => "F16",
        TensorDtype::Bf16 => "BF16",
        TensorDtype::F64 => "F64",
        TensorDtype::I8 => "I8",
        TensorDtype::I16 => "I16",
        TensorDtype::I32 => "I32",
        TensorDtype::I64 => "I64",
        _ => "U8",
    }
}

// keep the dims_product re-export reachable so this module doesn't go stale
#[allow(dead_code)]
fn _silence(_: u64, _: u32) -> u64 {
    dims_product(&[1, 1, 1, 1], 1)
}
