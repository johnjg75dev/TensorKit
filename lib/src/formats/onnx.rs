//! ONNX model reader and writer.
//!
//! Parses the ONNX protobuf format (`ModelProto`) using `prost` and
//! implements the `Model` trait for integration with the analysis pipeline.
//!
//! The reader supports tensor metadata (names, shapes, dtypes), raw data
//! extraction, and block structure from common naming conventions.
//!
//! The writer (`OnnxWriter`) builds `ModelProto` with tensors, metadata,
//! and graph inputs/outputs, then serializes via prost encoding.

use crate::error::{Error, Result};
use crate::model::{MetadataValue, Model, ModelFormat, Tensor, TensorDtype};
use prost::Message;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Minimal ONNX protobuf types (prost-derived, field numbers match onnx.proto3)
// ---------------------------------------------------------------------------

/// Top-level ONNX model.
#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = 1)]
    pub ir_version: i64,
    #[prost(message, repeated, tag = 2)]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = 3)]
    pub producer_name: String,
    #[prost(string, tag = 4)]
    pub producer_version: String,
    #[prost(string, tag = 5)]
    pub domain: String,
    #[prost(int64, tag = 6)]
    pub model_version: i64,
    #[prost(string, tag = 7)]
    pub doc_string: String,
    #[prost(message, tag = 8)]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = 9)]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// A single key-value metadata entry.
#[derive(Clone, PartialEq, Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = 1)]
    pub key: String,
    #[prost(string, tag = 2)]
    pub value: String,
}

/// The computation graph.
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = 1)]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = 2)]
    pub output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = 3)]
    pub initializer: Vec<TensorProto>,
    #[prost(string, tag = 5)]
    pub name: String,
}

/// A named value (input / output descriptor).
#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = 1)]
    pub name: String,
    #[prost(string, tag = 3)]
    pub doc_string: String,
}

/// Tensor data payload.
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = 1)]
    pub dims: Vec<i64>,
    #[prost(int32, tag = 2)]
    pub data_type: i32,
    #[prost(float, repeated, packed = false, tag = 4)]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, packed = false, tag = 5)]
    pub int32_data: Vec<i32>,
    #[prost(bytes, repeated, tag = 6)]
    pub string_data: Vec<Vec<u8>>,
    #[prost(int64, repeated, packed = false, tag = 7)]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = 8)]
    pub name: String,
    #[prost(bytes, tag = 10)]
    pub raw_data: Vec<u8>,
    #[prost(double, repeated, packed = false, tag = 13)]
    pub double_data: Vec<f64>,
    #[prost(uint64, repeated, packed = false, tag = 14)]
    pub uint64_data: Vec<u64>,
}

/// Operator set identifier.
#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = 1)]
    pub domain: String,
    #[prost(int64, tag = 2)]
    pub version: i64,
}

// ---------------------------------------------------------------------------
// ONNX data-type → TensorDtype mapping
// ---------------------------------------------------------------------------

/// ONNX TensorProto_DataType enum values (from onnx.proto3).
#[allow(non_upper_case_globals)]
mod onnx_dtypes {
    pub const FLOAT: i32 = 1;
    pub const UINT8: i32 = 2;
    pub const INT8: i32 = 3;
    pub const UINT16: i32 = 4;
    pub const INT16: i32 = 5;
    pub const INT32: i32 = 6;
    pub const INT64: i32 = 7;
    #[allow(dead_code)]
    pub const STRING: i32 = 8;
    pub const BOOL: i32 = 9;
    pub const FLOAT16: i32 = 10;
    pub const DOUBLE: i32 = 11;
    pub const UINT32: i32 = 12;
    pub const UINT64: i32 = 13;
    pub const BFLOAT16: i32 = 16;
}

fn onnx_dtype_to_tensor(dt: i32) -> Option<TensorDtype> {
    Some(match dt {
        onnx_dtypes::FLOAT => TensorDtype::F32,
        onnx_dtypes::FLOAT16 => TensorDtype::F16,
        onnx_dtypes::BFLOAT16 => TensorDtype::Bf16,
        onnx_dtypes::DOUBLE => TensorDtype::F64,
        onnx_dtypes::INT8 => TensorDtype::I8,
        onnx_dtypes::INT16 => TensorDtype::I16,
        onnx_dtypes::INT32 => TensorDtype::I32,
        onnx_dtypes::INT64 => TensorDtype::I64,
        onnx_dtypes::UINT8 => TensorDtype::Unknown(8),
        onnx_dtypes::UINT16 => TensorDtype::Unknown(16),
        onnx_dtypes::UINT32 => TensorDtype::Unknown(32),
        onnx_dtypes::UINT64 => TensorDtype::Unknown(64),
        onnx_dtypes::BOOL => TensorDtype::Unknown(0),
        _ => return None,
    })
}

fn byte_size_for_dtype(dt: i32) -> Option<u64> {
    Some(match dt {
        onnx_dtypes::FLOAT => 4,
        onnx_dtypes::FLOAT16 => 2,
        onnx_dtypes::BFLOAT16 => 2,
        onnx_dtypes::DOUBLE => 8,
        onnx_dtypes::INT8 | onnx_dtypes::UINT8 | onnx_dtypes::BOOL => 1,
        onnx_dtypes::INT16 | onnx_dtypes::UINT16 => 2,
        onnx_dtypes::INT32 | onnx_dtypes::UINT32 => 4,
        onnx_dtypes::INT64 | onnx_dtypes::UINT64 => 8,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// OnnxFile
// ---------------------------------------------------------------------------

/// An ONNX model opened for read-only analysis.
pub struct OnnxFile {
    pub proto: ModelProto,
    pub tensors: Vec<Tensor>,
    name_to_idx: HashMap<String, usize>,
}

impl OnnxFile {
    /// Open and parse an ONNX file from disk.
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(Error::Io)?;
        Self::from_bytes(&bytes)
    }

    /// Parse from an in-memory byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let proto = ModelProto::decode(bytes)?;

        let graph = proto
            .graph
            .as_ref()
            .ok_or_else(|| Error::Onnx("no graph in model".into()))?;

        let mut name_to_idx = HashMap::new();
        let mut tensors = Vec::with_capacity(graph.initializer.len());

        for t in &graph.initializer {
            let dtype = onnx_dtype_to_tensor(t.data_type).ok_or_else(|| {
                Error::Onnx(format!("unsupported ONNX data_type {}", t.data_type))
            })?;
            let shape: Vec<u64> = t.dims.iter().map(|&d| d.max(0) as u64).collect();
            let elem_size = byte_size_for_dtype(t.data_type).unwrap_or(4);
            let n_elems: u64 = shape.iter().product();
            let byte_size = n_elems * elem_size;

            let idx = tensors.len();
            name_to_idx.insert(t.name.clone(), idx);
            tensors.push(Tensor {
                name: t.name.clone(),
                dtype,
                shape,
                byte_size,
                data_offset: 0,
            });
        }

        Ok(Self {
            proto,
            tensors,
            name_to_idx,
        })
    }

    /// Iterate over all initializer tensor names (convenience).
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.name_to_idx.keys().map(|s| s.as_str())
    }

    /// Access the raw `TensorProto` by name.
    pub fn tensor_proto(&self, name: &str) -> Option<&TensorProto> {
        let graph = self.proto.graph.as_ref()?;
        graph.initializer.iter().find(|t| t.name == name)
    }

    /// List graph input names (dynamic inputs, not initializers).
    pub fn input_names(&self) -> Vec<&str> {
        self.proto
            .graph
            .as_ref()
            .map(|g| g.input.iter().map(|v| v.name.as_str()).collect())
            .unwrap_or_default()
    }

    /// List graph output names.
    pub fn output_names(&self) -> Vec<&str> {
        self.proto
            .graph
            .as_ref()
            .map(|g| g.output.iter().map(|v| v.name.as_str()).collect())
            .unwrap_or_default()
    }
}

impl Model for OnnxFile {
    fn format(&self) -> ModelFormat {
        ModelFormat::Onnx
    }

    fn name(&self) -> Option<&str> {
        if self.proto.producer_name.is_empty() {
            self.proto.graph.as_ref().and_then(|g| {
                if g.name.is_empty() {
                    None
                } else {
                    Some(g.name.as_str())
                }
            })
        } else {
            Some(self.proto.producer_name.as_str())
        }
    }

    fn architecture(&self) -> Option<&str> {
        // ONNX doesn't have a standard "architecture" field; try metadata.
        for prop in &self.proto.metadata_props {
            if prop.key == "architecture" || prop.key == "model_type" {
                return Some(prop.value.as_str());
            }
        }
        None
    }

    fn block_count(&self) -> Option<usize> {
        // Common ONNX block naming: layern, block.n., transformer.h.n., etc.
        let mut max_idx: Option<i32> = None;
        for t in &self.tensors {
            if let Some(idx) = block_index_from_name_onnx(&t.name) {
                max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
            }
        }
        max_idx.map(|i| (i as usize) + 1)
    }

    fn tensors(&self) -> &[Tensor] {
        &self.tensors
    }

    fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.name_to_idx.get(name).map(|&i| &self.tensors[i])
    }

    fn metadata(&self, key: &str) -> Option<MetadataValue<'_>> {
        for prop in &self.proto.metadata_props {
            if prop.key == key {
                return Some(MetadataValue::String(prop.value.as_str()));
            }
        }
        None
    }

    fn read_tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let tp = self
            .tensor_proto(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;

        // Try raw_data first, then typed fields.
        if !tp.raw_data.is_empty() {
            return Ok(Cow::Owned(tp.raw_data.clone()));
        }

        // Fall back to typed repeated fields based on data_type.
        let bytes: Vec<u8> = match tp.data_type {
            onnx_dtypes::FLOAT => tp
                .float_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            onnx_dtypes::INT32 | onnx_dtypes::UINT32 => tp
                .int32_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            onnx_dtypes::INT64 | onnx_dtypes::UINT64 => tp
                .int64_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            onnx_dtypes::DOUBLE => tp
                .double_data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            onnx_dtypes::UINT8 | onnx_dtypes::INT8 | onnx_dtypes::BOOL => {
                // No raw_data and no typed byte field — try int32_data.
                tp.int32_data.iter().map(|&v| v as u8).collect()
            }
            _ => {
                return Err(Error::Onnx(format!(
                    "no readable data for tensor '{name}' (data_type {})",
                    tp.data_type
                )));
            }
        };
        Ok(Cow::Owned(bytes))
    }
}

// ---------------------------------------------------------------------------
// Block index extraction from ONNX tensor naming conventions
// ---------------------------------------------------------------------------

pub fn block_index_from_name_onnx(name: &str) -> Option<i32> {
    // Common ONNX patterns:
    //   layer.N.*          →  N
    //   blocks.N.*         →  N
    //   transformer.h.N.*  →  N
    //   model.layers.N.*   →  N
    //   encoder.layer.N.*  →  N  (BERT-style)
    let patterns: &[&str] = &[
        "layer.",
        "layers.",
        "block.",
        "blocks.",
        "transformer.h.",
        "encoder.layer.",
        "decoder.layer.",
        "model.layers.",
        "model.block.",
    ];
    for prefix in patterns {
        if let Some(rest) = name.strip_prefix(prefix) {
            let mut parts = rest.split('.');
            if let Ok(n) = parts.next()?.parse::<i32>() {
                return Some(n);
            }
        }
    }
    // Fallback: look for _N_ or _N. suffix patterns.
    None
}

// ---------------------------------------------------------------------------
// tensor_dtype -> ONNX data_type
// ---------------------------------------------------------------------------

fn tensor_dtype_to_onnx(dtype: TensorDtype) -> Option<i32> {
    Some(match dtype {
        TensorDtype::F32 => onnx_dtypes::FLOAT,
        TensorDtype::F16 => onnx_dtypes::FLOAT16,
        TensorDtype::Bf16 => onnx_dtypes::BFLOAT16,
        TensorDtype::F64 => onnx_dtypes::DOUBLE,
        TensorDtype::I8 => onnx_dtypes::INT8,
        TensorDtype::I16 => onnx_dtypes::INT16,
        TensorDtype::I32 => onnx_dtypes::INT32,
        TensorDtype::I64 => onnx_dtypes::INT64,
        TensorDtype::Unknown(0) => onnx_dtypes::BOOL,
        TensorDtype::Unknown(8) => onnx_dtypes::UINT8,
        TensorDtype::Unknown(16) => onnx_dtypes::UINT16,
        TensorDtype::Unknown(32) => onnx_dtypes::UINT32,
        TensorDtype::Unknown(64) => onnx_dtypes::UINT64,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// OnnxWriter
// ---------------------------------------------------------------------------

/// A writer that builds an ONNX protobuf (`ModelProto`) and serializes it.
///
/// Follows the same pattern as `SafetensorsWriter`: create, add tensors,
/// write.
///
/// The output is a valid ONNX model with all tensors stored in `raw_data`
/// (tag 10) for efficient binary storage. Graph `input` entries are
/// automatically created for each added tensor.
pub struct OnnxWriter {
    ir_version: i64,
    producer_name: String,
    producer_version: String,
    graph_name: String,
    metadata: Vec<(String, String)>,
    graph_inputs: Vec<(String, Vec<i64>)>,
    graph_outputs: Vec<(String, Vec<i64>)>,
    tensors: Vec<OnnxWriterTensor>,
}

/// A single tensor that will be serialized into the ONNX `initializer` list.
#[derive(Debug, Clone)]
pub struct OnnxWriterTensor {
    pub name: String,
    pub data_type: i32,
    pub shape: Vec<i64>,
    pub bytes: Vec<u8>,
}

impl OnnxWriter {
    pub fn new() -> Self {
        Self {
            ir_version: 9,
            producer_name: "tensorkit".into(),
            producer_version: "0.1".into(),
            graph_name: "graph".into(),
            metadata: Vec::new(),
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn ir_version(mut self, v: i64) -> Self {
        self.ir_version = v;
        self
    }

    pub fn producer(mut self, name: &str, version: &str) -> Self {
        self.producer_name = name.into();
        self.producer_version = version.into();
        self
    }

    pub fn graph_name(mut self, name: &str) -> Self {
        self.graph_name = name.into();
        self
    }

    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.push((key.to_string(), value.to_string()));
    }

    pub fn add_graph_input(&mut self, name: &str, shape: &[i64]) {
        self.graph_inputs.push((name.to_string(), shape.to_vec()));
    }

    pub fn add_graph_output(&mut self, name: &str, shape: &[i64]) {
        self.graph_outputs.push((name.to_string(), shape.to_vec()));
    }

    /// Add a tensor from a `WriterTensor`-like triple.
    pub fn add_raw(&mut self, name: &str, data_type: i32, shape: &[i64], bytes: &[u8]) {
        self.tensors.push(OnnxWriterTensor {
            name: name.to_string(),
            data_type,
            shape: shape.to_vec(),
            bytes: bytes.to_vec(),
        });
    }

    /// Add a tensor from a `TensorDtype` + raw bytes. Returns an error if the
    /// dtype cannot be mapped to an ONNX data type.
    pub fn add_tensor(&mut self, name: &str, dtype: TensorDtype, shape: &[u64], bytes: &[u8]) -> Result<()> {
        let data_type = tensor_dtype_to_onnx(dtype).ok_or_else(|| {
            Error::Onnx(format!(
                "unsupported TensorDtype {:?} for ONNX output",
                dtype
            ))
        })?;
        let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        self.add_raw(name, data_type, &shape_i64, bytes);
        Ok(())
    }

    /// How many tensors have been added.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Build the `ModelProto` and serialize to bytes, then write to `path`.
    pub fn write_to<W: std::io::Write>(&self, w: W) -> Result<()> {
        let proto = self.build_proto();
        let bytes = proto.encode_to_vec();
        let mut w = w;
        w.write_all(&bytes)?;
        Ok(())
    }

    /// Build the `ModelProto` and serialize to bytes.
    pub fn into_bytes(&self) -> Vec<u8> {
        let proto = self.build_proto();
        proto.encode_to_vec()
    }

    fn build_proto(&self) -> ModelProto {
        let tensor_protos: Vec<TensorProto> = self
            .tensors
            .iter()
            .map(|t| TensorProto {
                name: t.name.clone(),
                data_type: t.data_type,
                dims: t.shape.clone(),
                raw_data: t.bytes.clone(),
                ..Default::default()
            })
            .collect();

        // Auto-create graph inputs from initializer tensors if none were
        // manually specified.
        let inputs: Vec<ValueInfoProto> = if self.graph_inputs.is_empty() {
            tensor_protos
                .iter()
                .map(|t| ValueInfoProto {
                    name: t.name.clone(),
                    doc_string: String::new(),
                })
                .collect()
        } else {
            self.graph_inputs
                .iter()
                .map(|(name, _shape)| ValueInfoProto {
                    name: name.clone(),
                    doc_string: String::new(),
                })
                .collect()
        };

        let outputs: Vec<ValueInfoProto> = self
            .graph_outputs
            .iter()
            .map(|(name, _shape)| ValueInfoProto {
                name: name.clone(),
                doc_string: String::new(),
            })
            .collect();

        let metadata_props: Vec<StringStringEntryProto> = self
            .metadata
            .iter()
            .map(|(k, v)| StringStringEntryProto {
                key: k.clone(),
                value: v.clone(),
            })
            .collect();

        ModelProto {
            ir_version: self.ir_version,
            producer_name: self.producer_name.clone(),
            producer_version: self.producer_version.clone(),
            graph: Some(GraphProto {
                name: self.graph_name.clone(),
                input: inputs,
                output: outputs,
                initializer: tensor_protos,
            }),
            metadata_props,
            ..Default::default()
        }
    }
}

impl Default for OnnxWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/formats/onnx.rs"]
mod tests;