//! Safetensors reader.
//!
//! File layout:
//! ```text
//!   [u64 LE] header_len
//!   [header_len bytes] JSON header
//!   [rest] tensor data, contiguous
//! ```
//!
//! Header is a JSON object mapping tensor names to
//! `{ "dtype": "F32"|"F16"|"BF16"|"I8"|..., "shape": [..], "data_offsets": [start, end] }`.
//! A special key `__metadata__` carries free-form metadata.

use crate::error::{Error, Result};
use crate::model::{MetadataValue, Model, ModelFormat, Tensor, TensorDtype};
use memmap2::Mmap;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct SafetensorsFile {
    mmap: Mmap,
    header_len: u64,
    pub name_to_idx: HashMap<String, usize>,
    pub tensors: Vec<Tensor>,
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsHeader {
    #[serde(rename = "__metadata__", default)]
    metadata: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(flatten)]
    entries: HashMap<String, TensorEntry>,
}

#[derive(Debug, Deserialize)]
struct TensorEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl SafetensorsFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // SAFETY: we hold the file open and only read.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| Error::Safetensors(format!("mmap failed: {e}")))?;
        Self::from_mmap(mmap)
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        if mmap.len() < 8 {
            return Err(Error::Safetensors("file too small".into()));
        }
        let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        if header_end > mmap.len() {
            return Err(Error::Safetensors("header length exceeds file size".into()));
        }
        let header: SafetensorsHeader = serde_json::from_slice(&mmap[8..header_end])
            .map_err(|e| Error::Safetensors(format!("json header: {e}")))?;

        let metadata = header.metadata;

        let mut name_to_idx = HashMap::with_capacity(header.entries.len());
        let mut tensors = Vec::with_capacity(header.entries.len());
        for (name, e) in header.entries {
            let dtype = dtype_from_str(&e.dtype).ok_or_else(|| {
                Error::Safetensors(format!("unsupported dtype {dtype:?}", dtype = e.dtype))
            })?;
            let byte_size = e.data_offsets[1] - e.data_offsets[0];
            let idx = tensors.len();
            name_to_idx.insert(name.clone(), idx);
            tensors.push(Tensor {
                name,
                dtype,
                shape: e.shape,
                byte_size,
                data_offset: e.data_offsets[0],
            });
        }
        Ok(Self {
            mmap,
            header_len: header_len as u64,
            name_to_idx,
            tensors,
            metadata,
        })
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata
            .as_ref()?
            .get(key)?
            .as_str()
    }

    pub fn metadata_json(&self) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.metadata.as_ref()
    }
}

impl Model for SafetensorsFile {
    fn format(&self) -> ModelFormat {
        ModelFormat::Safetensors
    }
    fn name(&self) -> Option<&str> {
        self.metadata_str("general.name")
    }
    fn architecture(&self) -> Option<&str> {
        self.metadata_str("general.architecture")
    }
    fn block_count(&self) -> Option<usize> {
        // The convention is the same as GGUF: blk.N.*.
        let mut max_idx: Option<i32> = None;
        for t in &self.tensors {
            if let Some(idx) = block_index_from_name(&t.name) {
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
        let v = self.metadata.as_ref()?.get(key)?;
        match v {
            serde_json::Value::String(s) => Some(MetadataValue::String(s)),
            serde_json::Value::Number(n) => {
                if let Some(v) = n.as_u64() {
                    Some(MetadataValue::U64(v))
                } else if let Some(v) = n.as_i64() {
                    Some(MetadataValue::I64(v))
                } else { n.as_f64().map(MetadataValue::F64) }
            }
            serde_json::Value::Bool(b) => Some(MetadataValue::Bool(*b)),
            _ => None,
        }
    }
    fn read_tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let t = self
            .name_to_idx
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
        let data_section_offset = 8 + self.header_len;
        let start = data_section_offset as usize + t.data_offset as usize;
        let end = start + t.byte_size as usize;
        if end > self.mmap.len() {
            return Err(Error::Safetensors("tensor past end of file".into()));
        }
        Ok(Cow::Borrowed(&self.mmap[start..end]))
    }
}

fn dtype_from_str(s: &str) -> Option<TensorDtype> {
    Some(match s {
        "F32" => TensorDtype::F32,
        "F16" => TensorDtype::F16,
        "BF16" => TensorDtype::Bf16,
        "F64" => TensorDtype::F64,
        "I8" => TensorDtype::I8,
        "I16" => TensorDtype::I16,
        "I32" => TensorDtype::I32,
        "I64" => TensorDtype::I64,
        "U8" => TensorDtype::Unknown(8),
        "U16" => TensorDtype::Unknown(16),
        "U32" => TensorDtype::Unknown(32),
        "U64" => TensorDtype::Unknown(64),
        "BOOL" => TensorDtype::Unknown(0),
        _ => return None,
    })
}

fn block_index_from_name(name: &str) -> Option<i32> {
    let rest = name.strip_prefix("blk.")?;
    let mut parts = rest.split('.');
    parts.next()?.parse().ok()
}
