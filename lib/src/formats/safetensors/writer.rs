//! Safetensors writer used for pruned-output generation.

use crate::error::Result;
use crate::model::TensorDtype;
use std::collections::BTreeMap;
use std::io::Write;

pub struct SafetensorsWriter {
    pub tensors: Vec<WriterTensor>,
    pub metadata: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct WriterTensor {
    pub name: String,
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

impl Default for SafetensorsWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetensorsWriter {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            metadata: None,
        }
    }

    pub fn with_metadata(metadata: serde_json::Map<String, serde_json::Value>) -> Self {
        Self {
            tensors: Vec::new(),
            metadata: Some(metadata),
        }
    }

    pub fn set_metadata(&mut self, key: &str, value: serde_json::Value) {
        let m = self.metadata.get_or_insert_with(serde_json::Map::new);
        m.insert(key.to_string(), value);
    }

    pub fn add(&mut self, t: WriterTensor) {
        self.tensors.push(t);
    }

    /// Convenience: add a tensor from a name + raw bytes.
    pub fn add_raw(&mut self, name: String, dtype: TensorDtype, shape: Vec<u64>, bytes: &[u8]) {
        self.tensors.push(WriterTensor {
            name,
            dtype,
            shape,
            bytes: bytes.to_vec(),
        });
    }

    /// Build and write the file. We sort tensor names by data-offset order
    /// (which here equals the call order) and write the JSON header.
    pub fn write_to<W: Write>(&self, mut w: W) -> Result<()> {
        let mut offset: u64 = 0;
        let mut header_obj: BTreeMap<String, serde_json::Value> = BTreeMap::new();

        if let Some(ref m) = self.metadata
            && !m.is_empty() {
                header_obj.insert(
                    "__metadata__".to_string(),
                    serde_json::to_value(m).unwrap_or_default(),
                );
            }

        for t in &self.tensors {
            let start = offset;
            let end = start + t.bytes.len() as u64;
            header_obj.insert(
                t.name.clone(),
                serde_json::json!({
                    "dtype": dtype_to_str(t.dtype),
                    "shape": t.shape,
                    "data_offsets": [start, end],
                }),
            );
            offset = end;
        }
        let header_json = serde_json::to_vec(&header_obj)?;
        w.write_all(&(header_json.len() as u64).to_le_bytes())?;
        w.write_all(&header_json)?;
        for t in &self.tensors {
            w.write_all(&t.bytes)?;
        }
        Ok(())
    }
}

fn dtype_to_str(d: TensorDtype) -> &'static str {
    match d {
        TensorDtype::F32 => "F32",
        TensorDtype::F16 => "F16",
        TensorDtype::Bf16 => "BF16",
        TensorDtype::F64 => "F64",
        TensorDtype::I8 => "I8",
        TensorDtype::I16 => "I16",
        TensorDtype::I32 => "I32",
        TensorDtype::I64 => "I64",
        TensorDtype::Unknown(0) => "BOOL",
        TensorDtype::Unknown(8) => "U8",
        TensorDtype::Unknown(16) => "U16",
        TensorDtype::Unknown(32) => "U32",
        TensorDtype::Unknown(64) => "U64",
        _ => "U8",
    }
}
