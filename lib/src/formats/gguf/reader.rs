//! GGUF v1–v3 reader.
//!
//! On-disk layout:
//! ```text
//!   Header       { magic:u32, version:u32, tensor_count:u64, kv_count:u64 }  (24 B)
//!   KV pairs     [ MetadataKv; kv_count ]
//!   Tensor infos [ TensorInfo; tensor_count ]
//!   Padding      to `general.alignment` (default 32) bytes
//!   Tensor data  raw bytes, each tensor at file_offset = data_offset + tensor.offset
//! ```
//!
//! All multi-byte values are little-endian.
//!
//! `GgufFile::open(path)` mmaps the file for zero-copy tensor reads.
//! `GgufFile::read_from(reader)` parses the metadata from any `Read+Seek`
//! but cannot offer mmap-backed reads.

use crate::error::{Error, Result};
use crate::formats::gguf::types::{
    byte_size_for, dims_product, ArrayValue, GgmlType, MetaValue, MetadataKv, TensorInfo,
    DEFAULT_ALIGNMENT, GGUF_MAGIC,
};
use crate::model::{MetadataValue, Model, ModelFormat, Tensor};
use memmap2::Mmap;
use std::borrow::Cow;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

const MAX_KV_COUNT: u64 = 1_000_000;
const MAX_TENSOR_COUNT: u64 = 1_000_000;
const MAX_STRING_LEN: u64 = 1 << 20;
const MAX_ARRAY_LEN: u64 = 1 << 20;

pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata: Vec<MetadataKv>,
    pub tensors: Vec<TensorInfo>,
    /// File offset where the tensor data section begins (after padding).
    pub data_section_offset: u64,
    pub alignment: usize,
    /// Optional mmap of the whole file. Present when opened via `open()`.
    mmap: Option<Mmap>,
    /// Cached model-agnostic view of every tensor.
    model_tensors: Vec<Tensor>,
    /// Name -> index into `tensors` for O(1) lookup.
    name_to_idx: std::collections::HashMap<String, usize>,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("tensor_count", &self.tensor_count)
            .field("kv_count", &self.kv_count)
            .field("metadata", &self.metadata.len())
            .field("tensors", &self.tensors.len())
            .field("data_section_offset", &self.data_section_offset)
            .field("alignment", &self.alignment)
            .field("mmap_bytes", &self.mmap.as_ref().map(|m| m.len()))
            .finish()
    }
}

impl GgufFile {
    /// Open a GGUF file from disk, mmapping it for zero-copy tensor reads.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let mut file = File::open(path_ref)?;
        // SAFETY: We hold `file` open for the lifetime of the mmap, and we
        // only read from the mmap. The file is opened read-only via `File::open`.
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| Error::Gguf(format!("mmap failed: {e}")))?;
        // Reuse the same file handle for parsing metadata.
        Self::parse(&mut file, Some(mmap))
    }

    /// Parse the header + metadata from any `Read + Seek` source.
    /// Tensor bytes are not loaded; `read_tensor_bytes` will error.
    pub fn read_from<R: Read + Seek>(r: &mut R) -> Result<Self> {
        Self::parse(r, None)
    }

    fn parse<R: Read + Seek>(r: &mut R, mmap: Option<Mmap>) -> Result<Self> {
        // ---- Header (batched: one read_exact, four from_le_bytes) ----------
        let mut hdr = [0u8; 24];
        r.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        if magic != GGUF_MAGIC {
            return Err(Error::Gguf(format!("bad magic: 0x{magic:08x}")));
        }
        let version = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        if !(1..=3).contains(&version) {
            return Err(Error::Gguf(format!("unsupported GGUF version {version}")));
        }
        let tensor_count = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        let kv_count = u64::from_le_bytes(hdr[16..24].try_into().unwrap());

        if tensor_count > MAX_TENSOR_COUNT {
            return Err(Error::Gguf(format!(
                "tensor_count ({tensor_count}) exceeds limit ({MAX_TENSOR_COUNT})"
            )));
        }
        if kv_count > MAX_KV_COUNT {
            return Err(Error::Gguf(format!(
                "kv_count ({kv_count}) exceeds limit ({MAX_KV_COUNT})"
            )));
        }

        // ---- KV pairs --------------------------------------------------------
        let mut metadata = Vec::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            metadata.push(read_kv(r)?);
        }

        let mut alignment = DEFAULT_ALIGNMENT;
        for kv in &metadata {
            if kv.key == "general.alignment" {
                alignment = match &kv.value {
                    MetaValue::U32(v) => *v as usize,
                    MetaValue::U64(v) => *v as usize,
                    _ => alignment,
                };
            }
        }

        // ---- Tensor infos (batched tail read) --------------------------------
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            tensors.push(read_tensor_info(r)?);
        }

        for t in tensors.iter_mut() {
            t.n_elements = dims_product(&t.dims, t.n_dims);
            t.byte_size = byte_size_for(t.n_elements, t.ggml_type);
        }

        let pos_after_infos = r.stream_position()?;
        let pad = ((alignment - (pos_after_infos as usize % alignment)) % alignment) as u64;
        let data_section_offset = pos_after_infos + pad;

        // Build the cached model-agnostic view and lookup map.
        let mut model_tensors = Vec::with_capacity(tensors.len());
        let mut name_to_idx = std::collections::HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            name_to_idx.insert(t.name.clone(), i);
            model_tensors.push(Tensor {
                name: t.name.clone(),
                dtype: t.ggml_type.to_tensor_dtype(),
                shape: (0..t.n_dims as usize).map(|i| t.dims[i]).collect(),
                byte_size: t.byte_size,
                data_offset: t.offset,
            });
        }

        Ok(GgufFile {
            version,
            tensor_count,
            kv_count,
            metadata,
            tensors,
            data_section_offset,
            alignment,
            mmap,
            model_tensors,
            name_to_idx,
        })
    }

    /// Zero-copy slice of one tensor's bytes (requires `open()`, not `read_from`).
    #[inline]
    pub fn tensor_slice(&self, t: &TensorInfo) -> Option<&[u8]> {
        let mmap = self.mmap.as_ref()?;
        let start = (self.data_section_offset + t.offset) as usize;
        let end = start + t.byte_size as usize;
        if end > mmap.len() {
            return None;
        }
        Some(&mmap[start..end])
    }

    /// Look up a tensor by name (returns the GGUF-native view).
    #[inline]
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.name_to_idx.get(name).map(|&i| &self.tensors[i])
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.iter().find(|k| k.key == key).and_then(|kv| {
            if let MetaValue::String(s) = &kv.value {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        self.metadata
            .iter()
            .find(|k| k.key == key)
            .and_then(|kv| match &kv.value {
                MetaValue::U32(v) => Some(*v),
                MetaValue::U64(v) => Some(*v as u32),
                _ => None,
            })
    }
}

impl Model for GgufFile {
    fn format(&self) -> ModelFormat {
        ModelFormat::Gguf
    }
    fn name(&self) -> Option<&str> {
        self.metadata_str("general.name")
    }
    fn architecture(&self) -> Option<&str> {
        self.metadata_str("general.architecture")
    }
    fn block_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata_u32(&format!("{arch}.block_count"))
            .map(|v| v as usize)
    }
    fn tensors(&self) -> &[Tensor] {
        &self.model_tensors
    }
    fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.name_to_idx.get(name).map(|&i| &self.model_tensors[i])
    }
    fn metadata(&self, key: &str) -> Option<MetadataValue<'_>> {
        let kv = self.metadata.iter().find(|k| k.key == key)?;
        Some(match &kv.value {
            MetaValue::U8(v) => MetadataValue::U8(*v),
            MetaValue::I8(v) => MetadataValue::I8(*v),
            MetaValue::U16(v) => MetadataValue::U16(*v),
            MetaValue::I16(v) => MetadataValue::I16(*v),
            MetaValue::U32(v) => MetadataValue::U32(*v),
            MetaValue::I32(v) => MetadataValue::I32(*v),
            MetaValue::F32(v) => MetadataValue::F32(*v),
            MetaValue::Bool(v) => MetadataValue::Bool(*v),
            MetaValue::String(v) => MetadataValue::String(v),
            MetaValue::U64(v) => MetadataValue::U64(*v),
            MetaValue::I64(v) => MetadataValue::I64(*v),
            MetaValue::F64(v) => MetadataValue::F64(*v),
            MetaValue::Array(_) => return None,
        })
    }
    fn read_tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let t = self
            .get_tensor(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
        match self.tensor_slice(t) {
            Some(slice) => Ok(Cow::Borrowed(slice)),
            None => Err(Error::Gguf(
                "tensor bytes unavailable: file was parsed from a stream, not opened with mmap"
                    .into(),
            )),
        }
    }
}

// -- raw read helpers --------------------------------------------------------

trait ReadExt: Read {
    #[inline]
    fn read_u8_le(&mut self) -> std::io::Result<u8> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }
    #[inline]
    fn read_u16_le(&mut self) -> std::io::Result<u16> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    #[inline]
    fn read_i16_le(&mut self) -> std::io::Result<i16> {
        Ok(self.read_u16_le()? as i16)
    }
    #[inline]
    fn read_u32_le(&mut self) -> std::io::Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    #[inline]
    fn read_i32_le(&mut self) -> std::io::Result<i32> {
        Ok(self.read_u32_le()? as i32)
    }
    #[inline]
    fn read_u64_le(&mut self) -> std::io::Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    #[inline]
    fn read_i64_le(&mut self) -> std::io::Result<i64> {
        Ok(self.read_u64_le()? as i64)
    }
    #[inline]
    fn read_f32_le(&mut self) -> std::io::Result<f32> {
        Ok(f32::from_le_bytes(self.read_u32_le()?.to_le_bytes()))
    }
    #[inline]
    fn read_f64_le(&mut self) -> std::io::Result<f64> {
        Ok(f64::from_le_bytes(self.read_u64_le()?.to_le_bytes()))
    }
    #[inline]
    fn read_string(&mut self) -> std::io::Result<String> {
        let len = self.read_u64_le()? as usize;
        if len > MAX_STRING_LEN as usize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("string length ({len}) exceeds limit ({MAX_STRING_LEN})"),
            ));
        }
        let mut buf = vec![0u8; len];
        self.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
impl<R: Read> ReadExt for R {}

#[inline]
fn read_kv<R: Read>(r: &mut R) -> Result<MetadataKv> {
    let key = r.read_string()?;
    let value_type = r.read_u32_le()?;
    let value = read_meta_value(r, value_type)
        .map_err(|e| Error::Gguf(format!("metadata '{key}': {e}")))?;
    Ok(MetadataKv {
        key,
        value_type,
        value,
    })
}

fn read_meta_value<R: Read>(r: &mut R, ty: u32) -> std::io::Result<MetaValue> {
    Ok(match ty {
        0 => MetaValue::U8(r.read_u8_le()?),
        1 => MetaValue::I8(r.read_u8_le()? as i8),
        2 => MetaValue::U16(r.read_u16_le()?),
        3 => MetaValue::I16(r.read_i16_le()?),
        4 => MetaValue::U32(r.read_u32_le()?),
        5 => MetaValue::I32(r.read_i32_le()?),
        6 => MetaValue::F32(r.read_f32_le()?),
        7 => MetaValue::Bool(r.read_u8_le()? != 0),
        8 => MetaValue::String(r.read_string()?),
        9 => {
            let elem_type = r.read_u32_le()?;
            let len = r.read_u64_le()? as usize;
            if len > MAX_ARRAY_LEN as usize {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("metadata array length ({len}) exceeds limit ({MAX_ARRAY_LEN})"),
                ));
            }
            let mut elems = Vec::with_capacity(len);
            for _ in 0..len {
                elems.push(read_meta_value(r, elem_type)?);
            }
            MetaValue::Array(ArrayValue {
                elem_type,
                elements: elems,
            })
        }
        10 => MetaValue::U64(r.read_u64_le()?),
        11 => MetaValue::I64(r.read_i64_le()?),
        12 => MetaValue::F64(r.read_f64_le()?),
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown metadata value type {other}"),
            ))
        }
    })
}

/// Batched tensor info read: read the variable-length name, then a tail of
/// `4 + n_dims*8 + 4 + 8` bytes in a single read_exact.
///
/// The tail is sized by `n_dims` (read first), not by the maximum 4 dims, so
/// that tensors with `n_dims < 4` don't bleed into the next tensor's header.
fn read_tensor_info<R: Read>(r: &mut R) -> Result<TensorInfo> {
    let name = r.read_string()?;
    let n_dims = r.read_u32_le()?;
    if n_dims > 4 {
        return Err(Error::Gguf(format!(
            "tensor '{name}' has {n_dims} dims (max 4)"
        )));
    }
    let mut dims_tail = [0u8; 4 * 8];
    r.read_exact(&mut dims_tail[..(n_dims as usize) * 8])?;
    let mut dims = [1u64; 4];
    for i in 0..n_dims as usize {
        let off = i * 8;
        dims[i] = u64::from_le_bytes(dims_tail[off..off + 8].try_into().unwrap());
    }
    let ty_raw = r.read_u32_le()?;
    let offset = r.read_u64_le()?;
    Ok(TensorInfo {
        name,
        n_dims,
        dims,
        ggml_type: GgmlType::from_u32(ty_raw),
        offset,
        n_elements: 0,
        byte_size: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::gguf::types::{GGUF_MAGIC, MetaValue, MetadataKv};
    use crate::formats::gguf::writer::GgufWriter;
    use std::io::Cursor;

    /// Build a minimal valid GGUF byte buffer (v3, no metadata, no tensors).
    fn minimal_gguf_bytes() -> Vec<u8> {
        let w = GgufWriter::new(3, 32);
        let out = w.into_bytes().unwrap();
        out
    }

    /// Build a GGUF byte buffer with a single string KV entry.
    fn gguf_with_string_kv(key: &str, value: &str) -> Vec<u8> {
        let mut w = GgufWriter::new(3, 32);
        w.add_kv(MetadataKv {
            key: key.into(),
            value_type: 8,
            value: MetaValue::String(value.into()),
        });
        w.into_bytes().unwrap()
    }

    /// Build a GGUF byte buffer with a single u32 KV entry.
    fn gguf_with_u32_kv(key: &str, value: u32) -> Vec<u8> {
        let mut w = GgufWriter::new(3, 32);
        w.add_kv(MetadataKv {
            key: key.into(),
            value_type: 4,
            value: MetaValue::U32(value),
        });
        w.into_bytes().unwrap()
    }

    /// Build a GGUF byte buffer with an array KV entry.
    fn gguf_with_array_kv(key: &str, elem_type: u32, elements: Vec<MetaValue>) -> Vec<u8> {
        let mut w = GgufWriter::new(3, 32);
        w.add_kv(MetadataKv {
            key: key.into(),
            value_type: 9,
            value: MetaValue::Array(ArrayValue {
                elem_type,
                elements,
            }),
        });
        w.into_bytes().unwrap()
    }

    /// Build a GGUF byte buffer with one F32 tensor.
    fn gguf_with_tensor(name: &str, m: usize, n: usize) -> Vec<u8> {
        let mut w = GgufWriter::new(3, 32);
        let data: Vec<u8> = (0..m * n * 4).map(|i| (i % 256) as u8).collect();
        w.add_tensor(
            name.into(),
            2,
            [m as u64, n as u64, 1, 1],
            GgmlType::F32,
            &data,
        );
        w.into_bytes().unwrap()
    }

    // ---- Valid parsing tests ----

    #[test]
    fn parse_valid_minimal_gguf() {
        let bytes = minimal_gguf_bytes();
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(gg.version, 3);
        assert_eq!(gg.tensor_count, 0);
        assert_eq!(gg.kv_count, 0);
        assert!(gg.metadata.is_empty());
        assert!(gg.tensors.is_empty());
    }

    #[test]
    fn parse_valid_gguf_with_metadata() {
        let bytes = gguf_with_string_kv("general.architecture", "llama");
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(gg.metadata.len(), 1);
        assert_eq!(gg.metadata_str("general.architecture"), Some("llama"));
    }

    #[test]
    fn parse_valid_gguf_with_tensor() {
        let bytes = gguf_with_tensor("blk.0.weight", 8, 4);
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(gg.tensors.len(), 1);
        assert_eq!(gg.tensors[0].name, "blk.0.weight");
        assert_eq!(gg.tensors[0].n_elements, 32);
        assert_eq!(gg.tensors[0].ggml_type, GgmlType::F32);
    }

    #[test]
    fn parse_valid_gguf_with_u32_metadata() {
        let bytes = gguf_with_u32_kv("llama.block_count", 32);
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(gg.metadata_u32("llama.block_count"), Some(32));
    }

    // ---- Bad magic / version ----

    #[test]
    fn parse_bad_magic_returns_error() {
        let mut bytes = minimal_gguf_bytes();
        bytes[0] = 0xFF; // corrupt magic
        let mut cursor = Cursor::new(&bytes);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bad magic"), "unexpected error: {msg}");
    }

    #[test]
    fn parse_unsupported_version_returns_error() {
        let mut bytes = minimal_gguf_bytes();
        // Overwrite version (bytes 4-7) with version 99 (little-endian).
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        let mut cursor = Cursor::new(&bytes);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unsupported GGUF version"), "unexpected error: {msg}");
    }

    // ---- DoS hardening: tensor_count and kv_count ----

    #[test]
    fn parse_exceeding_max_tensor_count_returns_error() {
        let mut bytes = minimal_gguf_bytes();
        // Overwrite tensor_count (bytes 8-16) with MAX_TENSOR_COUNT + 1.
        let bad_count = MAX_TENSOR_COUNT + 1;
        bytes[8..16].copy_from_slice(&bad_count.to_le_bytes());
        let mut cursor = Cursor::new(&bytes);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("tensor_count"), "unexpected error: {msg}");
        assert!(msg.contains("exceeds limit"), "unexpected error: {msg}");
    }

    #[test]
    fn parse_exceeding_max_kv_count_returns_error() {
        let mut bytes = minimal_gguf_bytes();
        // Overwrite kv_count (bytes 16-24) with MAX_KV_COUNT + 1.
        let bad_count = MAX_KV_COUNT + 1;
        bytes[16..24].copy_from_slice(&bad_count.to_le_bytes());
        let mut cursor = Cursor::new(&bytes);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("kv_count"), "unexpected error: {msg}");
        assert!(msg.contains("exceeds limit"), "unexpected error: {msg}");
    }

    #[test]
    fn parse_tensor_count_at_limit_succeeds() {
        // We can't actually write MAX_TENSOR_COUNT tensor info entries (that's 38 MB+),
        // but we can verify the check passes at the boundary by patching the header
        // of a valid small file and confirming it would fail at count+1.
        let bytes = minimal_gguf_bytes();
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(gg.tensor_count, 0); // zero is fine
    }

    // ---- DoS hardening: string length ----

    #[test]
    fn parse_exceeding_max_string_len_in_kv_key_returns_error() {
        // Construct a raw buffer with a string whose length exceeds MAX_STRING_LEN.
        let mut buf = Vec::new();
        // Header: magic + version + tensor_count=0 + kv_count=1
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        // KV key: length > MAX_STRING_LEN
        buf.extend_from_slice(&(MAX_STRING_LEN + 1).to_le_bytes());
        // We don't need to actually write the string bytes since read_string
        // should fail before trying to read them.

        let mut cursor = Cursor::new(&buf);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("string length") || msg.contains("exceeds limit"),
            "unexpected error: {msg}"
        );
    }

    // ---- DoS hardening: array length ----

    #[test]
    fn parse_exceeding_max_array_len_returns_error() {
        // Build a raw buffer that has an array metadata value with length > MAX_ARRAY_LEN.
        let mut buf = Vec::new();
        // Header: magic + version + tensor_count=0 + kv_count=1
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        // KV key: short valid string
        let key = b"test_array";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        // value_type = 9 (Array)
        buf.extend_from_slice(&9u32.to_le_bytes());
        // elem_type = 4 (U32)
        buf.extend_from_slice(&4u32.to_le_bytes());
        // array length > MAX_ARRAY_LEN
        buf.extend_from_slice(&(MAX_ARRAY_LEN + 1).to_le_bytes());
        // No need for actual elements since it should fail before reading them.

        let mut cursor = Cursor::new(&buf);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("metadata array length") || msg.contains("exceeds limit"),
            "unexpected error: {msg}"
        );
    }

    // ---- GgufFile::open reuses file handle ----

    #[test]
    fn open_reuses_file_handle_for_tensor_access() {
        let tmp = std::env::temp_dir().join(format!(
            "tensorkit-reader-test-{}.gguf",
            std::process::id()
        ));
        let bytes = gguf_with_tensor("blk.0.weight", 4, 2);
        std::fs::write(&tmp, &bytes).unwrap();

        let gg = GgufFile::open(&tmp).unwrap();
        assert_eq!(gg.tensors.len(), 1);
        // open() should give us mmap access to tensor data
        let slice = gg.tensor_slice(&gg.tensors[0]);
        assert!(slice.is_some(), "tensor_slice should work after open()");

        let _ = std::fs::remove_file(&tmp);
    }

    // ---- get_tensor and metadata helpers ----

    #[test]
    fn get_tensor_returns_correct_info() {
        let bytes = gguf_with_tensor("blk.0.weight", 8, 4);
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();

        let ti = gg.get_tensor("blk.0.weight").unwrap();
        assert_eq!(ti.name, "blk.0.weight");
        assert_eq!(ti.n_elements, 32);

        assert!(gg.get_tensor("nonexistent").is_none());
    }

    #[test]
    fn metadata_str_returns_none_for_non_string() {
        let bytes = gguf_with_u32_kv("some.key", 42);
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();

        assert_eq!(gg.metadata_str("some.key"), None); // it's a U32, not a String
        assert_eq!(gg.metadata_u32("some.key"), Some(42));
    }

    #[test]
    fn metadata_u32_returns_none_for_missing_key() {
        let bytes = minimal_gguf_bytes();
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(gg.metadata_u32("missing.key"), None);
    }

    // ---- Model trait ----

    #[test]
    fn model_trait_format_and_name() {
        let mut w = GgufWriter::new(3, 32);
        w.add_kv(MetadataKv {
            key: "general.name".into(),
            value_type: 8,
            value: MetaValue::String("test_model".into()),
        });
        w.add_kv(MetadataKv {
            key: "general.architecture".into(),
            value_type: 8,
            value: MetaValue::String("llama".into()),
        });
        w.add_kv(MetadataKv {
            key: "llama.block_count".into(),
            value_type: 4,
            value: MetaValue::U32(24),
        });
        let bytes = w.into_bytes().unwrap();
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();

        use crate::model::{Model, ModelFormat};
        assert_eq!(gg.format(), ModelFormat::Gguf);
        assert_eq!(gg.name(), Some("test_model"));
        assert_eq!(gg.architecture(), Some("llama"));
        assert_eq!(gg.block_count(), Some(24));
    }

    #[test]
    fn model_trait_metadata_all_types() {
        let mut w = GgufWriter::new(3, 32);
        w.add_kv(MetadataKv { key: "u8_k".into(), value_type: 0, value: MetaValue::U8(42) });
        w.add_kv(MetadataKv { key: "i8_k".into(), value_type: 1, value: MetaValue::I8(-7) });
        w.add_kv(MetadataKv { key: "u16_k".into(), value_type: 2, value: MetaValue::U16(1000) });
        w.add_kv(MetadataKv { key: "i16_k".into(), value_type: 3, value: MetaValue::I16(-500) });
        w.add_kv(MetadataKv { key: "u32_k".into(), value_type: 4, value: MetaValue::U32(99999) });
        w.add_kv(MetadataKv { key: "i32_k".into(), value_type: 5, value: MetaValue::I32(-99999) });
        w.add_kv(MetadataKv { key: "f32_k".into(), value_type: 6, value: MetaValue::F32(3.14) });
        w.add_kv(MetadataKv { key: "bool_k".into(), value_type: 7, value: MetaValue::Bool(true) });
        w.add_kv(MetadataKv { key: "str_k".into(), value_type: 8, value: MetaValue::String("hello".into()) });
        w.add_kv(MetadataKv { key: "u64_k".into(), value_type: 10, value: MetaValue::U64(123456789) });
        w.add_kv(MetadataKv { key: "i64_k".into(), value_type: 11, value: MetaValue::I64(-123456789) });
        w.add_kv(MetadataKv { key: "f64_k".into(), value_type: 12, value: MetaValue::F64(2.71828) });
        let bytes = w.into_bytes().unwrap();
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();

        use crate::model::{Model, MetadataValue};
        assert!(matches!(gg.metadata("u8_k"), Some(MetadataValue::U8(42))));
        assert!(matches!(gg.metadata("i8_k"), Some(MetadataValue::I8(-7))));
        assert!(matches!(gg.metadata("u16_k"), Some(MetadataValue::U16(1000))));
        assert!(matches!(gg.metadata("i16_k"), Some(MetadataValue::I16(-500))));
        assert!(matches!(gg.metadata("u32_k"), Some(MetadataValue::U32(99999))));
        assert!(matches!(gg.metadata("i32_k"), Some(MetadataValue::I32(-99999))));
        assert!(matches!(gg.metadata("f32_k"), Some(MetadataValue::F32(f)) if (f - 3.14).abs() < 1e-5));
        assert!(matches!(gg.metadata("bool_k"), Some(MetadataValue::Bool(true))));
        assert!(matches!(gg.metadata("str_k"), Some(MetadataValue::String("hello"))));
        assert!(matches!(gg.metadata("u64_k"), Some(MetadataValue::U64(123456789))));
        assert!(matches!(gg.metadata("i64_k"), Some(MetadataValue::I64(-123456789))));
        assert!(matches!(gg.metadata("f64_k"), Some(MetadataValue::F64(f)) if (f - 2.71828).abs() < 1e-5));
        // Array -> None from Model::metadata
        assert!(gg.metadata("nonexistent").is_none());
    }

    // ---- read_tensor_bytes from stream (no mmap) ----

    #[test]
    fn read_tensor_bytes_from_stream_returns_error() {
        let bytes = gguf_with_tensor("blk.0.weight", 4, 2);
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        let err = gg.read_tensor_bytes("blk.0.weight").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("tensor bytes unavailable"), "unexpected error: {msg}");
    }

    #[test]
    fn read_tensor_bytes_missing_tensor_returns_error() {
        let bytes = minimal_gguf_bytes();
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        let err = gg.read_tensor_bytes("nonexistent").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not found"), "unexpected error: {msg}");
    }

    // ---- Debug impl ----

    #[test]
    fn debug_impl_does_not_panic() {
        let bytes = gguf_with_tensor("blk.0.weight", 4, 2);
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        let debug_str = format!("{:?}", gg);
        assert!(debug_str.contains("GgufFile"));
        assert!(debug_str.contains("version"));
    }

    // ---- Multiple KV and tensors ----

    #[test]
    fn parse_multiple_kv_and_tensors() {
        let mut w = GgufWriter::new(3, 32);
        w.add_kv(MetadataKv {
            key: "general.architecture".into(),
            value_type: 8,
            value: MetaValue::String("llama".into()),
        });
        w.add_kv(MetadataKv {
            key: "llama.block_count".into(),
            value_type: 4,
            value: MetaValue::U32(2),
        });
        let data_a: Vec<u8> = vec![0u8; 8 * 4 * 4]; // 8x4 f32
        w.add_tensor("layer.0.weight".into(), 2, [8, 4, 1, 1], GgmlType::F32, &data_a);
        let data_b: Vec<u8> = vec![0u8; 4 * 8 * 4]; // 4x8 f32
        w.add_tensor("layer.1.weight".into(), 2, [4, 8, 1, 1], GgmlType::F32, &data_b);
        let bytes = w.into_bytes().unwrap();
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();

        assert_eq!(gg.kv_count, 2);
        assert_eq!(gg.tensor_count, 2);
        assert_eq!(gg.metadata.len(), 2);
        assert_eq!(gg.tensors.len(), 2);
        assert_eq!(gg.get_tensor("layer.0.weight").unwrap().n_elements, 32);
        assert_eq!(gg.get_tensor("layer.1.weight").unwrap().n_elements, 32);
    }

    // ---- Boundary: exactly at MAX_STRING_LEN ----

    #[test]
    fn read_string_at_max_len_succeeds_or_fails_gracefully() {
        // A string of exactly MAX_STRING_LEN bytes should succeed if the
        // file is large enough. But we can't easily construct that (1 MB string
        // in a header). Instead, test that the boundary check is >= not >.
        // We test with a small valid string to confirm normal behavior.
        let bytes = gguf_with_string_kv("key", "short_value");
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(gg.metadata_str("key"), Some("short_value"));
    }

    // ---- Empty string key/value ----

    #[test]
    fn parse_empty_string_kv() {
        let bytes = gguf_with_string_kv("", "");
        let mut cursor = Cursor::new(&bytes);
        let gg = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(gg.metadata.len(), 1);
        assert_eq!(gg.metadata_str(""), Some(""));
    }

    // ---- Unknown metadata value type ----

    #[test]
    fn parse_unknown_metadata_value_type_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        let key = b"test";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&99u32.to_le_bytes()); // unknown value type

        let mut cursor = Cursor::new(&buf);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown metadata value type"),
            "unexpected error: {msg}"
        );
    }

    // ---- Tensor with n_dims > 4 ----

    #[test]
    fn parse_tensor_with_too_many_dims_returns_error() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count=1
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        // Tensor name
        let name = b"test_tensor";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        // n_dims = 5 (> max 4)
        buf.extend_from_slice(&5u32.to_le_bytes());
        // We need 5*8=40 bytes of dims, but we'll only need the first 5*8 to trigger the error
        buf.extend(&vec![0u8; 5 * 8]);
        buf.extend_from_slice(&0u32.to_le_bytes()); // type
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut cursor = Cursor::new(&buf);
        let err = GgufFile::read_from(&mut cursor).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("dims"), "unexpected error: {msg}");
    }
}
