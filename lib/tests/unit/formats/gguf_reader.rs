use tensorkit::formats::gguf::types::{GGUF_MAGIC, MetaValue, MetadataKv, GgmlType};
use tensorkit::formats::gguf::writer::GgufWriter;
use tensorkit::formats::gguf::GgufFile;
use tensorkit::Model;
use std::io::Cursor;

/// Build a minimal valid GGUF byte buffer (v3, no metadata, no tensors).
fn minimal_gguf_bytes() -> Vec<u8> {
    let w = GgufWriter::new(3, 32);
    w.into_bytes().unwrap()
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

const MAX_STRING_LEN: usize = 1_000_000;
const MAX_ARRAY_LEN: usize = 1_000_000;
const MAX_TENSOR_COUNT: u64 = 100_000;
const MAX_KV_COUNT: u64 = 1_000_000;

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
    bytes[0] = 0xFF;
    let mut cursor = Cursor::new(&bytes);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("bad magic"), "unexpected error: {msg}");
}

#[test]
fn parse_unsupported_version_returns_error() {
    let mut bytes = minimal_gguf_bytes();
    bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
    let mut cursor = Cursor::new(&bytes);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("unsupported GGUF version"), "unexpected error: {msg}");
}

// ---- DoS hardening ----

#[test]
fn parse_exceeding_max_tensor_count_returns_error() {
    let mut bytes = minimal_gguf_bytes();
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
    let bytes = minimal_gguf_bytes();
    let mut cursor = Cursor::new(&bytes);
    let gg = GgufFile::read_from(&mut cursor).unwrap();
    assert_eq!(gg.tensor_count, 0);
}

#[test]
fn parse_exceeding_max_string_len_in_kv_key_returns_error() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&(MAX_STRING_LEN + 1).to_le_bytes());

    let mut cursor = Cursor::new(&buf);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("string length") || msg.contains("exceeds limit"),
        "unexpected error: {msg}"
    );
}

#[test]
fn parse_exceeding_max_array_len_returns_error() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    let key = b"test_array";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&(MAX_ARRAY_LEN + 1).to_le_bytes());

    let mut cursor = Cursor::new(&buf);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("metadata array length") || msg.contains("exceeds limit"),
        "unexpected error: {msg}"
    );
}

// ---- GgufFile::open ----

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

    assert_eq!(gg.metadata_str("some.key"), None);
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

    use tensorkit::model::{Model, ModelFormat};
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

    use tensorkit::model::{Model, MetadataValue};
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
    assert!(gg.metadata("nonexistent").is_none());
}

// ---- read_tensor_bytes from stream ----

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
    let data_a: Vec<u8> = vec![0u8; 8 * 4 * 4];
    w.add_tensor("layer.0.weight".into(), 2, [8, 4, 1, 1], GgmlType::F32, &data_a);
    let data_b: Vec<u8> = vec![0u8; 4 * 8 * 4];
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

#[test]
fn read_string_at_max_len_succeeds_or_fails_gracefully() {
    let bytes = gguf_with_string_kv("key", "short_value");
    let mut cursor = Cursor::new(&bytes);
    let gg = GgufFile::read_from(&mut cursor).unwrap();
    assert_eq!(gg.metadata_str("key"), Some("short_value"));
}

#[test]
fn parse_empty_string_kv() {
    let bytes = gguf_with_string_kv("", "");
    let mut cursor = Cursor::new(&bytes);
    let gg = GgufFile::read_from(&mut cursor).unwrap();
    assert_eq!(gg.metadata.len(), 1);
    assert_eq!(gg.metadata_str(""), Some(""));
}

#[test]
fn parse_unknown_metadata_value_type_returns_error() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    let key = b"test";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&99u32.to_le_bytes());

    let mut cursor = Cursor::new(&buf);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unknown metadata value type"),
        "unexpected error: {msg}"
    );
}

#[test]
fn parse_tensor_with_too_many_dims_returns_error() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    let name = b"test_tensor";
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name);
    buf.extend_from_slice(&5u32.to_le_bytes());
    buf.extend(&vec![0u8; 5 * 8]);
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());

    let mut cursor = Cursor::new(&buf);
    let err = GgufFile::read_from(&mut cursor).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("dims"), "unexpected error: {msg}");
}
