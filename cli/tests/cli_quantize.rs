use std::path::PathBuf;
use std::process::Command;
use tensorkit::formats::gguf::types::{GgmlType, MetaValue, MetadataKv};
use tensorkit::formats::gguf::writer::GgufWriter;

struct Cleanup(PathBuf, PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let _ = std::fs::remove_file(&self.1);
    }
}

fn make_test_gguf(path: &PathBuf) {
    let mut w = GgufWriter::new(3, 32);
    w.add_kv(MetadataKv {
        key: "general.architecture".into(),
        value_type: 8,
        value: MetaValue::String("llama".into()),
    });
    let m = 64usize;
    let n = 32usize;
    let data: Vec<f32> = (0..m * n).map(|i| ((i as f32) * 0.1).sin() * 5.0).collect();
    let mut bytes = Vec::with_capacity(m * n * 4);
    for v in &data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.add_tensor(
        "blk.0.attn_q.weight".into(),
        2,
        [m as u64, n as u64, 1, 1],
        GgmlType::F32,
        &bytes,
    );
    let out = w.into_bytes().unwrap();
    std::fs::write(path, &out).unwrap();
}

#[test]
fn cli_quantize_q4_0_smoke() {
    let in_path =
        std::env::temp_dir().join(format!("tensorkit-cli-in-{}.gguf", std::process::id()));
    let out_path =
        std::env::temp_dir().join(format!("tensorkit-cli-out-{}.gguf", std::process::id()));
    let _cleanup = Cleanup(in_path.clone(), out_path.clone());
    make_test_gguf(&in_path);

    let exe = std::env::var("CARGO_BIN_EXE_tensorkit")
        .unwrap_or_else(|_| "target/debug/tensorkit".to_string());
    let out = Command::new(&exe)
        .arg("quant")
        .arg(&in_path)
        .arg("--target")
        .arg("q4_0")
        .arg("--out")
        .arg(&out_path)
        .arg("--yes")
        .output()
        .expect("run tensorkit");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Q4_0") || stdout.contains("q4_0"), "stdout: {stdout}");
    assert!(
        stdout.contains("1") && (stdout.contains("quantized") || stdout.contains("tensors")),
        "stdout: {stdout}"
    );

    // Verify the output file is a valid GGUF with the right dtype.
    let gg = tensorkit::formats::gguf::GgufFile::open(&out_path).expect("reopen");
    let ti = gg
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.attn_q.weight")
        .unwrap();
    assert_eq!(ti.ggml_type, GgmlType::Q4_0);
}
