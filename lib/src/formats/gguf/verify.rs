//! Re-open a pruned file and verify its structural integrity.

use crate::error::Result;
use crate::formats::gguf::reader::GgufFile;
use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, serde::Serialize)]
pub struct VerifyReport {
    pub ok: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub kept_tensors: usize,
    pub total_bytes: u64,
    pub file_size: u64,
}

pub fn verify(path: &Path, expected_names: &[String]) -> Result<VerifyReport> {
    let file_size = std::fs::metadata(path)?.len();
    let gguf = GgufFile::open(path)?;
    let mut report = VerifyReport {
        ok: true,
        errors: Vec::new(),
        warnings: Vec::new(),
        kept_tensors: 0,
        total_bytes: 0,
        file_size,
    };

    if !(1..=3).contains(&gguf.version) {
        report.errors.push(format!("bad version {}", gguf.version));
    }

    let present: HashSet<&str> = gguf.tensors.iter().map(|t| t.name.as_str()).collect();
    for n in expected_names {
        if !present.contains(n.as_str()) {
            report.errors.push(format!("missing expected tensor: {n}"));
        }
    }
    report.kept_tensors = gguf.tensors.len();

    for t in &gguf.tensors {
        use crate::formats::gguf::types::byte_size_for;
        let expected = byte_size_for(t.n_elements, t.ggml_type);
        if expected != t.byte_size {
            report.errors.push(format!(
                "tensor '{}': byte_size {} != computed {expected}",
                t.name, t.byte_size
            ));
        }
        report.total_bytes += t.byte_size;
    }

    let mut ranges: Vec<(u64, u64, &str)> = gguf
        .tensors
        .iter()
        .map(|t| {
            let start = gguf.data_section_offset + t.offset;
            let end = start + t.byte_size;
            (start, end, t.name.as_str())
        })
        .collect();
    ranges.sort_by_key(|(s, _, _)| *s);
    for w in ranges.windows(2) {
        let (s1, e1, _) = w[0];
        let (s2, _, n2) = w[1];
        if s2 < e1 {
            report.errors.push(format!(
                "tensor overlap: {s1}..{e1} collides with '{n2}' at {s2}"
            ));
        }
    }
    if let Some((_, e, last)) = ranges.last()
        && *e > file_size {
            report.errors.push(format!(
                "last tensor '{last}' ends at {e} past file size {file_size}"
            ));
        }

    if gguf.tensors.is_empty() {
        report.warnings.push("no tensors in pruned file".into());
    } else if report.total_bytes == 0 {
        report
            .warnings
            .push("all tensors report zero byte size".into());
    }

    report.ok = report.errors.is_empty();
    Ok(report)
}

/// FNV-1a checksum over the tensor data section.
pub fn data_checksum(path: &Path) -> Result<u64> {
    let gguf = GgufFile::open(path)?;
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(gguf.data_section_offset))?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        for &b in &buf[..n] {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    Ok(hash)
}
