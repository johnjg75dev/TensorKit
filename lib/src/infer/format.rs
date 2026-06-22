//! Format detection and unified model opening.

use crate::error::{Error, Result};
use crate::model::ModelFormat;
use std::path::Path;

use super::WeightProvider;

/// Open any supported model format for inference.
///
/// Auto-detects format from file extension:
/// - `.gguf` → GGUF provider
/// - `.safetensors` → single-file Safetensors provider
/// - Directory → check for `model.safetensors.index.json` (sharded) or single file
/// - `.onnx` → ONNX provider
pub fn open_infer(path: &Path) -> Result<Box<dyn WeightProvider>> {
    if path.is_dir() {
        return open_directory(path);
    }

    let fmt = ModelFormat::from_path(path);
    match fmt {
        ModelFormat::Gguf => {
            let model = super::model::InferenceModel::open(path)?;
            Ok(Box::new(model))
        }
        ModelFormat::Safetensors => {
            let provider = super::provider_safetensors::SafetensorsProvider::open(path)?;
            Ok(Box::new(provider))
        }
        ModelFormat::Onnx => {
            let provider = super::provider_onnx::OnnxProvider::open(path)?;
            Ok(Box::new(provider))
        }
        ModelFormat::Unknown => {
            // Try to detect by reading first bytes
            let mut file = std::fs::File::open(path).map_err(Error::Io)?;
            let mut magic = [0u8; 4];
            use std::io::Read;
            file.read_exact(&mut magic).map_err(Error::Io)?;

            // GGUF magic: 0x46554747 ("GGUF" in LE)
            if magic == [0x47, 0x47, 0x55, 0x46] {
                let model = super::model::InferenceModel::open(path)?;
                return Ok(Box::new(model));
            }

            Err(Error::Infer(format!(
                "unsupported model format: {}",
                path.display()
            )))
        }
    }
}

/// Open a directory, looking for sharded or single-file models.
fn open_directory(dir: &Path) -> Result<Box<dyn WeightProvider>> {
    // Check for sharded safetensors index
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let provider =
            super::provider_safetensors::SafetensorsProvider::open_sharded(dir)?;
        return Ok(Box::new(provider));
    }

    // Check for single safetensors file
    let st_path = dir.join("model.safetensors");
    if st_path.exists() {
        let provider = super::provider_safetensors::SafetensorsProvider::open(&st_path)?;
        return Ok(Box::new(provider));
    }

    // Check for GGUF files in directory
    if let Some(gguf) = find_gguf_in_dir(dir)? {
        let model = super::model::InferenceModel::open(&gguf)?;
        return Ok(Box::new(model));
    }

    Err(Error::Infer(format!(
        "no supported model found in directory: {}",
        dir.display()
    )))
}

/// Find the primary GGUF file in a directory (shard 00001 or the only one).
fn find_gguf_in_dir(dir: &Path) -> Result<Option<std::path::PathBuf>> {
    use std::fs;

    let mut gguf_files: Vec<std::path::PathBuf> = Vec::new();
    for entry in fs::read_dir(dir).map_err(Error::Io)? {
        let entry = entry.map_err(Error::Io)?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
            gguf_files.push(path);
        }
    }

    if gguf_files.is_empty() {
        return Ok(None);
    }

    // Prefer shard 00001 if multiple files
    gguf_files.sort();
    Ok(Some(gguf_files.into_iter().next().unwrap()))
}
