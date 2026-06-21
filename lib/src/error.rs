use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid GGUF file: {0}")]
    Gguf(String),

    #[error("invalid safetensors file: {0}")]
    Safetensors(String),

    #[error("unsupported tensor type: {0}")]
    UnsupportedType(String),

    #[error("tensor '{0}' not found")]
    TensorNotFound(String),

    #[error("invalid selection '{0}': {1}")]
    InvalidSelection(String, String),

    #[error("invalid SVD config: {0}")]
    InvalidSvdConfig(String),

    #[error("SVD error: {0}")]
    Svd(String),

    #[error("quantize error: {0}")]
    Quantize(String),

    #[error("no prunable blocks found in model")]
    NoPrunableBlocks,

    #[error("calibration error: {0}")]
    Calibration(String),

    #[error("invalid UTF-8 in metadata: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid ONNX file: {0}")]
    Onnx(String),

    #[error("protobuf error: {0}")]
    Protobuf(#[from] prost::DecodeError),

    #[error("task arithmetic error: {0}")]
    TaskArith(String),

    #[error("embedding prune error: {0}")]
    EmbedPrune(String),

    #[error("inference error: {0}")]
    Infer(String),
}
