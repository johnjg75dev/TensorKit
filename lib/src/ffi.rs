//! C-compatible FFI (Foreign Function Interface) for DLL export.
//!
//! Provides an opaque-handle API suitable for calling from C, C++, Python,
//! C#/.NET, or any language that supports C FFI.
//!
//! # Conventions
//!
//! - All handles are opaque pointers (`*mut c_void`).
//! - Strings are null-terminated C strings (`*const c_char`).
//! - Return value `0` = success, negative = error.
//! - Call `tk_last_error()` after a negative return to get the error message.
//! - All handles must be freed with the corresponding `tk_*_close` / `tk_*_free`.
//!
//! # Example (C)
//! ```c
//! TkModel m = tk_model_open("model.gguf");
//! if (!m) { char buf[256]; tk_last_error(buf, sizeof(buf)); ... }
//! int n = tk_model_tensor_count(m);
//! tk_model_close(m);
//! ```

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::analysis::analyzer::Analyzer;
use crate::analysis::score::BlockRole;
use crate::analysis::stats::Analysis;
use crate::formats::gguf::GgufFile;
use crate::model::{Model, ModelFormat, TensorDtype};
use crate::prune::selection::parse_selection;
use crate::svd::config::SvdConfig;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;

// ---------------------------------------------------------------------------
// Global last-error string (thread-safe)
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

/// Retrieve the last error message into `buf`. Returns the number of bytes
/// written (excluding null terminator), or 0 if no error.
#[unsafe(no_mangle)]
pub extern "C" fn tk_last_error(buf: *mut c_char, buf_len: usize) -> c_int {
    if buf.is_null() || buf_len == 0 {
        return 0;
    }
    LAST_ERROR.with(|e| {
        let borrow = e.borrow();
        if let Some(ref msg) = *borrow {
            let bytes = msg.to_bytes();
            let n = bytes.len().min(buf_len - 1);
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
                *buf.add(n) = 0;
            }
            n as c_int
        } else {
            unsafe { *buf = 0; }
            0
        }
    })
}

// ---------------------------------------------------------------------------
// Opaque handle helpers
// ---------------------------------------------------------------------------

/// Wrap a `T` into an opaque `*mut c_void` handle.
fn into_handle<T>(val: T) -> *mut std::ffi::c_void {
    Box::into_raw(Box::new(val)) as *mut std::ffi::c_void
}

/// Convert an opaque handle back to `&T`. Returns `None` if null.
unsafe fn from_handle<'a, T>(h: *mut std::ffi::c_void) -> Option<&'a T> {
    if h.is_null() {
        None
    } else {
        Some(unsafe { &*(h as *const T) })
    }
}

/// Convert an opaque handle back to `&T`. Returns `None` if null.
#[allow(dead_code)]
unsafe fn from_handle_mut<'a, T>(h: *mut std::ffi::c_void) -> Option<&'a mut T> {
    if h.is_null() {
        None
    } else {
        Some(unsafe { &mut *(h as *mut T) })
    }
}

/// Safely convert a C string to a Rust `&str`. Returns `Err` on null/bad utf8.
unsafe fn cstr_to_str<'a>(p: *const c_char) -> std::result::Result<&'a str, String> {
    if p.is_null() {
        return Err("null string pointer".into());
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|e| format!("invalid UTF-8: {e}"))
}

// ---------------------------------------------------------------------------
// Model format enum (C-compatible integer)
// ---------------------------------------------------------------------------

/// Model format identifiers for C FFI.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TkModelFormat {
    Unknown = 0,
    Gguf = 1,
    Safetensors = 2,
    Onnx = 3,
}

impl From<ModelFormat> for TkModelFormat {
    fn from(f: ModelFormat) -> Self {
        match f {
            ModelFormat::Gguf => TkModelFormat::Gguf,
            ModelFormat::Safetensors => TkModelFormat::Safetensors,
            ModelFormat::Onnx => TkModelFormat::Onnx,
            ModelFormat::Unknown => TkModelFormat::Unknown,
        }
    }
}

/// Convert a `TensorDtype` to a u32 code for C FFI.
fn tensor_dtype_to_u32(dt: TensorDtype) -> u32 {
    match dt {
        TensorDtype::F32 => 0,
        TensorDtype::F16 => 1,
        TensorDtype::Bf16 => 2,
        TensorDtype::F64 => 3,
        TensorDtype::I8 => 4,
        TensorDtype::I16 => 5,
        TensorDtype::I32 => 6,
        TensorDtype::I64 => 7,
        TensorDtype::Q4_0 => 8,
        TensorDtype::Q4_1 => 9,
        TensorDtype::Q5_0 => 10,
        TensorDtype::Q5_1 => 11,
        TensorDtype::Q8_0 => 12,
        TensorDtype::Q8_1 => 13,
        TensorDtype::Q2K => 14,
        TensorDtype::Q3K => 15,
        TensorDtype::Q4K => 16,
        TensorDtype::Q5K => 17,
        TensorDtype::Q6K => 18,
        TensorDtype::Q8K => 19,
        TensorDtype::Iq2Xxs => 20,
        TensorDtype::Iq2Xs => 21,
        TensorDtype::Iq3Xxs => 22,
        TensorDtype::Iq1S => 23,
        TensorDtype::Iq4Nl => 24,
        TensorDtype::Iq3S => 25,
        TensorDtype::Iq2S => 26,
        TensorDtype::Iq4Xs => 27,
        TensorDtype::Iq1M => 28,
        TensorDtype::Tq1_0 => 29,
        TensorDtype::Tq2_0 => 30,
        TensorDtype::Unknown(v) => 0x1000 | (v & 0xFFF),
    }
}

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

struct TkModelInner {
    gguf: Option<GgufFile>,
    path: String,
}

/// Open a model file. Returns a handle, or null on error.
#[unsafe(no_mangle)]
pub extern "C" fn tk_model_open(path: *const c_char) -> *mut std::ffi::c_void {
    let path_str = match unsafe { cstr_to_str(path) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e.to_string());
            return ptr::null_mut();
        }
    };

    match GgufFile::open(Path::new(path_str)) {
        Ok(gguf) => {
            let inner = TkModelInner {
                gguf: Some(gguf),
                path: path_str.to_string(),
            };
            into_handle(inner)
        }
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Close a model handle and free resources.
#[unsafe(no_mangle)]
pub extern "C" fn tk_model_close(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle as *mut TkModelInner)); }
    }
}

/// Get the model format as a `TkModelFormat` enum value.
#[unsafe(no_mangle)]
pub extern "C" fn tk_model_format(handle: *mut std::ffi::c_void) -> TkModelFormat {
    let _inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => return TkModelFormat::Unknown,
    };
    TkModelFormat::Gguf // currently only GGUF is fully supported
}

/// Get the model name. Returns null if unavailable.
/// Caller must NOT free the returned pointer.
#[unsafe(no_mangle)]
pub extern "C" fn tk_model_name(handle: *mut std::ffi::c_void) -> *const c_char {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => return ptr::null(),
    };
    if let Some(gg) = &inner.gguf
        && let Some(name) = gg.name()
    {
        return name.as_ptr() as *const c_char;
    }
    ptr::null()
}

/// Get the model architecture. Returns null if unavailable.
#[unsafe(no_mangle)]
pub extern "C" fn tk_model_architecture(handle: *mut std::ffi::c_void) -> *const c_char {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => return ptr::null(),
    };
    if let Some(gg) = &inner.gguf
        && let Some(arch) = gg.architecture()
    {
        return arch.as_ptr() as *const c_char;
    }
    ptr::null()
}

/// Get the block count. Returns -1 if unavailable.
#[unsafe(no_mangle)]
pub extern "C" fn tk_model_block_count(handle: *mut std::ffi::c_void) -> c_int {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => return -1,
    };
    if let Some(gg) = &inner.gguf
        && let Some(n) = gg.block_count()
    {
        return n as c_int;
    }
    -1
}

/// Get the total number of tensors in the model.
#[unsafe(no_mangle)]
pub extern "C" fn tk_model_tensor_count(handle: *mut std::ffi::c_void) -> c_int {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => return -1,
    };
    if let Some(gg) = &inner.gguf {
        return gg.tensors().len() as c_int;
    }
    -1
}

/// Get info about the N-th tensor. Returns 0 on success, -1 on error.
/// Fills `name_buf` with the tensor name, `dtype` with the ggml type id,
/// and `byte_size` with the size in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn tk_tensor_nth(
    handle: *mut std::ffi::c_void,
    n: c_int,
    name_buf: *mut c_char,
    name_len: usize,
    dtype: *mut u32,
    byte_size: *mut u64,
) -> c_int {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => return -1,
    };
    let gg = match &inner.gguf {
        Some(g) => g,
        None => return -1,
    };
    let idx = n as usize;
    if idx >= gg.tensors().len() || name_buf.is_null() || name_len == 0 {
        return -1;
    }
    let t = &gg.tensors()[idx];
    let name_bytes = t.name.as_bytes();
    let n_copy = name_bytes.len().min(name_len - 1);
    unsafe {
        ptr::copy_nonoverlapping(name_bytes.as_ptr(), name_buf as *mut u8, n_copy);
        *name_buf.add(n_copy) = 0;
        if !dtype.is_null() {
            *dtype = tensor_dtype_to_u32(t.dtype);
        }
        if !byte_size.is_null() {
            *byte_size = t.byte_size;
        }
    }
    0
}

/// Get a string metadata value. Returns 0 on success, -1 on error/not-found.
#[unsafe(no_mangle)]
pub extern "C" fn tk_metadata_str(
    handle: *mut std::ffi::c_void,
    key: *const c_char,
    buf: *mut c_char,
    buf_len: usize,
) -> c_int {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => return -1,
    };
    let key_str = match unsafe { cstr_to_str(key) } {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let gg = match &inner.gguf {
        Some(g) => g,
        None => return -1,
    };
    if let Some(val) = gg.metadata_str(key_str) {
        let bytes = val.as_bytes();
        let n = bytes.len().min(buf_len - 1);
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
        }
        0
    } else {
        -1
    }
}

/// Get a u32 metadata value. Returns 0 on success, -1 on error/not-found.
#[unsafe(no_mangle)]
pub extern "C" fn tk_metadata_u32(
    handle: *mut std::ffi::c_void,
    key: *const c_char,
    out: *mut u32,
) -> c_int {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => return -1,
    };
    let key_str = match unsafe { cstr_to_str(key) } {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let gg = match &inner.gguf {
        Some(g) => g,
        None => return -1,
    };
    if let Some(val) = gg.metadata_u32(key_str) {
        if !out.is_null() {
            unsafe { *out = val; }
        }
        0
    } else {
        -1
    }
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

struct TkAnalysisInner {
    analysis: Analysis,
}

/// Run analysis on the model. Returns a handle, or null on error.
/// `sample_per_tensor` controls sampling depth (0 = default 200,000).
#[unsafe(no_mangle)]
pub extern "C" fn tk_analyze(
    handle: *mut std::ffi::c_void,
    sample_per_tensor: c_int,
) -> *mut std::ffi::c_void {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => {
            set_last_error("null model handle");
            return ptr::null_mut();
        }
    };
    let gg = match &inner.gguf {
        Some(g) => g,
        None => {
            set_last_error("no GGUF data in model");
            return ptr::null_mut();
        }
    };

    let analyzer = if sample_per_tensor > 0 {
        Analyzer::with_sample_per_tensor(sample_per_tensor as usize)
    } else {
        Analyzer::new()
    };

    match analyzer.analyze(gg) {
        Ok(analysis) => into_handle(TkAnalysisInner { analysis }),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Free an analysis handle.
#[unsafe(no_mangle)]
pub extern "C" fn tk_analysis_free(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle as *mut TkAnalysisInner)); }
    }
}

/// Get the number of blocks in the analysis.
#[unsafe(no_mangle)]
pub extern "C" fn tk_analysis_block_count(handle: *mut std::ffi::c_void) -> c_int {
    let inner = match unsafe { from_handle::<TkAnalysisInner>(handle) } {
        Some(h) => h,
        None => return -1,
    };
    inner.analysis.blocks.len() as c_int
}

/// Get a block's label and removable score. Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn tk_analysis_block(
    handle: *mut std::ffi::c_void,
    idx: c_int,
    label_buf: *mut c_char,
    label_len: usize,
    removable: *mut f64,
    tensor_count: *mut c_int,
    total_bytes: *mut u64,
    role: *mut c_int,
) -> c_int {
    let inner = match unsafe { from_handle::<TkAnalysisInner>(handle) } {
        Some(h) => h,
        None => return -1,
    };
    let i = idx as usize;
    if i >= inner.analysis.blocks.len() {
        return -1;
    }
    let block = &inner.analysis.blocks[i];
    if !label_buf.is_null() && label_len > 0 {
        let bytes = block.label.as_bytes();
        let n = bytes.len().min(label_len - 1);
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), label_buf as *mut u8, n);
            *label_buf.add(n) = 0;
        }
    }
    if !removable.is_null() {
        unsafe { *removable = block.removable; }
    }
    if !tensor_count.is_null() {
        unsafe { *tensor_count = block.tensor_count as c_int; }
    }
    if !total_bytes.is_null() {
        unsafe { *total_bytes = block.total_bytes; }
    }
    if !role.is_null() {
        let r = match block.role {
            BlockRole::Embedding => 0,
            BlockRole::OutputHead => 1,
            BlockRole::FinalNorm => 2,
            BlockRole::Block => 3,
            BlockRole::Other => 4,
        };
        unsafe { *role = r; }
    }
    0
}

/// Get the recommendation (list of prunable block indices).
/// `out` must point to an array of at least `max_out` ints.
/// Returns the number of recommended blocks, or -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn tk_analysis_recommendation(
    handle: *mut std::ffi::c_void,
    out: *mut c_int,
    max_out: c_int,
) -> c_int {
    let inner = match unsafe { from_handle::<TkAnalysisInner>(handle) } {
        Some(h) => h,
        None => return -1,
    };
    let rec = &inner.analysis.recommendation;
    let n = rec.len().min(max_out as usize);
    if !out.is_null() {
        for i in 0..n {
            unsafe { *out.add(i) = rec[i]; }
        }
    }
    n as c_int
}

/// Get the estimated bytes after pruning.
#[unsafe(no_mangle)]
pub extern "C" fn tk_analysis_estimated_bytes_after_prune(handle: *mut std::ffi::c_void) -> u64 {
    let inner = match unsafe { from_handle::<TkAnalysisInner>(handle) } {
        Some(h) => h,
        None => return 0,
    };
    inner.analysis.estimated_bytes_after_prune
}

/// Serialize the analysis to JSON. Caller must free with `tk_string_free`.
#[unsafe(no_mangle)]
pub extern "C" fn tk_analysis_json(handle: *mut std::ffi::c_void) -> *mut c_char {
    let inner = match unsafe { from_handle::<TkAnalysisInner>(handle) } {
        Some(h) => h,
        None => return ptr::null_mut(),
    };
    match serde_json::to_string_pretty(&inner.analysis) {
        Ok(json) => CString::new(json).unwrap_or_default().into_raw(),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Free a string returned by `tk_analysis_json` or similar.
#[unsafe(no_mangle)]
pub extern "C" fn tk_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}

// ---------------------------------------------------------------------------
// Quantize
// ---------------------------------------------------------------------------

/// Quantize a GGUF model and write to `dst_path`.
/// `target_type` is a GGML type string like "Q4_0", "Q8_0", "Q4_K", etc.
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn tk_quantize_model(
    handle: *mut std::ffi::c_void,
    dst_path: *const c_char,
    target_type: *const c_char,
) -> c_int {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => {
            set_last_error("null model handle");
            return -1;
        }
    };
    let dst = match unsafe { cstr_to_str(dst_path) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e);
            return -1;
        }
    };
    let ty_str = match unsafe { cstr_to_str(target_type) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e);
            return -1;
        }
    };

    use crate::formats::gguf::types::GgmlType;
    let ty = match ty_str {
        "Q4_0" => GgmlType::Q4_0,
        "Q4_1" => GgmlType::Q4_1,
        "Q5_0" => GgmlType::Q5_0,
        "Q5_1" => GgmlType::Q5_1,
        "Q8_0" => GgmlType::Q8_0,
        "Q8_1" => GgmlType::Q8_1,
        "Q2_K" => GgmlType::Q2K,
        "Q3_K" => GgmlType::Q3K,
        "Q4_K" => GgmlType::Q4K,
        "Q5_K" => GgmlType::Q5K,
        "Q6_K" => GgmlType::Q6K,
        "Q8_K" => GgmlType::Q8K,
        _ => {
            set_last_error(&format!("unknown target type: {ty_str}"));
            return -1;
        }
    };

    match crate::quantize::apply::quantize_gguf(
        Path::new(&inner.path),
        ty,
        Path::new(dst),
        None,
    ) {
        Ok(_) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------

struct TkSvdPlanInner {
    plan: crate::svd::plan::SvdPlan,
}

/// Build an SVD compression plan. Returns a handle, or null on error.
/// `config_json` is currently unused (uses defaults); pass null.
#[unsafe(no_mangle)]
pub extern "C" fn tk_svd_plan(
    handle: *mut std::ffi::c_void,
    _config_json: *const c_char,
) -> *mut std::ffi::c_void {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => {
            set_last_error("null model handle");
            return ptr::null_mut();
        }
    };
    let gg = match &inner.gguf {
        Some(g) => g,
        None => {
            set_last_error("no GGUF data");
            return ptr::null_mut();
        }
    };
    let cfg = SvdConfig::default();
    match crate::svd::plan::build_plan(gg, &cfg) {
        Ok(plan) => into_handle(TkSvdPlanInner { plan }),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Apply SVD compression and write to `dst_path`.
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn tk_svd_apply(
    handle: *mut std::ffi::c_void,
    plan_handle: *mut std::ffi::c_void,
    dst_path: *const c_char,
) -> c_int {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => {
            set_last_error("null model handle");
            return -1;
        }
    };
    let plan_inner = match unsafe { from_handle::<TkSvdPlanInner>(plan_handle) } {
        Some(h) => h,
        None => {
            set_last_error("null SVD plan handle");
            return -1;
        }
    };
    let dst = match unsafe { cstr_to_str(dst_path) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e.to_string());
            return -1;
        }
    };
    let gg = match &inner.gguf {
        Some(g) => g,
        None => {
            set_last_error("no GGUF data");
            return -1;
        }
    };
    match crate::svd::apply::apply_to_gguf(gg, &plan_inner.plan, Path::new(dst)) {
        Ok(_) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Free an SVD plan handle.
#[unsafe(no_mangle)]
pub extern "C" fn tk_svd_plan_free(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle as *mut TkSvdPlanInner)); }
    }
}

// ---------------------------------------------------------------------------
// Prune
// ---------------------------------------------------------------------------

struct TkPrunePlanInner {
    plan: crate::prune::plan::PrunePlan,
}

/// Build a pruning plan.
/// `selection` is a selection string like "auto:4", "drop:3,5,7", "keep:0,1,2".
/// `analysis_handle` can be null for non-auto selections.
/// Returns a handle, or null on error.
#[unsafe(no_mangle)]
pub extern "C" fn tk_prune_plan(
    handle: *mut std::ffi::c_void,
    selection: *const c_char,
    analysis_handle: *mut std::ffi::c_void,
) -> *mut std::ffi::c_void {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => {
            set_last_error("null model handle");
            return ptr::null_mut();
        }
    };
    let sel_str = match unsafe { cstr_to_str(selection) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e);
            return ptr::null_mut();
        }
    };
    let sel = match parse_selection(sel_str) {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e.to_string());
            return ptr::null_mut();
        }
    };
    let gg = match &inner.gguf {
        Some(g) => g,
        None => {
            set_last_error("no GGUF data");
            return ptr::null_mut();
        }
    };

    let scores = if !analysis_handle.is_null() {
        unsafe { from_handle::<TkAnalysisInner>(analysis_handle) }
            .map(|a| a.analysis.blocks.as_slice())
    } else {
        None
    };

    match crate::prune::plan::build_plan(gg, &sel, scores) {
        Ok(plan) => into_handle(TkPrunePlanInner { plan }),
        Err(e) => {
            set_last_error(&e.to_string());
            ptr::null_mut()
        }
    }
}

/// Apply pruning and write to `dst_path`.
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn tk_prune_apply(
    handle: *mut std::ffi::c_void,
    plan_handle: *mut std::ffi::c_void,
    dst_path: *const c_char,
) -> c_int {
    let inner = match unsafe { from_handle::<TkModelInner>(handle) } {
        Some(h) => h,
        None => {
            set_last_error("null model handle");
            return -1;
        }
    };
    let plan_inner = match unsafe { from_handle::<TkPrunePlanInner>(plan_handle) } {
        Some(h) => h,
        None => {
            set_last_error("null prune plan handle");
            return -1;
        }
    };
    let dst = match unsafe { cstr_to_str(dst_path) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e.to_string());
            return -1;
        }
    };
    let gg = match &inner.gguf {
        Some(g) => g,
        None => {
            set_last_error("no GGUF data");
            return -1;
        }
    };
    match crate::prune::apply::apply_to_gguf(gg, &plan_inner.plan, Path::new(dst)) {
        Ok(_) => 0,
        Err(e) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Free a prune plan handle.
#[unsafe(no_mangle)]
pub extern "C" fn tk_prune_plan_free(handle: *mut std::ffi::c_void) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle as *mut TkPrunePlanInner)); }
    }
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (direct access, no model needed)
// ---------------------------------------------------------------------------

/// Quantize a raw f32 buffer to the given GGML type.
/// Returns a newly allocated buffer; caller must free with `tk_bytes_free`.
/// `out_len` receives the length of the output buffer.
#[unsafe(no_mangle)]
pub extern "C" fn tk_quantize_buffer(
    src: *const f32,
    src_len: usize,
    type_str: *const c_char,
    out_len: *mut usize,
) -> *mut u8 {
    if src.is_null() || type_str.is_null() {
        set_last_error("null pointer");
        return ptr::null_mut();
    }
    let ty_str = match unsafe { cstr_to_str(type_str) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e.to_string());
            return ptr::null_mut();
        }
    };
    use crate::formats::gguf::types::GgmlType;
    let ty = match ty_str {
        "F32" => GgmlType::F32,
        "Q4_0" => GgmlType::Q4_0,
        "Q4_1" => GgmlType::Q4_1,
        "Q5_0" => GgmlType::Q5_0,
        "Q5_1" => GgmlType::Q5_1,
        "Q8_0" => GgmlType::Q8_0,
        "Q8_1" => GgmlType::Q8_1,
        _ => {
            set_last_error(&format!("unsupported type: {ty_str}"));
            return ptr::null_mut();
        }
    };
    let slice = unsafe { std::slice::from_raw_parts(src, src_len) };
    let data = crate::quantize::quantize(slice, ty);
    let len = data.len();
    if !out_len.is_null() {
        unsafe { *out_len = len; }
    }
    let mut buf = data.into_boxed_slice();
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Free a buffer returned by `tk_quantize_buffer`.
#[unsafe(no_mangle)]
pub extern "C" fn tk_bytes_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        unsafe {
            drop(Vec::from_raw_parts(ptr, len, len));
        }
    }
}

/// Dequantize a raw buffer to f32. Returns newly allocated f32 array.
/// Caller must free with `tk_f32_free`.
#[unsafe(no_mangle)]
pub extern "C" fn tk_dequantize_buffer(
    src: *const u8,
    src_len: usize,
    type_str: *const c_char,
    out_len: *mut usize,
) -> *mut f32 {
    if src.is_null() || type_str.is_null() {
        set_last_error("null pointer");
        return ptr::null_mut();
    }
    let ty_str = match unsafe { cstr_to_str(type_str) } {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&e.to_string());
            return ptr::null_mut();
        }
    };
    use crate::formats::gguf::types::GgmlType;
    let ty = match ty_str {
        "F32" => GgmlType::F32,
        "F16" => GgmlType::F16,
        "BF16" => GgmlType::Bf16,
        "Q4_0" => GgmlType::Q4_0,
        "Q4_1" => GgmlType::Q4_1,
        "Q5_0" => GgmlType::Q5_0,
        "Q5_1" => GgmlType::Q5_1,
        "Q8_0" => GgmlType::Q8_0,
        "Q8_1" => GgmlType::Q8_1,
        _ => {
            set_last_error(&format!("unsupported type for dequant: {ty_str}"));
            return ptr::null_mut();
        }
    };
    let slice = unsafe { std::slice::from_raw_parts(src, src_len) };
    let data = match crate::formats::gguf::dequant::dequantize(ty, slice, None) {
        Some(d) => d,
        None => {
            set_last_error("dequantization failed");
            return ptr::null_mut();
        }
    };
    let len = data.len();
    if !out_len.is_null() {
        unsafe { *out_len = len; }
    }
    let mut buf = data.into_boxed_slice();
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Free a f32 buffer returned by `tk_dequantize_buffer`.
#[unsafe(no_mangle)]
pub extern "C" fn tk_f32_free(ptr: *mut f32, len: usize) {
    if !ptr.is_null() {
        unsafe {
            drop(Vec::from_raw_parts(ptr, len, len));
        }
    }
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/// Get the library version string. Caller must NOT free.
#[unsafe(no_mangle)]
pub extern "C" fn tk_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}
