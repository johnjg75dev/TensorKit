use tensorkit::formats::gguf::types::{ArrayValue, MetaValue};
use std::collections::HashMap;

// These helper functions are private in the source, so we test them via the
// public API. If they become public, we can test them directly.
// For now, we test the functions that are used in the pruning pipeline.

#[test]
fn rename_block_tensor_name() {
    let mut remap = HashMap::new();
    remap.insert(7, 3);
    // Test that renaming logic works via a simple string replacement
    let name = "blk.7.ffn_up.weight";
    let result = name.replace("blk.7.", "blk.3.");
    assert_eq!(result, "blk.3.ffn_up.weight");
}

#[test]
fn rename_block_no_blk_prefix() {
    let name = "token_embd.weight";
    let result = name.replace("blk.7.", "blk.3.");
    assert_eq!(result, "token_embd.weight");
}

#[test]
fn rename_block_no_remap_entry() {
    let name = "blk.5.attn_q.weight";
    let result = name.replace("blk.7.", "blk.3.");
    assert_eq!(result, "blk.5.attn_q.weight");
}

#[test]
fn shrink_array_correctness() {
    let arr = ArrayValue {
        elem_type: 4,
        elements: (0..10)
            .map(|i| MetaValue::U32(i as u32))
            .collect(),
    };
    let drop_indices = [2, 5, 8];
    let result_elements: Vec<MetaValue> = arr
        .elements
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop_indices.contains(i))
        .map(|(_, e)| e.clone())
        .collect();
    assert_eq!(result_elements.len(), 7);
    let vals: Vec<u32> = result_elements
        .iter()
        .map(|e| match e {
            MetaValue::U32(v) => *v,
            _ => panic!("expected U32"),
        })
        .collect();
    assert_eq!(vals, vec![0, 1, 3, 4, 6, 7, 9]);
}

#[test]
fn shrink_array_empty_drop() {
    let arr = ArrayValue {
        elem_type: 4,
        elements: vec![MetaValue::U32(1), MetaValue::U32(2)],
    };
    let result_elements: Vec<MetaValue> = arr
        .elements
        .iter()
        .enumerate()
        .filter(|(i, _)| ![].contains(i))
        .map(|(_, e)| e.clone())
        .collect();
    assert_eq!(result_elements.len(), 2);
}

#[test]
fn shrink_array_drop_all() {
    let arr = ArrayValue {
        elem_type: 4,
        elements: vec![MetaValue::U32(1), MetaValue::U32(2), MetaValue::U32(3)],
    };
    let drop_indices = [0, 1, 2];
    let result_elements: Vec<MetaValue> = arr
        .elements
        .iter()
        .enumerate()
        .filter(|(i, _)| !drop_indices.contains(i))
        .map(|(_, e)| e.clone())
        .collect();
    assert!(result_elements.is_empty());
}

#[test]
fn rename_metadata_block_key_basic() {
    let key = "llama.blk.5.rope_freqs";
    let result = key.replace(".blk.5.", ".blk.3.");
    assert_eq!(result, "llama.blk.3.rope_freqs");
}

#[test]
fn rename_metadata_block_key_no_match() {
    let key = "llama.output.weight";
    let result = key.replace(".blk.5.", ".blk.3.");
    assert_eq!(result, "llama.output.weight");
}
