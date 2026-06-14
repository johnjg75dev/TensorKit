use super::*;

// ---------------------------------------------------------------------------
// Mock model for testing
// ---------------------------------------------------------------------------

struct MockEmbedModel {
    vocab_size: u32,
    hidden_dim: u32,
}

impl MockEmbedModel {
    fn new(vocab_size: u32, hidden_dim: u32) -> Self {
        Self {
            vocab_size,
            hidden_dim,
        }
    }
}

impl crate::model::Model for MockEmbedModel {
    fn format(&self) -> crate::model::ModelFormat {
        crate::model::ModelFormat::Unknown
    }
    fn name(&self) -> Option<&str> {
        None
    }
    fn architecture(&self) -> Option<&str> {
        None
    }
    fn block_count(&self) -> Option<usize> {
        None
    }
    fn tensors(&self) -> &[crate::model::Tensor] {
        &[]
    }
    fn tensor(&self, name: &str) -> Option<&crate::model::Tensor> {
        match name {
            "token_embd.weight" => Some(Box::leak(Box::new(crate::model::Tensor {
                name: name.to_string(),
                dtype: crate::model::TensorDtype::F32,
                shape: vec![self.vocab_size as u64, self.hidden_dim as u64],
                byte_size: (self.vocab_size * self.hidden_dim * 4) as u64,
                data_offset: 0,
            }))),
            "output.weight" => Some(Box::leak(Box::new(crate::model::Tensor {
                name: name.to_string(),
                dtype: crate::model::TensorDtype::F32,
                shape: vec![self.hidden_dim as u64, self.vocab_size as u64],
                byte_size: (self.hidden_dim * self.vocab_size * 4) as u64,
                data_offset: 0,
            }))),
            _ => None,
        }
    }
    fn metadata(&self, _key: &str) -> Option<crate::model::MetadataValue<'_>> {
        None
    }
    fn read_tensor_bytes(&self, name: &str) -> crate::Result<Cow<'static, [u8]>> {
        let nelem = (self.vocab_size * self.hidden_dim) as usize;
        let data: Vec<u8> = match name {
            "token_embd.weight" => (0..nelem).flat_map(|i| (i as f32).to_le_bytes()).collect(),
            "output.weight" => (0..nelem)
                .flat_map(|i| (i as f32 + 10000.0).to_le_bytes())
                .collect(),
            _ => return Err(crate::Error::TensorNotFound(name.to_string())),
        };
        let leaked: &'static [u8] = Box::leak(data.into_boxed_slice());
        Ok(Cow::Borrowed(leaked))
    }
}

// ---------------------------------------------------------------------------
// plan_embed_prune
// ---------------------------------------------------------------------------

#[test]
fn plan_by_id_basic() {
    let model = MockEmbedModel::new(4, 8);
    let selection = TokenSelection::ById(vec![0, 2, 3]);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    assert_eq!(plan.keep_rows, vec![0, 2, 3]);
    assert_eq!(plan.original_vocab_size, 4);
    assert_eq!(plan.new_vocab_size, 3);
    assert_eq!(plan.remap[&0], 0);
    assert_eq!(plan.remap[&2], 1);
    assert_eq!(plan.remap[&3], 2);
}

#[test]
fn plan_top_n() {
    let model = MockEmbedModel::new(10, 8);
    let selection = TokenSelection::TopN(3);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    assert_eq!(plan.keep_rows, vec![0, 1, 2]);
    assert_eq!(plan.new_vocab_size, 3);
}

#[test]
fn plan_top_n_exceeds_vocab_clamps() {
    let model = MockEmbedModel::new(4, 8);
    let selection = TokenSelection::TopN(100);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    assert_eq!(plan.keep_rows, vec![0, 1, 2, 3]);
    assert_eq!(plan.new_vocab_size, 4);
}

#[test]
fn plan_by_pattern() {
    let model = MockEmbedModel::new(4, 8);
    let vocab = vec!["hello".into(), "world".into(), "foo".into(), "bar".into()];
    let selection = TokenSelection::ByPattern(regex::Regex::new("^(hello|bar)$").unwrap());
    let plan = plan_embed_prune(&model, &selection, Some(&vocab)).unwrap();
    assert_eq!(plan.keep_rows, vec![0, 3]);
    assert_eq!(plan.new_vocab_size, 2);
}

#[test]
fn plan_by_pattern_no_match_errors() {
    let model = MockEmbedModel::new(4, 8);
    let vocab = vec!["a".into(), "b".into(), "c".into(), "d".into()];
    let selection = TokenSelection::ByPattern(regex::Regex::new("^xyz$").unwrap());
    let err = plan_embed_prune(&model, &selection, Some(&vocab)).unwrap_err();
    match err {
        Error::EmbedPrune(msg) => assert!(msg.contains("matched 0 tokens")),
        other => panic!("expected EmbedPrune, got {other:?}"),
    }
}

#[test]
fn plan_by_pattern_without_vocab_errors() {
    let model = MockEmbedModel::new(4, 8);
    let selection = TokenSelection::ByPattern(regex::Regex::new(".*").unwrap());
    let err = plan_embed_prune(&model, &selection, None).unwrap_err();
    match err {
        Error::EmbedPrune(msg) => assert!(msg.contains("token strings")),
        other => panic!("expected EmbedPrune, got {other:?}"),
    }
}

#[test]
fn plan_by_id_filters_invalid() {
    let model = MockEmbedModel::new(4, 8);
    // IDs 0, 2, 5, 7 — 5 and 7 are >= vocab_size 4
    let selection = TokenSelection::ById(vec![0, 2, 5, 7]);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    assert_eq!(plan.keep_rows, vec![0, 2]);
    assert_eq!(plan.new_vocab_size, 2);
}

#[test]
fn plan_by_id_deduplicates() {
    let model = MockEmbedModel::new(4, 8);
    let selection = TokenSelection::ById(vec![1, 1, 2, 2]);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    assert_eq!(plan.keep_rows, vec![1, 2]);
    assert_eq!(plan.new_vocab_size, 2);
}

#[test]
fn plan_no_embedding_tensor_errors() {
    struct EmptyModel;
    impl crate::model::Model for EmptyModel {
        fn format(&self) -> crate::model::ModelFormat {
            crate::model::ModelFormat::Unknown
        }
        fn name(&self) -> Option<&str> {
            None
        }
        fn architecture(&self) -> Option<&str> {
            None
        }
        fn block_count(&self) -> Option<usize> {
            None
        }
        fn tensors(&self) -> &[crate::model::Tensor] {
            &[]
        }
        fn tensor(&self, _name: &str) -> Option<&crate::model::Tensor> {
            None
        }
        fn metadata(&self, _key: &str) -> Option<crate::model::MetadataValue<'_>> {
            None
        }
        fn read_tensor_bytes(&self, _name: &str) -> crate::Result<Cow<'static, [u8]>> {
            Err(crate::Error::TensorNotFound("".into()))
        }
    }
    let model = EmptyModel;
    let selection = TokenSelection::TopN(10);
    let err = plan_embed_prune(&model, &selection, None).unwrap_err();
    match err {
        Error::EmbedPrune(msg) => assert!(msg.contains("no embedding tensor")),
        other => panic!("expected EmbedPrune, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// apply_embed_prune
// ---------------------------------------------------------------------------

#[test]
fn apply_prune_produces_correct_sizes() {
    let model = MockEmbedModel::new(4, 8);
    let selection = TokenSelection::ById(vec![0, 2]);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    let result = apply_embed_prune(&model, &plan).unwrap();
    // Embedding: 2 rows × 8 cols × 4 bytes = 64 bytes
    let embed = result
        .iter()
        .find(|(n, _)| n == "token_embd.weight")
        .unwrap();
    assert_eq!(embed.1.len(), 2 * 8 * 4);
    // Output: 8 rows × 2 cols × 4 bytes = 64 bytes
    let output = result.iter().find(|(n, _)| n == "output.weight").unwrap();
    assert_eq!(output.1.len(), 8 * 2 * 4);
}

#[test]
fn apply_prune_preserves_row_values() {
    let model = MockEmbedModel::new(4, 2); // 4 rows, 2 cols
    let selection = TokenSelection::ById(vec![1, 3]);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    let result = apply_embed_prune(&model, &plan).unwrap();
    let embed = result
        .iter()
        .find(|(n, _)| n == "token_embd.weight")
        .unwrap();
    let f32s: Vec<f32> = embed
        .1
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Row 1: [2.0, 3.0], Row 3: [6.0, 7.0]
    assert_eq!(f32s, vec![2.0, 3.0, 6.0, 7.0]);
}

#[test]
fn apply_prune_output_cols_preserved() {
    let model = MockEmbedModel::new(4, 2); // 4 rows, 2 cols
    let selection = TokenSelection::ById(vec![0, 3]);
    let plan = plan_embed_prune(&model, &selection, None).unwrap();
    let result = apply_embed_prune(&model, &plan).unwrap();
    let output = result.iter().find(|(n, _)| n == "output.weight").unwrap();
    let f32s: Vec<f32> = output
        .1
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // output.weight is [hidden=2, vocab=4]
    // Col 0: [10000, 10004], Col 3: [10003, 10007]
    // Row 0: [10000, 10003], Row 1: [10004, 10007]
    assert_eq!(f32s, vec![10000.0, 10003.0, 10004.0, 10007.0]);
}

// ---------------------------------------------------------------------------
// extract_rows / extract_cols
// ---------------------------------------------------------------------------

#[test]
fn extract_rows_basic() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows × 2 cols
    let result = extract_rows(&data, 3, 2, &[0, 2]);
    let f32s: Vec<f32> = result
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(f32s, vec![1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn extract_cols_basic() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows × 3 cols
    let result = extract_cols(&data, 2, 3, &[0, 2]);
    let f32s: Vec<f32> = result
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Row 0: col 0=1.0, col 2=3.0; Row 1: col 0=4.0, col 2=6.0
    assert_eq!(f32s, vec![1.0, 3.0, 4.0, 6.0]);
}

#[test]
fn extract_rows_empty_keep() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let result = extract_rows(&data, 2, 2, &[]);
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// resolve_selection edge cases
// ---------------------------------------------------------------------------

#[test]
fn resolve_selection_top_n_zero() {
    let model = MockEmbedModel::new(10, 8);
    let selection = TokenSelection::TopN(0);
    let err = plan_embed_prune(&model, &selection, None).unwrap_err();
    match err {
        Error::EmbedPrune(msg) => assert!(msg.contains("0 tokens")),
        other => panic!("expected EmbedPrune, got {other:?}"),
    }
}

#[test]
fn resolve_selection_by_file() {
    let model = MockEmbedModel::new(4, 8);
    let vocab = vec!["hello".into(), "world".into(), "foo".into(), "bar".into()];

    // Create a temp file with tokens to keep
    let dir = std::env::temp_dir().join("tensorkit_test_embed");
    std::fs::create_dir_all(&dir).unwrap();
    let token_file = dir.join("tokens.txt");
    std::fs::write(&token_file, "hello\nbar\n").unwrap();

    let selection = TokenSelection::ByFile(token_file.clone());
    let plan = plan_embed_prune(&model, &selection, Some(&vocab)).unwrap();
    assert_eq!(plan.keep_rows, vec![0, 3]);
    assert_eq!(plan.new_vocab_size, 2);

    // Cleanup
    std::fs::remove_file(&token_file).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn resolve_selection_by_file_empty_errors() {
    let model = MockEmbedModel::new(4, 8);
    let vocab = vec!["a".into(), "b".into(), "c".into(), "d".into()];

    let dir = std::env::temp_dir().join("tensorkit_test_embed_empty");
    std::fs::create_dir_all(&dir).unwrap();
    let token_file = dir.join("empty.txt");
    std::fs::write(&token_file, "").unwrap();

    let selection = TokenSelection::ByFile(token_file.clone());
    let err = plan_embed_prune(&model, &selection, Some(&vocab)).unwrap_err();
    match err {
        Error::EmbedPrune(msg) => assert!(msg.contains("empty")),
        other => panic!("expected EmbedPrune, got {other:?}"),
    }

    std::fs::remove_file(&token_file).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}
