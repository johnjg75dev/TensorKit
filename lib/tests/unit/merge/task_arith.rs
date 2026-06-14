use super::*;

// ---------------------------------------------------------------------------
// compute_task_vector
// ---------------------------------------------------------------------------

#[test]
fn compute_task_vector_basic() {
    let base = [1.0f32, 2.0, 3.0];
    let ft = [1.5f32, 2.5, 2.0];
    let tau = compute_task_vector(&base, &ft).unwrap();
    assert_eq!(tau, vec![0.5, 0.5, -1.0]);
}

#[test]
fn compute_task_vector_length_mismatch() {
    let base = [1.0f32, 2.0];
    let ft = [1.0f32, 2.0, 3.0];
    let err = compute_task_vector(&base, &ft).unwrap_err();
    match err {
        Error::TaskArith(msg) => assert!(msg.contains("length mismatch")),
        other => panic!("expected TaskArith, got {other:?}"),
    }
}

#[test]
fn compute_task_vector_identical() {
    let a = [5.0f32, 10.0, 15.0];
    let b = [5.0f32, 10.0, 15.0];
    let tau = compute_task_vector(&a, &b).unwrap();
    assert_eq!(tau, vec![0.0, 0.0, 0.0]);
}

#[test]
fn compute_task_vector_empty() {
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let tau = compute_task_vector(&a, &b).unwrap();
    assert!(tau.is_empty());
}

// ---------------------------------------------------------------------------
// apply_task_vector
// ---------------------------------------------------------------------------

#[test]
fn apply_task_vector_zero_alpha() {
    let base = [1.0f32, 2.0, 3.0];
    let tau = [10.0f32, 10.0, 10.0];
    let result = apply_task_vector_owned(&base, &tau, 0.0);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn apply_task_vector_alpha_one() {
    let base = [1.0f32, 2.0, 3.0];
    let tau = [0.5f32, 0.5, -1.0];
    let result = apply_task_vector_owned(&base, &tau, 1.0);
    assert_eq!(result, vec![1.5, 2.5, 2.0]);
}

#[test]
fn apply_task_vector_alpha_half() {
    let base = [0.0f32, 0.0, 0.0];
    let tau = [2.0f32, 4.0, 6.0];
    let result = apply_task_vector_owned(&base, &tau, 0.5);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn apply_task_vector_into_reuses_buffer() {
    let base = [10.0f32, 20.0];
    let tau = [1.0f32, -2.0];
    let mut out = [0.0f32; 2];
    apply_task_vector(&mut out, &base, &tau, 3.0);
    // 10 + 3*1 = 13, 20 + 3*(-2) = 14
    assert_eq!(out, [13.0, 14.0]);
}

#[test]
#[should_panic(expected = "out length")]
fn apply_task_vector_panics_on_out_mismatch() {
    let base = [1.0f32, 2.0];
    let tau = [1.0f32, 2.0];
    let mut out = [0.0f32; 3];
    apply_task_vector(&mut out, &base, &tau, 1.0);
}

#[test]
#[should_panic(expected = "base length")]
fn apply_task_vector_panics_on_tau_mismatch() {
    let base = [1.0f32, 2.0];
    let tau = [1.0f32];
    let mut out = [0.0f32; 2];
    apply_task_vector(&mut out, &base, &tau, 1.0);
}

// ---------------------------------------------------------------------------
// trim_task_vector
// ---------------------------------------------------------------------------

#[test]
fn trim_top_50_percent() {
    // |τ|: 1, 2, 3, 4 → keep top 2 (indices 3, 2)
    let tau = [1.0f32, 2.0, 3.0, 4.0];
    let trimmed = trim_task_vector(&tau, 0.5).unwrap();
    assert_eq!(trimmed, vec![0.0, 0.0, 3.0, 4.0]);
}

#[test]
fn trim_density_one_keeps_all() {
    let tau = [1.0f32, -2.0, 3.0];
    let trimmed = trim_task_vector(&tau, 1.0).unwrap();
    assert_eq!(trimmed, vec![1.0, -2.0, 3.0]);
}

#[test]
fn trim_density_zero_errors() {
    let tau = [1.0f32, 2.0];
    let err = trim_task_vector(&tau, 0.0).unwrap_err();
    match err {
        Error::TaskArith(msg) => assert!(msg.contains("density")),
        other => panic!("expected TaskArith, got {other:?}"),
    }
}

#[test]
fn trim_density_negative_errors() {
    let tau = [1.0f32, 2.0];
    let err = trim_task_vector(&tau, -0.5).unwrap_err();
    assert!(matches!(err, Error::TaskArith(_)));
}

#[test]
fn trim_density_above_one_errors() {
    let tau = [1.0f32, 2.0];
    let err = trim_task_vector(&tau, 1.5).unwrap_err();
    assert!(matches!(err, Error::TaskArith(_)));
}

#[test]
fn trim_empty_vector() {
    let tau: [f32; 0] = [];
    let trimmed = trim_task_vector(&tau, 0.5).unwrap();
    assert!(trimmed.is_empty());
}

#[test]
fn trim_small_dense_keeps_at_least_one() {
    // density=0.01 on 4 elements → ceil(0.04) = 1 element kept
    let tau = [1.0f32, 2.0, 3.0, 4.0];
    let trimmed = trim_task_vector(&tau, 0.01).unwrap();
    // Only the largest element (4.0 at index 3) should survive
    assert_eq!(trimmed, vec![0.0, 0.0, 0.0, 4.0]);
}

// ---------------------------------------------------------------------------
// elect_sign
// ---------------------------------------------------------------------------

#[test]
fn elect_sign_majority_positive() {
    let a = [1.0f32, -2.0, 3.0];
    let b = [0.5f32, -1.0, 4.0];
    let signs = elect_sign(&[&a, &b]);
    assert_eq!(signs, vec![1.0, -1.0, 1.0]);
}

#[test]
fn elect_sign_tiebreak_positive() {
    let a = [1.0f32, 0.0];
    let b = [-1.0f32, 0.0];
    let signs = elect_sign(&[&a, &b]);
    // coord 0: 1 positive, 1 negative → tie → +1.0
    // coord 1: all zero → +1.0
    assert_eq!(signs, vec![1.0, 1.0]);
}

#[test]
fn elect_sign_all_zero() {
    let a = [0.0f32, 0.0];
    let b = [0.0f32, 0.0];
    let signs = elect_sign(&[&a, &b]);
    assert_eq!(signs, vec![1.0, 1.0]);
}

#[test]
fn elect_sign_empty() {
    let signs: Vec<f32> = elect_sign(&[]);
    assert!(signs.is_empty());
}

#[test]
fn elect_sign_single_vector() {
    let a = [-1.0f32, 5.0];
    let signs = elect_sign(&[&a]);
    assert_eq!(signs, vec![-1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// ties_merge
// ---------------------------------------------------------------------------

#[test]
fn ties_merge_basic() {
    let a = [10.0f32, 0.1, -5.0];
    let b = [-8.0f32, 0.2, -4.0];
    let config = TiesConfig {
        density: 0.67,
        elect_sign: true,
    };
    let merged = ties_merge(&[&a, &b], &config).unwrap();
    // density 0.67 → keep top 2 of 3 per vector
    // a trimmed: [10.0, 0.0, -5.0]
    // b trimmed: [-8.0, 0.0, -4.0]
    // elect sign: coord 0: pos(2) > neg(0) → +, coord 1: all zero → +, coord 2: neg(2) > pos(0) → -
    // merged: coord 0: (10 + -8)/2 = 1.0, coord 1: 0.0, coord 2: (-5 + -4)/2 = -4.5
    assert_eq!(merged.len(), 3);
    assert!((merged[0] - 1.0).abs() < 1e-5);
    assert!(merged[1].abs() < 1e-5);
    assert!((merged[2] - (-4.5)).abs() < 1e-5);
}

#[test]
fn ties_merge_single_vector() {
    let a = [1.0f32, -2.0, 3.0];
    let config = TiesConfig {
        density: 0.5,
        elect_sign: true,
    };
    let merged = ties_merge(&[&a], &config).unwrap();
    // Keep top 2 of 3: |1|=1, |-2|=2, |3|=3 → keep 3 and -2
    assert_eq!(merged, vec![0.0, -2.0, 3.0]);
}

#[test]
fn ties_merge_empty_vectors_errors() {
    let config = TiesConfig {
        density: 0.5,
        elect_sign: true,
    };
    let err = ties_merge(&[], &config).unwrap_err();
    assert!(matches!(err, Error::TaskArith(_)));
}

#[test]
fn ties_merge_no_elect_sign() {
    let a = [10.0f32, -5.0];
    let b = [-8.0f32, -4.0];
    let config = TiesConfig {
        density: 1.0,
        elect_sign: false,
    };
    let merged = ties_merge(&[&a, &b], &config).unwrap();
    // No elect sign → default +1.0 for all coords
    // coord 0: only positive values kept → 10.0 (b[0] is negative, skipped)
    // coord 1: only positive values kept → none are positive → 0.0
    assert!((merged[0] - 10.0).abs() < 1e-5);
    assert!(merged[1].abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// MultiplierHub
// ---------------------------------------------------------------------------

#[test]
fn multiplier_hub_basic() {
    let base = [0.0f32, 0.0, 0.0];
    let tau1 = [1.0f32, 2.0, 3.0];
    let tau2 = [3.0f32, 2.0, 1.0];
    let mut hub = MultiplierHub::new();
    hub.add("task_a", tau1, 0.5);
    hub.add("task_b", tau2, 0.5);
    // 0 + 0.5*1 + 0.5*3 = 2.0, 0 + 0.5*2 + 0.5*2 = 2.0, 0 + 0.5*3 + 0.5*1 = 2.0
    let result = hub.apply_owned(&base);
    assert_eq!(result, vec![2.0, 2.0, 2.0]);
}

#[test]
fn multiplier_hub_remove() {
    let mut hub = MultiplierHub::new();
    hub.add("a", vec![1.0], 1.0);
    assert!(hub.remove("a"));
    assert!(!hub.remove("nonexistent"));
    assert!(hub.list().is_empty());
}

#[test]
fn multiplier_hub_set_alpha() {
    let mut hub = MultiplierHub::new();
    hub.add("a", vec![10.0], 1.0);
    assert!(hub.set_alpha("a", 2.0));
    assert!(!hub.set_alpha("nonexistent", 1.0));
    let result = hub.apply_owned(&[0.0]);
    assert_eq!(result, vec![20.0]);
}

#[test]
fn multiplier_hub_empty_is_noop() {
    let hub = MultiplierHub::new();
    let result = hub.apply_owned(&[1.0, 2.0, 3.0]);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn multiplier_hub_list() {
    let mut hub = MultiplierHub::new();
    hub.add("x", vec![1.0], 0.5);
    hub.add("y", vec![2.0], 1.5);
    let list = hub.list();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0], ("x", 0.5));
    assert_eq!(list[1], ("y", 1.5));
}

#[test]
fn multiplier_hub_apply_into() {
    let base = [1.0f32, 2.0];
    let tau = [3.0f32, 4.0];
    let mut hub = MultiplierHub::new();
    hub.add("t", tau, 0.25);
    let mut out = [0.0f32; 2];
    hub.apply(&mut out, &base);
    // 1 + 0.25*3 = 1.75, 2 + 0.25*4 = 3.0
    assert!((out[0] - 1.75).abs() < 1e-5);
    assert!((out[1] - 3.0).abs() < 1e-5);
}

#[test]
fn multiplier_hub_apply_ties() {
    let base = [0.0f32, 0.0, 0.0];
    let mut hub = MultiplierHub::new();
    hub.add("a", vec![10.0, 0.1, -5.0], 1.0);
    hub.add("b", vec![-8.0, 0.2, -4.0], 1.0);
    let config = TiesConfig {
        density: 0.67,
        elect_sign: true,
    };
    let result = hub.apply_ties(&base, &config).unwrap();
    // Same as ties_merge result since base is zero
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!(result[1].abs() < 1e-5);
    assert!((result[2] - (-4.5)).abs() < 1e-5);
}

#[test]
fn multiplier_hub_apply_ties_empty() {
    let hub = MultiplierHub::new();
    let config = TiesConfig {
        density: 0.5,
        elect_sign: true,
    };
    let result = hub.apply_ties(&[1.0, 2.0], &config).unwrap();
    assert_eq!(result, vec![1.0, 2.0]);
}

#[test]
#[should_panic(expected = "out length")]
fn multiplier_hub_apply_panics_on_length_mismatch() {
    let hub = MultiplierHub::new();
    let mut out = [0.0f32; 3];
    hub.apply(&mut out, &[1.0, 2.0]);
}
