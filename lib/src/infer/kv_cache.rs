/// KV cache for autoregressive (multi-token) inference.
///
/// Stores keys and values for all previous tokens at each transformer block,
/// enabling efficient attention computation during generation.

/// Per-block key/value storage.
#[derive(Debug, Clone)]
pub struct KvCacheEntry {
    /// Keys: [n_kv_heads * seq_len * head_dim], flattened.
    pub k: Vec<f32>,
    /// Values: [n_kv_heads * seq_len * head_dim], flattened.
    pub v: Vec<f32>,
    /// Number of tokens stored (seq_len).
    pub seq_len: usize,
}

/// KV cache for all transformer blocks.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub entries: Vec<KvCacheEntry>,
}

impl KvCache {
    /// Create an empty KV cache for the given number of blocks.
    pub fn new(n_blocks: usize) -> Self {
        Self {
            entries: vec![KvCacheEntry {
                k: Vec::new(),
                v: Vec::new(),
                seq_len: 0,
            }; n_blocks],
        }
    }

    /// Append new K and V for a single token at the given block.
    ///
    /// `k` and `v` should be `[n_kv_heads * head_dim]` for a single token.
    pub fn append(&mut self, block_idx: usize, k: &[f32], v: &[f32]) {
        let entry = &mut self.entries[block_idx];
        entry.k.extend_from_slice(k);
        entry.v.extend_from_slice(v);
        entry.seq_len += 1;
    }

    /// Get cached K for a block, shaped as `[seq_len * n_kv_heads * head_dim]`.
    pub fn k(&self, block_idx: usize) -> &[f32] {
        &self.entries[block_idx].k
    }

    /// Get cached V for a block, shaped as `[seq_len * n_kv_heads * head_dim]`.
    pub fn v(&self, block_idx: usize) -> &[f32] {
        &self.entries[block_idx].v
    }

    /// Current sequence length.
    pub fn seq_len(&self) -> usize {
        self.entries.first().map_or(0, |e| e.seq_len)
    }

    /// Extract attention scores for all heads given a query.
    ///
    /// Returns `[n_heads * seq_len]` raw scores (before softmax), where
    /// `score[h, t] = q_h · k_{h,t} / sqrt(head_dim)`.
    pub fn compute_attention_scores(
        &self,
        block_idx: usize,
        q: &[f32],           // [n_heads * head_dim]
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) -> Vec<f32> {
        let seq_len = self.seq_len();
        let n_kv = self.entries[block_idx].seq_len;
        if n_kv == 0 {
            return vec![0.0; n_heads * seq_len];
        }

        let n_rep = n_heads / n_kv_heads;
        let mut scores = vec![0.0f32; n_heads * seq_len];

        for h in 0..n_heads {
            let kv_head = h / n_rep.max(1);
            let q_base = h * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();

            for t in 0..seq_len {
                let k_base = t * n_kv_heads * head_dim + kv_head * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_base + d] * self.entries[block_idx].k[k_base + d];
                }
                scores[h * seq_len + t] = dot * scale;
            }
        }

        scores
    }

    /// Extract per-head attention output from cached values.
    ///
    /// Given attention weights `[n_heads * seq_len]`, compute the weighted
    /// sum over cached values for each head.
    pub fn attention_output(
        &self,
        block_idx: usize,
        attn_weights: &[f32], // [n_heads * seq_len] (after softmax)
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) -> Vec<f32> {
        let seq_len = self.seq_len();
        let n_rep = n_heads / n_kv_heads;
        let mut out = vec![0.0f32; n_heads * head_dim];

        for h in 0..n_heads {
            let kv_head = h / n_rep.max(1);
            let o_base = h * head_dim;

            for t in 0..seq_len {
                let weight = attn_weights[h * seq_len + t];
                let k_base = t * n_kv_heads * head_dim + kv_head * head_dim;
                for d in 0..head_dim {
                    out[o_base + d] += weight * self.entries[block_idx].v[k_base + d];
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cache() {
        let cache = KvCache::new(4);
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn append_and_read() {
        let mut cache = KvCache::new(1);
        // Single head, dim=4
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(0, &k, &v);
        assert_eq!(cache.seq_len(), 1);
        assert_eq!(cache.k(0), &k);
        assert_eq!(cache.v(0), &v);

        // Append another token
        let k2 = vec![9.0, 10.0, 11.0, 12.0];
        let v2 = vec![13.0, 14.0, 15.0, 16.0];
        cache.append(0, &k2, &v2);
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(cache.k(0).len(), 8);
    }

    #[test]
    fn attention_scores_single_token() {
        let mut cache = KvCache::new(1);
        // 1 head, dim=2
        let k = vec![1.0, 0.0];
        let v = vec![0.0, 1.0];
        cache.append(0, &k, &v);

        // Query matches key exactly → score should be high
        let q = vec![1.0, 0.0];
        let scores = cache.compute_attention_scores(0, &q, 1, 2, 1);
        assert_eq!(scores.len(), 1);
        // dot = 1*1 + 0*0 = 1.0; scale = 1/sqrt(2) ≈ 0.7071
        let expected = 1.0 / 2.0_f32.sqrt();
        assert!((scores[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn attention_output_single_head() {
        let mut cache = KvCache::new(1);
        // 1 head, dim=2, 1 token
        cache.append(0, &[1.0, 0.0], &[10.0, 20.0]);

        let weights = vec![1.0]; // single token, weight=1
        let out = cache.attention_output(0, &weights, 1, 2, 1);
        assert_eq!(out, vec![10.0, 20.0]);
    }

    #[test]
    fn multi_head_attention_scores() {
        let mut cache = KvCache::new(1);
        // 2 heads, dim=2, 1 token
        // head 0 key: [1, 0], head 1 key: [0, 1]
        cache.append(0, &[1.0, 0.0, 0.0, 1.0], &[10.0, 20.0, 30.0, 40.0]);

        // query: head 0 = [1,0] (matches k0), head 1 = [0,1] (matches k1)
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let scores = cache.compute_attention_scores(0, &q, 2, 2, 2);
        assert_eq!(scores.len(), 2);
        let expected = 1.0 / 2.0_f32.sqrt();
        assert!((scores[0] - expected).abs() < 1e-5);
        assert!((scores[1] - expected).abs() < 1e-5);
    }
}
