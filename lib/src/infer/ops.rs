/// Matrix multiply: C [m×n] = A [m×k] × B [k×n], all row-major.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_val = a[i * k + l];
            let base_b = l * n;
            let base_c = i * n;
            for j in 0..n {
                c[base_c + j] += a_val * b[base_b + j];
            }
        }
    }
    c
}

/// RMSNorm: y[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(x[i] * inv * weight[i]);
    }
    out
}

/// Rotary position embedding (RoPE) applied in-place to q and k.
/// `pos` is the token position, `head_dim` is the dimension per head.
pub fn rope_inplace(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize) {
    let n_heads_q = q.len() / head_dim;
    let n_heads_k = k.len() / head_dim;

    for h in 0..n_heads_q {
        let base = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let theta = (pos as f32) * 10000.0f32.powf(-(i as f32 / head_dim as f32));
            let cos = theta.cos();
            let sin = theta.sin();
            let q0 = q[base + i];
            let q1 = q[base + i + 1];
            q[base + i] = q0 * cos - q1 * sin;
            q[base + i + 1] = q0 * sin + q1 * cos;
        }
    }
    for h in 0..n_heads_k {
        let base = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let theta = (pos as f32) * 10000.0f32.powf(-(i as f32 / head_dim as f32));
            let cos = theta.cos();
            let sin = theta.sin();
            let k0 = k[base + i];
            let k1 = k[base + i + 1];
            k[base + i] = k0 * cos - k1 * sin;
            k[base + i + 1] = k0 * sin + k1 * cos;
        }
    }
}

/// Softmax over the last dimension (each row independently).
pub fn softmax(x: &mut [f32], n_cols: usize) {
    for row in x.chunks_mut(n_cols) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// SiLU activation: x * sigmoid(x)
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= 1.0 / (1.0 + (-*v).exp());
    }
}

/// Add two vectors element-wise.
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += *b;
    }
}

/// Compute L2 norm of a vector.
pub fn l2_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}
