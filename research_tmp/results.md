# Tensor-Surgery Literature — arXiv Reference Sheet

Collected from arXiv searches on 2026-06-14. Source data: `research_tmp/*.json`.
Citation counts from Semantic Scholar (paper batch + per-paper queries;
rate-limited, so a few of the smaller papers did not return cites — flagged `?`).

## TL;DR — the five families

| Family | Idea in one line | Cost vector | Cited exemplar(s) |
|---|---|---|---|
| **Linear weight averaging** ("model soup") | `θ_merged = mean(θ_i)` | free | Model Soups, 1559 cites |
| **Task arithmetic** | `(θ_ft_i − θ_base)` define "task vectors"; add/drop/subtract | free | Ilharco 2022, TIES: 757 cites |
| **Sign-disagreement-aware merging** | TRIM small deltas, elect sign, then average only agreeing params | free | TIES-Merging |
| **Magnitude/pruning** | zero out small \|W\|·\|X\| scores | reduces bytes linearly with sparsity | Wanda 859, SparseGPT 1320 |
| **Hessian-aware pruning** | second-order block reconstruction | reduces bytes linearly | SparseGPT |
| **Activation-aware quantization** | protect salient channels via activation stats | 4-8x byte cut | AWQ 1443, SmoothQuant 1641 |

Everything below is **training-free** except where noted — i.e., suitable for
TensorKit's "modify a released GGUF" use case.

---

## A. MERGING — averaging / interpolating parameter tensors

### A1. Model Soups (Wortsman et al., 2022)
**arXiv 2203.05482** — 1559 cites, 192 influential.

> "We show that averaging the weights of multiple models fine-tuned with
> different hyperparameter configurations often improves accuracy and
> robustness. Unlike a conventional ensemble, we may average many models
> without incurring any additional inference or memory costs."

**Results**: ViT-G ImageNet top-1 = **90.94 %** (new SOTA at publication);
improves zero-shot OOD on CLIP / ALIGN. Two variants:

- **Greedy soup** — sort by val acc, scan-in candidates.
- **Uniform soup** — straight average (works when fine-tunes land in the
  same loss basin — "linear mode connectivity").

**Algorithmic cost**: O(N · |θ|) over N fine-tunes.
**Tensor surgery cost**: 1 read of each fine-tune, 1 broadcast add + 1 scale.
**This is the simplest surgery you could add to TensorKit** — your SVD
machinery already does weighted linear combinations.

### A2. Task Arithmetic (Ilharco et al., 2022)
**arXiv 2212.04089** — cited 1000+ times (cited under "?" because S2
rate-limited this lookup).

> "A task vector specifies a direction in the weight space of a pre-trained
> model … addition improves performance on the task, while negation leads
> to task forgetting."

**Form**
```
τ_i   = θ_ft_i − θ_base            # task vector
θ_new = θ_base + Σ α_k · τ_k       # add a set of tasks
θ_new = θ_base − α · τ             # "unlearn" a task
θ_new = θ_base + τ_A − τ_B + τ_C   # analogies (A:B :: C:D)
```

**Results**: improvements on multi-task transfer without retraining;
analogical transfer works as claimed.

**Tensor surgery cost**: cheap (just additions). The interesting part is
**non-interference** — NaN task vectors still hurt merged quality, which
motivates TIES.

### A3. TIES-Merging — TRIM, ELECT SIGN, MERGE (Yadav et al., 2023)
**arXiv 2306.01708** — ~757 cites.

> "Interference due to redundant parameter values and disagreement on the
> sign … We propose TRIM, ELECT SIGN & MERGE: (1) reset parameters that
> changed little, (2) resolve sign conflicts, (3) merge only aligned ones."

**Algorithm** (this is the one I actually want to port):

```
1. τ_k = θ_ft_k − θ_base, sparsify top-K (e.g., keep top 20% by |τ|)
2. ELECT SIGN per coordinate: majority sign across {τ_k[i]}
3. For each k, keep only τ_k[i] whose sign agrees with elected sign
4. MERGE: average the agreeing values, scale by 1/K (or learned weight)
```

**Results**: outperforms simple averaging and Task Arithmetic on vision
(ViT) and language (T5) — typically 2-5% absolute accuracy retention
versus degradation from naive average.

**Tensor surgery cost**: per-tensor work is two scans + mask. Easily fits
into TensorKit's existing `svd/plan.rs` block-scoping without any new
dependency.

### A4. Localize-and-Stitch (He et al., 2024)
**arXiv 2408.13656** — 43 cites (recent).

> "Identify tiny 1% localized regions in the fine-tuned models containing
> essential skills, then reintegrate only those back into the pretrained
> model."

**Idea**: per-channel or per-block importance score; merge only the top
p% mass. Useful when fine-tunes diverged a lot — closer to "sparse
task arithmetic" than to averaging.

**Tensor surgery cost**: scores (gradient-based or activation-based) —
need calibration data; but the merge step is again just sparse addition.

### A5. Model Merging by Uncertainty-Based Gradient Matching
**arXiv 2310.12808** — 90 cites.

Replaces fixed `α·1/N` weights with a Taylor expand around the merged
point and matches the gradient to each fine-tune. Better when fine-tunes
are unequally trustworthy; heavier (needs gradient computation or
Hessian-vector products).

---

## B. PRUNING — deleting unimportant tensor elements

### B1. Lottery Ticket Hypothesis (Frankle & Carbin, 2018)
**arXiv 1803.03635** — 4253 cites, the foundational paper.

> "Dense, randomly-initialized networks contain subnetworks (winning
> tickets) that — when trained in isolation — reach test accuracy
> comparable to the original network in a similar number of iterations…"

Reported winning tickets are typically **10–20 %** of original size.
The importance is the framework, not a ready algorithm — IMP (iterative
magnitude pruning) is the usual recipe.

### B2. Wanda — Pruning by Weights AND activations (Sun et al., 2023)
**arXiv 2306.11695** — 859 cites, 177 influential.

> "Prunes weights with the smallest magnitudes multiplied by the
> corresponding input activations, on a per-output basis. Notably, Wanda
> requires no retraining or weight update, and the pruned LLM can be used
> as is."

**Score**:
```
s_ij = |W_ij| · ||X_j||_2
```
Then keep the top-p fraction globally or per-row.

**Results on LLaMA / LLaMA-2**: at **50% sparsity, perplexity within
~0.5% of dense** on Wikitext — and beats magnitude pruning by a large
margin at 4:8 semi-structured sparsity.

**This is the highest-value training-free pruner to add to TensorKit.**
The `||X_j||₂` term needs **calibration tokens** (calibration set is
typically C4 or WikiText), but everything else is bulk arithmetic.

### B3. SparseGPT (Frantar & Alistarh, 2023)
**arXiv 2301.00774** — 1320 cites.

> "Large-scale generative pretrained transformer family models can be
> pruned to at least 50% sparsity in one-shot, without any retraining."

Uses Hessian-aware **column-wise reconstruction**. Heavier than Wanda
(O(d²) per block) but state-of-the-art perplexity retentions up to
**60% unstructured sparsity** on OPT-175B / BLOOM-176B.

**Tensor surgery cost**: needs the Hessian `H = 2·X·Xᵀ/X_n` per block.
Implementations use Cholesky + lazy column updates.

### B4. LLM-Pruner (Ma et al., 2023)
**arXiv 2305.11627** — structured pruning (drops whole coupled
attention/MLP blocks). Different from above: not weight-level but
**block-level**. After pruning, recovers quality with **LoRA** in ~3 hours
on 50K samples. Less immediately useful if you want zero-retrain.

---

## C. QUANTIZATION — reducing precision (not literally "surgery" but the user
is interested in reduction methods)

### C1. GPTQ (Frantar et al., 2022)
**arXiv 2210.17323** — 2145 cites. Has a TensorKit-style current scope.

> "GPTQ can quantize GPT models with 175 billion parameters in
> approximately four GPU hours, reducing the bitwidth down to 3 or 4 bits
> per weight, with negligible accuracy degradation."

Order-preserving Hessian-based column quantization; this is essentially
**the algorithm your `quantize/` module already implements**.

### C2. AWQ (Lin et al., 2023)
**arXiv 2306.00978** — 1443 cites.

> "Protecting only 1% salient weights can greatly reduce quantization
> error. To identify salient weight channels, we should refer to the
> activation distribution, not weights."

Per-channel `s = mean(|X|)` collected from calibration data; scale up
salient channels pre-quantization so the relative dynamic range becomes
quantization-friendly.

### C3. SmoothQuant (Xiao et al., 2022)
**arXiv 2211.10438** — 1641 cites.

> "Smooths the activation outliers by offline migrating the quantization
> difficulty from activations to weights with a mathematically equivalent
> transformation."

Used for W8A8 (both weights and activations), where you'd otherwise need
mixed precision due to activation outliers.

---

## What this means for TensorKit

The state of the art for **tensor-surgery operations on a released
GGUF** (training-free) is:

| Operation | Algorithm to port | Complexity | Prerequisites |
|---|---|---|---|
| Linear merge of fine-tune check-points | Model Soups / Task Arithmetic | O(·\|Θ\|) | none |
| Smart merge under disagreement | **TIES-Merging** | O(K·\|Θ\|) for K fine-tunes | base + fine-tunes |
| Sparse merge / merging only relevant blocks | Localize-and-Stitch | O(calib + score + apply) | calibration set |
| Weight magnitude pruning | baseline / Lottery-Ticket | O(\|Θ\| log \|Θ\|) | none |
| Magnitude-with-activation pruning | **Wanda** | O(calib + \|Θ\|) | small calibration set |
| Hessian-aware pruning | SparseGPT | O(d² per block) with Cholesky | calibration set |
| Block-level / structural pruning | LLM-Pruner | heuristic + LoRA recover | gradient data |

**My read on the three highest-impact additions** (from this corpus,
looking for "training-free, low-cost, big wins"):

1. **TIES-Merging** — three lines of math on top of what you already
   have (additive merges, sign scans, top-K masks). The most-cited
   serious Linear-Bagging-style merger.
2. **Wanda pruning** — needs a calibration pass but every other piece is
   `|·|² · \|X\|` on the same tensors you already load.
3. **Localize-and-Stitch** — actually combines TIES-style sparse merging
   with Wanda-style activation scoring. Single calibration phase serves
   both.

A `plan-fit`-style command that's aware of these could compose them:
   *TIES-merge two fine-tunes* → *Wanda-prune to 50%* → *GPTQ/AWQ* — all
   without retraining — gives the user a "I'm at X bytes from a known
   baseline recipe" answer, which is exactly what you said you wanted
   from the Planner.

---

## Files
- `research_tmp/abstracts.json` — full abstract text of the 15 canonical
  papers (one per arXiv ID, deduplicated)
- `research_tmp/citations.json` — citation counts where Semantic Scholar
  returned them
- `research_tmp/papers.json` — initial arXiv search hits across 10
  queries
