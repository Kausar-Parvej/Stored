# 🔍 Understanding BellmanFordMLP: Structure & Learning Process

This document explains the architecture and learning process of the `BellmanFordMLP` model — a neural shortest-path estimator — and compares it with a standard MLP.

---

## 🧱 Model Structure: BellmanFordMLP

### 🔹 Input

Each graph is processed as a complete matrix of edge-pair features:

- **Input shape**: `(B, N, N, 4)` where:
  - `X[i][j]`: edge weight
  - `M[i][j]`: edge mask (1 or 0)
  - `i_pos, j_pos`: normalized node indices

```python
H = torch.stack([X, M], dim=-1)  # (B, N, N, 2)
i_pos = torch.arange(N).view(1, N, 1).expand(B, N, N)
j_pos = torch.arange(N).view(1, 1, N).expand(B, N, N)
pos_feat = torch.stack([i_pos, j_pos], dim=-1).float().to(X.device) / N
H = torch.cat([H, pos_feat], dim=-1)  # → (B, N, N, 4)
H = self.input_proj(H)               # → (B, N, N, hidden)
```

---

### 🔹 Iterative Relaxation (Block Loop)

Each block mimics a learned Bellman-Ford relaxation step:

```python
row_ctx = A.mean(dim=2, keepdim=True)  # Avg outgoing from i
col_ctx = A.mean(dim=1, keepdim=True)  # Avg incoming to j
concat = A + row_ctx + col_ctx
flat = concat.view(B * N * N, H)
update = row_ff(flat) + col_ff(flat)
A = A + update  # Residual update
```

- Each block refines `d(i,j)` using learned context
- `n_blocks = n - 1` to match Bellman-Ford’s depth

---

### 🔹 Output

```python
raw = self.output_proj(H).squeeze(-1)  # → (B, N, N)
return 0.7 * raw + 0.3  # (placeholder blend)
```

---

## 🔁 Learning Process per Epoch

### 🔸 Step-by-Step

1. **Prepare input** `(X, M)` and ground truth `y`
2. **Forward pass**:
   - Project to hidden space
   - Iterate through blocks (refining edge distances)
   - Predict scalar distance matrix
3. **Compute loss**:
   - MSE + constraints (symmetry, diagonal, triangle, etc.)
4. **Backpropagation**:
   - Gradients update: input projection, block MLPs, output projection

### 🔸 Over Epochs

- Model improves path reasoning per edge
- Learns that `i → j` can be estimated through paths like `i → k → j`
- Gradually reduces prediction error across many graphs

---

## 🔄 Epoch vs Block

| Unit      | Scope                     | Learns                       |
|-----------|---------------------------|------------------------------|
| Epoch     | Across all training graphs | To generalize logic          |
| Block     | Within a single graph      | To refine per-pair distances |

---

## 🔬 Hyperparameter Meanings

```python
hidden=128         # Dimensionality of edge embedding
n_blocks=n - 1     # Number of relaxation steps (depth)
dropout=0.2        # Regularization during training
```

---

## 🔁 Comparison: Basic MLP vs BellmanFordMLP

| Aspect                      | Basic MLP                        | BellmanFordMLP                            |
|----------------------------|----------------------------------|--------------------------------------------|
| Input                      | Flat vector                      | Edge-pair matrix (i,j)                     |
| Structure awareness        | ❌ None                          | ✅ Positional + adjacency-aware           |
| Iterative refinement       | ❌ One-shot pass                 | ✅ Multiple update blocks                 |
| Context from neighbors     | ❌ No                            | ✅ Row/Column mean context                |
| Soft path logic            | ❌ No                            | ✅ Optional (soft_path_update)            |
| Goal                       | Generic prediction               | Shortest path approximation               |

---

## ✅ Summary

- BellmanFordMLP mimics path-finding behavior by refining distance estimates across learned layers
- It updates edge beliefs using context from both ends of each edge
- Over epochs, it evolves from random guesser → structure-aware predictor
- Compared to a basic MLP, it’s much more suited for structured graph tasks like APSP