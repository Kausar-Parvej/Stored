# 🔍 Understanding the MLP-based APSP Model

## 🎯 Goal: Learn Shortest Path Distance Matrix

Given a weighted graph (adjacency + mask), predict the shortest path matrix \( D[i][j] \approx \text{shortest\_path}(i, j) \) using a **pure neural network** — no explicit algorithms.

---

## 🧱 Architecture Overview

```
Input Features [edge, mask, i_pos, j_pos]
↓
Linear Projection (input_proj)
↓
Stack of K Enhanced Relaxation Blocks
↓
Linear Projection to scalar (output_proj)
↓
Soft Path Composition Layer (soft_path_update)
↓
Blended Output
```

---

## 🔹 1. Edge-wise Input Embedding  

```python
H = torch.stack([X, M], dim=-1)  # (B, N, N, 2)
i_pos = torch.arange(N).view(1, N, 1).expand(B, N, N)
j_pos = torch.arange(N).view(1, 1, N).expand(B, N, N)
pos_feat = torch.stack([i_pos, j_pos], dim=-1).float().to(X.device) / N
H = torch.cat([H, pos_feat], dim=-1)  # (B, N, N, 4)
H = self.input_proj(H)  # (B, N, N, hidden)
```

🧠 **Interpretation**: Each edge \( (i, j) \) gets:
- Edge weight
- Binary mask
- Positional IDs `i/N`, `j/N`  
Then projected to a hidden space.

> 🧩 **Insight**: Turns raw graph into learnable edge embeddings.

---

## 🔹 2. Relaxation Blocks — Bellman-Ford-Like

```python
for blk in self.blocks:
    H = blk(H)
```

Inside each block:

```python
row_ctx = A.mean(dim=2, keepdim=True).expand(-1, -1, N, -1)
col_ctx = A.mean(dim=1, keepdim=True).expand(-1, N, -1, -1)
concat = A + row_ctx + col_ctx
flat = concat.view(B * N * N, H)
update = row_ff(flat) + col_ff(flat)
return A + update
```

🧠 **Interpretation**: Each block updates distances using context:
- From source node (row mean)
- To destination node (col mean)

> 🧩 **Vital Insight**: Mimics path relaxation logic like Bellman-Ford.

---

## 🔹 3. Output Projection

```python
raw = self.output_proj(H).squeeze(-1)  # (B, N, N)
```

🧠 **Interpretation**: Predict scalar distance \( d(i, j) \) from hidden vector.

> 🧩 **Insight**: Represents learned belief about shortest-path distance.

---

## 🔹 4. Soft Path Composition (2-hop Logic)

```python
soft = soft_path_update(raw, W)  # (B, N, N)
```

Inside `soft_path_update()`:

```python
i_to_k = pred.unsqueeze(2)
k_to_j = weights.unsqueeze(1)
composed = i_to_k + k_to_j
soft_weights = torch.softmax(-composed / temperature, dim=2)
relaxed = (soft_weights * composed).sum(dim=2)
```

🧠 **Interpretation**: Implements

\[
d(i, j) \approx \min_k [d(i, k) + w(k, j)]
\]

> 🧩 **Vital Insight**: Injects explicit path composition logic.

---

## 🔹 5. Final Blending

```python
return 0.7 * raw + 0.3 * soft
```

🧠 **Interpretation**: Combines learned distances with soft-relaxed estimates.

---

## 🧠 How the Model Learns Shortest Paths

| Component               | Role                                                |
|------------------------|-----------------------------------------------------|
| `input_proj`           | Embed raw edge features                             |
| `relaxation blocks`    | Learn iterative path refinements                    |
| `output_proj`          | Predict final distance scalars                      |
| `soft_path_update`     | Enforce path-based composition via softmin          |
| Constraints/Losses     | Keep output valid: symmetric, non-negative, etc.    |

---

## ✅ Summary

This model works like a **learnable Bellman-Ford**:

- Each block mimics relaxation of paths.
- Final output blends raw prediction with soft composition.
- Supervision + constraints make it obey real-world distance properties.

> 🔍 It's a neural system that **learns shortest paths from examples** — not by rule, but by behavior.