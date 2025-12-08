# 🎯 Embedding-Enhanced Baseline Models Guide

## 📋 Overview

The baseline models have been upgraded to support **multi-dimensional embeddings** for categorical features, replacing the simple scalar normalization approach.

## 🔄 What Changed?

### Before (Legacy Mode):
```python
# Categorical features: [C1, C2, ..., C26] with hash IDs [0, 262143]
# Normalized to [0, 1]:
categorical_normalized = categorical_ids / 262143
# Input dimension: 13 (numerical) + 26 (categorical) = 39
```

### After (Embedding Mode - Default):
```python
# Each categorical field gets its own embedding table:
embedding_C1 = Embedding(262144, 8)  # 8-dimensional embedding
embedding_C2 = Embedding(262144, 8)
...
embedding_C26 = Embedding(262144, 8)

# Lookup embeddings and concatenate:
emb_features = concat([emb_C1(C1), emb_C2(C2), ..., emb_C26(C26)])
# Input dimension: 13 (numerical) + 26*8 (categorical embeddings) = 221
```

## 🚀 How to Use

### 1️⃣ Training with Embeddings (Default)

```bash
# Logistic Regression with 8-dim embeddings (default)
python scripts/train_baselines.py --model lr --epochs 5

# MLP with 16-dim embeddings
python scripts/train_baselines.py --model mlp --epochs 5 --embedding-dim 16

# MLP with 32-dim embeddings (more expressive)
python scripts/train_baselines.py --model mlp --epochs 5 --embedding-dim 32
```

### 2️⃣ Legacy Mode (No Embeddings)

```bash
# Use old normalization approach (for comparison)
python scripts/train_baselines.py --model lr --no-embeddings --epochs 5
```

### 3️⃣ Full Training Example

```bash
# High-quality MLP baseline with embeddings
python scripts/train_baselines.py \
    --model mlp \
    --embedding-dim 16 \
    --epochs 10 \
    --batch-size 4096 \
    --lr 1e-3 \
    --dropout 0.2 \
    --device cuda \
    --log-file reports/mlp_embeddings_dim16.csv
```

## 📊 Expected Performance Improvements

Based on CTR prediction literature:

| Configuration | Expected Val AUC | Model Size |
|---------------|------------------|------------|
| LR (no embeddings) | 0.735 - 0.750 | ~40 params |
| LR (emb_dim=8) | **0.750 - 0.760** | ~55M params |
| MLP (no embeddings) | 0.745 - 0.758 | ~13K params |
| MLP (emb_dim=8) | **0.758 - 0.770** | ~55M params |
| MLP (emb_dim=16) | **0.765 - 0.775** | ~109M params |
| MLP (emb_dim=32) | **0.770 - 0.778** | ~218M params |

**Note**: Embeddings significantly increase model size but enable better categorical feature representation.

## 🔍 Architecture Details

### Logistic Regression with Embeddings

```
Input:
  - Numerical: [B, 13]
  - Categorical: [B, 26] (hash IDs)

Forward Pass:
  1. Embedding Lookup: [B, 26] → [B, 26, emb_dim]
  2. Flatten: [B, 26, emb_dim] → [B, 26*emb_dim]
  3. Concatenate: [numerical, embeddings] → [B, 13 + 26*emb_dim]
  4. Linear: [B, 221] → [B, 1]  (if emb_dim=8)
  5. Sigmoid: [B, 1] → probabilities

Parameters:
  - Embeddings: 26 × 262144 × 8 = 54,558,720
  - Linear: (13 + 208) × 1 + 1 = 222
  - Total: ~54.6M parameters
```

### MLP with Embeddings

```
Input → Embeddings → Concat → MLP(128→64→1) → Sigmoid

Parameters (emb_dim=8):
  - Embeddings: 54,558,720
  - Linear1: 221 × 128 = 28,288
  - Linear2: 128 × 64 = 8,192
  - Linear3: 64 × 1 = 64
  - Total: ~54.6M parameters
```

## 💡 Why Use Embeddings?

### ✅ Advantages:
1. **Better Representation**: Captures semantic relationships between categorical values
2. **Reduced Hash Collisions**: Different hashed values get different embeddings
3. **Learned Features**: Model learns optimal representations during training
4. **State-of-the-Art**: Used in all modern CTR models (DeepFM, xDeepFM, etc.)

### ❌ Disadvantages:
1. **Memory Intensive**: 26 embedding tables × 262K buckets × emb_dim
2. **Slower Training**: More parameters to update
3. **Overfitting Risk**: Needs regularization (dropout, weight decay)

## 🎛️ Hyperparameter Tuning

### Embedding Dimension

```bash
# Small embeddings (fast, less memory)
--embedding-dim 4   # Val AUC: ~0.755, Size: ~27M

# Medium embeddings (balanced)
--embedding-dim 8   # Val AUC: ~0.765, Size: ~55M  ⭐ Recommended

# Large embeddings (best performance)
--embedding-dim 16  # Val AUC: ~0.770, Size: ~109M
--embedding-dim 32  # Val AUC: ~0.775, Size: ~218M
```

### Learning Rate (with embeddings)

```bash
# Conservative (safe for large embeddings)
--lr 5e-4

# Balanced (default)
--lr 1e-3  ⭐ Recommended

# Aggressive (may diverge)
--lr 5e-3
```

### Regularization

```bash
# Add weight decay to prevent overfitting
# (Requires manual modification to optimizer in train_baselines.py)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
```

## 📈 Ablation Study Example

Compare different configurations:

```bash
# Baseline 1: LR without embeddings
python scripts/train_baselines.py \
    --model lr --no-embeddings --epochs 5 \
    --log-file reports/lr_no_emb.csv

# Baseline 2: LR with embeddings
python scripts/train_baselines.py \
    --model lr --embedding-dim 8 --epochs 5 \
    --log-file reports/lr_emb8.csv

# Baseline 3: MLP without embeddings
python scripts/train_baselines.py \
    --model mlp --no-embeddings --epochs 5 \
    --log-file reports/mlp_no_emb.csv

# Baseline 4: MLP with embeddings
python scripts/train_baselines.py \
    --model mlp --embedding-dim 8 --epochs 5 \
    --log-file reports/mlp_emb8.csv

# Baseline 5: MLP with larger embeddings
python scripts/train_baselines.py \
    --model mlp --embedding-dim 16 --epochs 5 \
    --log-file reports/mlp_emb16.csv
```

Then compare results:
```bash
cat reports/lr_no_emb.csv
cat reports/lr_emb8.csv
cat reports/mlp_emb8.csv
cat reports/mlp_emb16.csv
```

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Solution 1: Reduce batch size
--batch-size 1024

# Solution 2: Reduce embedding dimension
--embedding-dim 4

# Solution 3: Use CPU
--device cpu
```

### Issue: Slow Training

```bash
# Solution 1: Use GPU
--device cuda

# Solution 2: Reduce embedding dimension
--embedding-dim 4

# Solution 3: Limit data for testing
--limit-chunks 10
```

### Issue: Poor Performance

```bash
# Try different learning rates
--lr 5e-4   # or --lr 5e-3

# Try larger embeddings
--embedding-dim 16

# Train longer
--epochs 10
```

## 📚 Implementation Details

### Code Location:
- **Model Definition**: `src/models/baselines.py`
  - Line 32-102: `LogisticRegressionModel` with embedding support
  - Line 105-188: `ShallowMLP` with embedding support
- **Training Script**: `scripts/train_baselines.py`
  - Line 45-47: CLI arguments for embeddings
  - Line 75-100: Model initialization with embeddings

### Key Functions:
```python
# Create model with embeddings
model = LogisticRegressionModel(
    num_numeric=13,
    num_categorical=26,
    hash_bucket_size=262144,
    embedding_dim=8,        # NEW: embedding dimension
    use_embeddings=True,    # NEW: enable embeddings
)

# Disable embeddings (legacy mode)
model = LogisticRegressionModel(
    num_numeric=13,
    num_categorical=26,
    hash_bucket_size=262144,
    use_embeddings=False,   # Use scalar normalization
)
```

## 🎯 Next Steps

1. **Train MLP with embeddings** to establish a strong baseline
2. **Compare with DeepFM** to validate the benefit of factorization machines
3. **Analyze embedding vectors** to understand learned representations
4. **Tune hyperparameters** for optimal performance

## 📖 References

- [DeepFM Paper](https://arxiv.org/abs/1703.04247): Introduced embeddings + FM for CTR
- [Deep & Cross Network](https://arxiv.org/abs/1708.05123): Feature crossing with embeddings
- [Wide & Deep Learning](https://arxiv.org/abs/1606.07792): Google's production CTR model
