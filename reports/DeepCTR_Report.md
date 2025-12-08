# DeepCTR Experiment Report

## Dataset & Preprocessing
- Source: Criteo display advertising challenge (4.5B impressions subset, ~5GB TSV).
- Data cleaning:
  - Streamed ingestion with 400k row chunks to cap memory.
  - Numeric features downcast to `float32`, missing values filled with zeros before standardization.
  - Categorical features hashed into 262k buckets per field to keep embeddings bounded.
- Cached parquet shards + manifest enable deterministic train/val/test splits without re-reading the raw TSV.

## Baseline Results (placeholders)
| Model | Epochs | Val AUC | Val Logloss | Notes |
|-------|--------|---------|-------------|-------|
| Logistic Regression | TODO | TODO | TODO | Linear baseline with hashed categorical scaling. |
| Shallow MLP | TODO | TODO | TODO | Two hidden layers (128/64) + dropout 0.1. |

## DeepFM Results (placeholders)
| Config | Emb Dim | Deep Layers | Val AUC | Val Logloss | Notes |
|--------|---------|-------------|---------|-------------|-------|
| Default | 16 | 256-128 | TODO | TODO | Mixed precision recommended on GPU. |

## Analysis & Next Steps
1. **Calibration** – Use `src/eval/metrics.py` to extend with calibration curves and reliability diagrams once metrics are logged.
2. **Feature Insights** – Leverage the notebook at `notebooks/model_report.ipynb` to inspect which hashed buckets contribute most to lift.
3. **Productionization** – Export best checkpoint from `scripts/train_deepfm.py` (`reports/deepfm.pt`) and wrap inference in a TorchScript or ONNX graph.
4. **Data Refresh Cadence** – Re-run `scripts/preprocess_criteo.py --build-manifest` for each new data dump to keep cached shards consistent.

