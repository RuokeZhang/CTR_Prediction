

from dataclasses import dataclass


@dataclass
class DeepModelConfig:
    # Data & hashing
    hash_buckets: int = 1 << 18  # Cap categorical space; balances collision vs. memory/params

    # Model
    embedding_dim: int = 16       # Higher-capacity than 8; still light for Criteo-scale
    use_embeddings: bool = True  # Enable categorical embeddings

    # Optimization
    lr: float = 8e-4             # Slightly conservative LR; stable with larger batch/embedding
    batch_size: int = 4096       # Throughput-friendly on GPU; reduce on CPU/low-memory devices
    epochs: int = 5              # Balanced for convergence without long runs; increase if underfitting
    lr_step_size: int = 1        # StepLR decay every epoch (keeps training stable)
    lr_gamma: float = 0.5        # Halve LR each step for smoother late-stage convergence

    # Reproducibility
    seed: int = 42               # Fixed seed for torch/np/random

    # Paths (recommended defaults; override via CLI)
    manifest: str = "data/processed/metadata/split_manifest.json"
    checkpoint: str = "checkpoints/deep_model.pt"
    metrics_file: str = "results/deep_model_metrics.csv"
    data_root: str | None = None  # Optional override if manifest paths differ (e.g., Colab /content)


# Rationale:
# - hash_buckets: 2^18 balances collision rate vs. parameter/memory footprint for Criteo-scale.
# - embedding_dim: 16 gives more capacity than 8 while remaining lightweight; lower if memory-bound.
# - lr: 8e-4 is conservative for Adam with larger batch/embeddings; pair with StepLR (gamma=0.5).
# - batch_size: 4096 is efficient on GPUs; lower (1024–2048) on CPU/limited memory.
# - epochs: 5 for a solid baseline; extend to 7–8 if metrics plateau low without overfitting.
# - seed: fixed to ensure reproducibility given the same manifest and hyperparameters.
# - data_root: set when manifest file paths differ from runtime location (e.g., Drive/Colab).

