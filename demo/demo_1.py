"""Demo for Logistic Regression CTR model on sample or dummy data."""

from __future__ import annotations

from pathlib import Path
import urllib.request

import torch
import pandas as pd

from src.models.baselines import LogisticRegressionModel
from src.data.module import Batch
from src.data.pipeline import NUMERIC_COLS, CATEGORICAL_COLS
from src.utils import make_dummy_batch


def run_demo_lr(
    checkpoint: Path = Path("checkpoints/deepfm.pt"),
    output: Path = Path("results/demo_predictions.csv"),
    batch_size: int = 8,
    hash_buckets: int = 1 << 18,
    sample_path: Path | None = Path("demo/sample.parquet"),
    model_download_link: str = "<YOUR_LR_MODEL_DOWNLOAD_LINK_HERE>",
) -> None:
    # Auto-download checkpoint if missing and link provided
    if not checkpoint.exists() and model_download_link and "<YOUR_LR_MODEL_DOWNLOAD_LINK_HERE>" not in model_download_link:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Checkpoint not found. Downloading from {model_download_link} ...")
            urllib.request.urlretrieve(model_download_link, checkpoint)
            print(f"Downloaded checkpoint to {checkpoint}")
        except Exception as e:
            raise RuntimeError(f"Failed to download checkpoint from {model_download_link}") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRegressionModel(
        num_numeric=13,
        num_categorical=26,
        hash_bucket_size=hash_buckets,
        embedding_dim=8,
        use_embeddings=True,
    ).to(device)

    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)

    else:
        print("Checkpoint not found; using randomly initialized LR model.")

    model.eval()

    if sample_path is not None and sample_path.exists():
        df = pd.read_parquet(sample_path)
        df = df.head(batch_size) if len(df) > batch_size else df
        numerical = torch.tensor(df[NUMERIC_COLS].to_numpy(dtype="float32"), device=device)
        categorical = torch.tensor(df[CATEGORICAL_COLS].to_numpy(dtype="int64"), device=device)
        labels = torch.zeros((len(df), 1), dtype=torch.float32, device=device)
        batch = Batch(numerical=numerical, categorical=categorical, label=labels)
        df_out = df.copy()
    else:
        batch = make_dummy_batch(batch_size, 13, 26, hash_buckets, device)
        df_out = pd.DataFrame(
            data=torch.cat([batch.numerical.cpu(), batch.categorical.cpu()], dim=1).numpy(),
            columns=NUMERIC_COLS + CATEGORICAL_COLS,
        )

    with torch.no_grad():
        probs = model(batch).squeeze(-1).cpu().numpy()

    df_out["probability"] = probs
    output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output, index=False)
    print(f"LR demo finished. Predictions with inputs saved to {output}")


if __name__ == "__main__":
    run_demo_lr()

