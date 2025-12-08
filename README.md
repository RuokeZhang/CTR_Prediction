# DeepCTR – Click-Through Rate Prediction with DeepFM

PyTorch implementation of a DeepFM-based CTR prediction stack for the 5GB Criteo
display advertising dataset. The project includes reproducible preprocessing,
baseline models (logistic regression, shallow MLP), and a production-style
DeepFM architecture.

## Project Layout
- `docs/deepctr_plan.md` – bilingual project plan.
- `src/data/` – ingestion pipeline, batching utilities, and split manifest helpers.
- `src/models/` – baseline networks and DeepFM implementation.
- `src/eval/` – shared evaluation metrics.
- `scripts/` – CLI entrypoints for preprocessing and model training.
- `reports/` – generated metrics, checkpoints, and written analyses.
- `data/raw/` – place the downloaded Criteo TSV here (ignored by git).
- `data/processed/` – auto-generated parquet caches + metadata.

## Handling the 5GB Dataset
1. Download `train.txt` from the [Criteo Kaggle competition](https://www.kaggle.com/competitions/display-advertising-challenge).
2. Move the file to `data/raw/criteo_train.txt`.
3. Run preprocessing with chunked streaming to avoid RAM spikes:
   ```bash
   # Make sure virtual environment is activated: source .venv/bin/activate
   python scripts/preprocess_criteo.py --chunk-size 400000 --limit-chunks 5 --build-manifest  # smoke test
   python scripts/preprocess_criteo.py --chunk-size 400000 --build-manifest                  # full run
   ```

## Training
Logistic regression or shallow MLP:
```bash
python scripts/train_baselines.py --model lr --epochs 1 --device cpu
python scripts/train_baselines.py --model mlp --epochs 3 --device cuda
```

DeepFM (with optional mixed precision on GPU):
```bash
python scripts/train_deepfm.py --epochs 3 --mixed-precision
```

Each script logs CSV metrics under `reports/` and DeepFM also stores
`reports/deepfm.pt` checkpoints when validation AUC improves.

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies and install the project in editable mode:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

This installs the project as a package, allowing `from src.data import ...` imports to work correctly.

