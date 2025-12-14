# CTR Prediction with DeepFM

PyTorch implementation of DeepFM  for click-through rate prediction on the Criteo dataset. Includes chunked preprocessing, training scripts, and ready-to-run demos.

## Project Overview
- Goal: Train and evaluate CTR model) on Criteo data with reproducible preprocessing and lightweight inference demos.
- Outputs: Metrics (AUC/Logloss), saved checkpoints, and demo predictions.

## Setup Instructions
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data
- Source: Criteo display advertising dataset (Kaggle). Place raw file at `data/raw/criteo_train.txt`.
- Preprocess (chunked, adjust `--limit-chunks` for speed):
```bash
python scripts/preprocess_criteo.py --chunk-size 400000 --limit-chunks 5 --build-manifest
```
- Provided small splits for quick runs: `data/processed/metadata/split_manifest_small.json` (tiny).

## How to Run

- **Model demo with real sample input (auto-download if checkpoint missing):**
```bash
python demo/demo_1.py \
  --checkpoint checkpoints/deep_model.pt \
  --output results/demo_model_predictions.csv \
  --sample-path demo/sample.parquet \
  --model-download-link "https://drive.google.com/file/d/1VFtnxu9fn0oJC9CFdJEbqlXWcgF84UC-/view?usp=sharing"
```
- **Train (example, small split):**


```bash
python src/main.py
```

## Expected Output
- Demos: CSVs in `results/` containing input features (if available) and predicted click probabilities.
- Training: Metrics CSV in `results/deepfm_metrics.csv` (or specified path) and best checkpoints in `checkpoints/`.

## Pre-trained Model Link
- Placeholder: ` https://drive.google.com/file/d/1VFtnxu9fn0oJC9CFdJEbqlXWcgF84UC-/view?usp=sharing` . Place downloaded file as `checkpoints/deepfm.pt` or pass `--checkpoint` to scripts.

## Acknowledgments
- Criteo display advertising dataset (Kaggle).
- DeepFM: Guo et al., 2017.
- Built with PyTorch and related ecosystem tools.

## Reproducibility & Hyperparameters
- Defaults and rationale are documented in `config.py` (hash buckets, embedding dim, lr schedule, batch size, epochs, seed, paths). `src/main.py` reads these defaults.
- Training entry (`src/main.py`) exposes the same knobs via CLI flags; keep manifest/data_root consistent with your environment.
- Set seed (default 42) to reproduce results given the same manifest and hyperparameters.

