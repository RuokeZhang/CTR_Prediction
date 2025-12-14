# CTR Prediction with DeepFM

PyTorch implementation of DeepFM for Criteo-style CTR prediction. Includes chunked preprocessing, train/eval scripts, and a minimal demo.

## Project Layout
```
├── README.md
├── requirements.txt
├── src/
│   ├── main.py       # quick demo entry
│   ├── utils.py      # helpers (dummy batch, save preds)
│   ├── model.py      # DeepFM loader / wrappers
│   └── models/…      # actual model implementations
├── data/             # raw/processed placeholders (Kaggle Criteo link)
├── checkpoints/      # saved model weights (deepfm.pt)
├── demo/             # demo script (demo.py)
├── results/          # generated outputs (metrics, demo predictions)
└── scripts/          # preprocess/train CLIs
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data (public source)
- Download Criteo display advertising train set from Kaggle and place at `data/raw/criteo_train.txt`.
- Process in chunks (adjust `--limit-chunks` for speed):
```bash
python scripts/preprocess_criteo.py --chunk-size 400000 --limit-chunks 5 --build-manifest
```
- A tiny split for fast experiments is provided at `data/processed/metadata/split_manifest_small.json`.

## How to Run
- **Quick demo (no data needed, uses dummy inputs):**
```bash
python src/main.py --checkpoint checkpoints/deepfm.pt --output results/demo_predictions.csv
```
- **Train DeepFM (GPU or CPU):**
```bash
python scripts/train_deepfm.py \
  --manifest data/processed/metadata/split_manifest_small.json \
  --batch-size 4096 --epochs 3 --mixed-precision
```
Switch `--manifest` to `data/processed/metadata/split_manifest.json` for full data.

## Expected Output
- Demo writes `results/demo_predictions.csv` with sample click probabilities.
- Training writes metrics to `results/deepfm_metrics.csv` and best checkpoint to `checkpoints/deepfm.pt`.

## Pre-trained Model
- Local checkpoint: `checkpoints/deepfm.pt` (copy of latest trained model). Upload to your storage and replace this link: `<YOUR_DOWNLOAD_LINK_HERE>`.

## Acknowledgments
- Criteo display advertising dataset (Kaggle).
- DeepFM paper: Guo et al., 2017.
- Based on PyTorch ecosystem.

