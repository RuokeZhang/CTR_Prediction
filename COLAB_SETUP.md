# 🚀 在 Google Colab 上训练模型指南

## 📦 方案 1: 上传预处理数据到 Google Drive（推荐）

### Step 1: 本地压缩数据

```bash
# 在项目根目录执行
cd /Users/ruoke/Documents/25FALL/DL/CTR-Prediction

# 压缩预处理的数据（~2.2GB → ~800MB）
tar -czf criteo_processed.tar.gz data/processed/

# 检查压缩文件大小
ls -lh criteo_processed.tar.gz
```

### Step 2: 上传到 Google Drive

1. 打开 [Google Drive](https://drive.google.com)
2. 创建文件夹 `CTR-Prediction`
3. 上传 `criteo_processed.tar.gz` 到这个文件夹
4. 记住文件路径（稍后在 Colab 中使用）

### Step 3: 在 Colab 中使用

创建新的 Colab Notebook，运行以下代码：

```python
# ===== Cell 1: 挂载 Google Drive =====
from google.colab import drive
drive.mount('/content/drive')

# ===== Cell 2: 解压数据 =====
!mkdir -p /content/CTR-Prediction
%cd /content/CTR-Prediction

# 解压（假设你上传到 Drive 根目录的 CTR-Prediction 文件夹）
!tar -xzf /content/drive/MyDrive/CTR-Prediction/criteo_processed.tar.gz

# 验证数据
!ls data/processed/ | head -10

# ===== Cell 3: 克隆代码（或上传代码压缩包）=====
# 方式 A: 如果有 Git 仓库
# !git clone https://github.com/YOUR_USERNAME/CTR-Prediction.git
# %cd CTR-Prediction

# 方式 B: 从 Drive 复制代码
!cp -r /content/drive/MyDrive/CTR-Prediction/src .
!cp -r /content/drive/MyDrive/CTR-Prediction/scripts .
!cp /content/drive/MyDrive/CTR-Prediction/requirements.txt .

# ===== Cell 4: 安装依赖 =====
!pip install -q numpy pandas pyarrow scikit-learn torch joblib

# ===== Cell 5: 检查 GPU =====
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ===== Cell 6: 训练模型 =====
# LR with embeddings
!python scripts/train_baselines.py \
    --model lr \
    --embedding-dim 8 \
    --epochs 5 \
    --device cuda \
    --batch-size 4096 \
    --log-file reports/colab_lr_emb8.csv

# MLP with embeddings
!python scripts/train_baselines.py \
    --model mlp \
    --embedding-dim 16 \
    --epochs 5 \
    --device cuda \
    --batch-size 4096 \
    --log-file reports/colab_mlp_emb16.csv

# ===== Cell 7: 查看结果 =====
!cat reports/colab_lr_emb8.csv
!cat reports/colab_mlp_emb16.csv

# ===== Cell 8: 保存结果回 Drive =====
!cp reports/*.csv /content/drive/MyDrive/CTR-Prediction/results/
```

---

## 📦 方案 2: 只上传部分数据（快速测试）

如果你只想快速测试，可以只上传少量数据：

### Step 1: 本地创建小数据集

```bash
# 只复制前 10 个 parquet 文件
mkdir -p data/processed_small/metadata
cp data/processed/criteo_chunk_00{00..09}.parquet data/processed_small/
cp data/processed/metadata/* data/processed_small/metadata/

# 修改 split_manifest.json（只保留这10个文件）
python3 << 'EOF'
import json
from pathlib import Path

manifest_path = Path("data/processed_small/metadata/split_manifest.json")
with open("data/processed/metadata/split_manifest.json") as f:
    manifest = json.load(f)

# 只保留前10个文件
small_manifest = {
    "train": [f for f in manifest["train"] if "000" in f or "001" in f][:8],
    "val": [f for f in manifest["val"] if "000" in f][:1],
    "test": [f for f in manifest["test"] if "000" in f][:1],
}

with open(manifest_path, "w") as f:
    json.dump(small_manifest, f, indent=2)

print("Small manifest created:")
print(json.dumps(small_manifest, indent=2))
EOF

# 压缩小数据集（~200MB）
tar -czf criteo_small.tar.gz data/processed_small/
ls -lh criteo_small.tar.gz
```

### Step 2: 在 Colab 中使用小数据集

```python
# 解压
!tar -xzf /content/drive/MyDrive/CTR-Prediction/criteo_small.tar.gz

# 训练（使用小数据集路径）
!python scripts/train_baselines.py \
    --model mlp \
    --embedding-dim 8 \
    --epochs 3 \
    --device cuda \
    --manifest data/processed_small/metadata/split_manifest.json
```

---

## 📦 方案 3: 在 Colab 上完整预处理（不推荐）

如果不想上传数据，可以在 Colab 上从头预处理：

```python
# ===== Cell 1: 下载 Criteo 原始数据 =====
# 需要从 Kaggle 下载 train.txt（10GB），这会很慢
# 或者从你的 Drive 上传 train.txt

# ===== Cell 2: 运行预处理 =====
!python scripts/preprocess_criteo.py \
    --chunk-size 400000 \
    --build-manifest

# ===== Cell 3: 训练 =====
!python scripts/train_baselines.py --model mlp --device cuda
```

**警告**: Colab 有时间限制（12小时），预处理可能会超时！

---

## 🎯 完整工作流（推荐）

### 本地操作：

```bash
# 1. 压缩数据
cd /Users/ruoke/Documents/25FALL/DL/CTR-Prediction
tar -czf criteo_processed.tar.gz data/processed/

# 2. 压缩代码
tar -czf ctr_code.tar.gz src/ scripts/ requirements.txt setup.py

# 3. 上传两个文件到 Google Drive
# - criteo_processed.tar.gz (~800MB)
# - ctr_code.tar.gz (~50KB)
```

### Colab Notebook 完整代码：

