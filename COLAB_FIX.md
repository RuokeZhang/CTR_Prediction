# 🔧 Colab ModuleNotFoundError 修复指南

## ❌ 错误信息
```
ModuleNotFoundError: No module named 'src'
```

## ✅ 解决方案

在 Colab Notebook 中，在 **Step 4 (Extract Code)** 之后，添加一个新的 Cell：

### Step 4.5: 安装项目为 Python 包

```python
# 确保在项目根目录
%cd /content/CTR-Prediction

# 安装项目为可编辑包（这样 src 模块就可以被导入了）
!pip install -e .

# 验证安装
print("\\n✅ Verifying installation...")
import sys
print(f"Python path: {sys.path[:3]}")

# 测试导入
try:
    from src.data.module import CriteoBatchIterator
    from src.models.baselines import LogisticRegressionModel, ShallowMLP
    from src.eval.metrics import compute_auc, compute_logloss
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

## 📝 完整的 Colab 工作流

```python
# ===== Cell 1: Mount Drive =====
from google.colab import drive
drive.mount('/content/drive')

# ===== Cell 2: Setup Environment =====
!mkdir -p /content/CTR-Prediction
%cd /content/CTR-Prediction

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# ===== Cell 3: Extract Data =====
DATA_PATH = "/content/drive/MyDrive/CTR-Prediction/criteo_processed.tar.gz"
!tar -xzf {DATA_PATH}
!ls data/processed/*.parquet | wc -l

# ===== Cell 4: Extract Code =====
CODE_PATH = "/content/drive/MyDrive/CTR-Prediction/ctr_code.tar.gz"
!tar -xzf {CODE_PATH}
!ls src/ scripts/

# ===== Cell 4.5: Install Project Package (重要！) =====
!pip install -e .

# 验证导入
from src.data.module import CriteoBatchIterator
from src.models.baselines import LogisticRegressionModel
print("✅ Modules loaded!")

# ===== Cell 5: Train Model =====
!python scripts/train_baselines.py \
    --model mlp \
    --embedding-dim 8 \
    --epochs 5 \
    --device cuda \
    --batch-size 4096 \
    --log-file reports/mlp_emb8.csv
```

## 🚨 如果 `pip install -e .` 也失败

### 方案 A: 手动设置 PYTHONPATH

```python
import sys
sys.path.insert(0, '/content/CTR-Prediction')
print(sys.path[:3])
```

### 方案 B: 修改训练脚本（不推荐）

在 `scripts/train_baselines.py` 的开头添加：

```python
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

但这需要修改代码，不够优雅。

## 📦 准备上传文件

### 本地操作：

```bash
cd /Users/ruoke/Documents/25FALL/DL/CTR-Prediction

# 1. 压缩数据
tar -czf criteo_processed.tar.gz data/processed/
ls -lh criteo_processed.tar.gz  # 应该是 ~800MB

# 2. 压缩代码（确保包含 setup.py）
tar -czf ctr_code.tar.gz src/ scripts/ setup.py requirements.txt
ls -lh ctr_code.tar.gz  # 应该是 ~10KB

# 3. 上传这两个文件到 Google Drive 的 CTR-Prediction 文件夹
```

## 🎯 最终 Colab Notebook 流程

保存以下代码为完整的 Notebook：

```python
# ========== CELL 1: 挂载 Drive ==========
from google.colab import drive
drive.mount('/content/drive')

# ========== CELL 2: 环境检查 ==========
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU! Go to Runtime → Change runtime type → GPU")

# ========== CELL 3: 创建工作目录 ==========
!mkdir -p /content/CTR-Prediction
%cd /content/CTR-Prediction

# ========== CELL 4: 解压数据 ==========
DATA_ARCHIVE = "/content/drive/MyDrive/CTR-Prediction/criteo_processed.tar.gz"
print("📦 Extracting data...")
!tar -xzf {DATA_ARCHIVE}

# 验证
parquet_count = !ls data/processed/*.parquet | wc -l
print(f"✅ Found {parquet_count[0]} parquet files")
!ls data/processed/metadata/

# ========== CELL 5: 解压代码 ==========
CODE_ARCHIVE = "/content/drive/MyDrive/CTR-Prediction/ctr_code.tar.gz"
print("📦 Extracting code...")
!tar -xzf {CODE_ARCHIVE}

# 验证
!ls -la
print("\\n📂 Source files:")
!ls src/
print("\\n📂 Scripts:")
!ls scripts/

# ========== CELL 6: 安装项目包 (关键!) ==========
print("📦 Installing project package...")
!pip install -e .

# 验证导入
print("\\n🔍 Testing imports...")
from src.data.module import CriteoBatchIterator
from src.models.baselines import LogisticRegressionModel, ShallowMLP
from src.eval.metrics import compute_auc
print("✅ All modules imported successfully!")

# ========== CELL 7: 训练 LR (no embeddings) ==========
!python scripts/train_baselines.py \
    --model lr \
    --no-embeddings \
    --epochs 5 \
    --device cuda \
    --batch-size 4096 \
    --log-file reports/lr_no_emb.csv

# ========== CELL 8: 训练 LR (with embeddings) ==========
!python scripts/train_baselines.py \
    --model lr \
    --embedding-dim 8 \
    --epochs 5 \
    --device cuda \
    --batch-size 4096 \
    --log-file reports/lr_emb8.csv

# ========== CELL 9: 训练 MLP (emb_dim=8) ==========
!python scripts/train_baselines.py \
    --model mlp \
    --embedding-dim 8 \
    --epochs 5 \
    --device cuda \
    --batch-size 4096 \
    --dropout 0.2 \
    --log-file reports/mlp_emb8.csv

# ========== CELL 10: 训练 MLP (emb_dim=16) ==========
!python scripts/train_baselines.py \
    --model mlp \
    --embedding-dim 16 \
    --epochs 5 \
    --device cuda \
    --batch-size 4096 \
    --dropout 0.2 \
    --log-file reports/mlp_emb16.csv

# ========== CELL 11: 查看结果 ==========
import pandas as pd

for model_name in ["lr_no_emb", "lr_emb8", "mlp_emb8", "mlp_emb16"]:
    csv_path = f"reports/{model_name}.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"\\n{'='*60}")
        print(f"📊 {model_name.upper()}")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print(f"\\nBest Val AUC: {df['val_auc'].max():.6f}")
    except FileNotFoundError:
        print(f"⚠️  {csv_path} not found")

# ========== CELL 12: 保存结果到 Drive ==========
!mkdir -p /content/drive/MyDrive/CTR-Prediction/results
!cp reports/*.csv /content/drive/MyDrive/CTR-Prediction/results/
print("\\n✅ Results saved to Drive!")
!ls -lh /content/drive/MyDrive/CTR-Prediction/results/
```

## ✅ 检查清单

在运行之前确认：

- [ ] 已上传 `criteo_processed.tar.gz` 到 Drive
- [ ] 已上传 `ctr_code.tar.gz` 到 Drive（包含 `setup.py`）
- [ ] Colab 运行时类型设置为 GPU
- [ ] Drive 路径正确（`/content/drive/MyDrive/CTR-Prediction/...`）
- [ ] 已运行 `!pip install -e .` 命令

## 🎉 成功的标志

如果看到以下输出，说明修复成功：

```
✅ All modules imported successfully!
Model: LR | Embeddings: Enabled | Embedding Dim: 8 | Parameters: 54,558,942
```
