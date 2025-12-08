# DeepCTR Project Plan

## Overview / 概览
Document the full workflow, covering data preparation, baseline experiments, DeepFM development, and evaluation strategies on the Criteo dataset.  
在本文件中记录完整工作流，涵盖数据准备、基线实验、DeepFM 开发与评估策略。

## Steps / 步骤
1. **Data Ingestion & Handling / 数据导入与管理**  
   - Download the 5GB Criteo dataset and store it under `data/raw/`. / 下载 5GB Criteo 数据集并放到 `data/raw/`。  
   - Implement streaming CSV readers with chunking + dtype downcasting to avoid RAM spikes. / 使用分块和类型降级的流式 CSV 读取避免内存暴涨。  
   - Cache intermediate parquet files in `data/processed/` for faster reloads. / 在 `data/processed/` 缓存中间 Parquet 文件以加速复用。

2. **Feature Engineering Pipeline / 特征工程流程**  
   - Build a preprocessing module (`src/data/pipeline.py`) covering missing-value fills, categorical hashing/label encoding, numerical normalization, and train/val/test splits with consistent random seeds. / 在 `src/data/pipeline.py` 中构建预处理模块，包含缺失值填充、类别哈希/编码、数值归一化及稳定随机种子的训练/验证/测试划分。  
   - Persist metadata (encoders, scalers) via joblib for reuse across models. / 用 joblib 持久化编码器和归一化器以便模型复用。

3. **Baselines (LR & Shallow MLP) / 基线模型（LR 与浅层 MLP）**  
   - Implement logistic regression and a lightweight MLP inside `src/models/baselines.py`, reusing the preprocessing artifacts. / 在 `src/models/baselines.py` 中实现逻辑回归和轻量 MLP，并复用预处理产物。  
   - Provide `scripts/train_baselines.py` with CLI args for batch size, epochs, and evaluation metrics (AUC, logloss). / 编写 `scripts/train_baselines.py` 支持批大小、轮数、AUC/Logloss 等 CLI 参数。  
   - Log metrics to TensorBoard or CSV for comparison. / 将指标记录到 TensorBoard 或 CSV 便于对比。

4. **DeepFM Model Development / DeepFM 模型开发**  
   - Define embedding layers, FM interaction components, and MLP tower in `src/models/deepfm.py`. / 在 `src/models/deepfm.py` 中定义嵌入层、FM 交互和 MLP 塔。  
   - Write `scripts/train_deepfm.py` to load processed data, configure field-wise embeddings, and support mixed-precision training. / 编写 `scripts/train_deepfm.py`，加载处理后数据，按字段配置嵌入并支持混合精度。  
   - Add checkpointing and inference hooks for production-like scoring. / 增加检查点与推理接口以模拟生产打分。

5. **Evaluation & Reporting / 评估与报告**  
   - Centralize evaluation logic in `src/eval/metrics.py` to compute AUC/logloss and plot calibration curves. / 在 `src/eval/metrics.py` 集中评估逻辑，计算 AUC/Logloss 并绘制校准曲线。  
   - Create `notebooks/model_report.ipynb` to summarize results, compare baselines vs DeepFM, and analyze feature importances. / 创建 `notebooks/model_report.ipynb` 汇总结果、比较基线与 DeepFM，并分析特征重要性。  
   - Document findings and deployment considerations in `reports/DeepCTR_Report.md`. / 在 `reports/DeepCTR_Report.md` 记录结论与部署建议。

## Todos
- todo-data: Implement chunked data ingestion + caching pipeline.  
- todo-baselines: Build LR and MLP baselines with shared preprocessing.  
- todo-deepfm: Implement and train the DeepFM architecture.  
- todo-report: Consolidate evaluation artifacts and write the final report.

