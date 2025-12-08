## preprocess
scripts/preprocess_criteo.py  
src/data/pipeline.py  
- 分块存储，每块 400,000 行。fit_numeric_scaler扫一遍每一块，扫的时候对每一块做缺失值处理。扫完之后，整体数据的mean和variance被保存在scaler中（增量学习），再写入磁盘。
- 接着再遍历每一块，用整体的scaler去处理每一块，同时做类别特征哈希 (262k桶)，然后把处理后的每一块存到data/processed中，以parquet格式存储
- 把已经缓存好的多个 parquet chunk 文件，先打乱顺序，然后按比例分配到 train / val / test 三个集合里，这个划分信息存储到data/processed/metadata/split_manifest.json
## baseline
### logistic regression
### Shallow MLP


