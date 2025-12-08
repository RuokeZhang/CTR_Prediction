#!/bin/bash
# 准备上传到 Colab 的文件

set -e  # 遇到错误立即退出

echo "🚀 Preparing files for Google Colab upload..."
echo "=" * 60

# 1. 压缩预处理数据（包含 metadata）
echo "📦 Step 1: Compressing processed data..."
if [ -d "data/processed" ]; then
    tar -czf criteo_processed.tar.gz data/processed/
    echo "✅ Created: criteo_processed.tar.gz ($(ls -lh criteo_processed.tar.gz | awk '{print $5}'))"

    # 验证 manifest 文件被包含
    if tar -tzf criteo_processed.tar.gz | grep -q "split_manifest.json"; then
        echo "   ✓ split_manifest.json included"
    else
        echo "   ❌ WARNING: split_manifest.json NOT found in archive!"
        exit 1
    fi

    if tar -tzf criteo_processed.tar.gz | grep -q "numeric_scaler.joblib"; then
        echo "   ✓ numeric_scaler.joblib included"
    else
        echo "   ❌ WARNING: numeric_scaler.joblib NOT found in archive!"
        exit 1
    fi
else
    echo "❌ Error: data/processed directory not found!"
    echo "   Please run preprocessing first: python scripts/preprocess_criteo.py"
    exit 1
fi

echo ""

# 2. 压缩代码
echo "📦 Step 2: Compressing code..."
tar -czf ctr_code.tar.gz \
    src/ \
    scripts/ \
    setup.py \
    requirements.txt \
    COLAB_FIX.md \
    EMBEDDING_GUIDE.md \
    README.md

echo "✅ Created: ctr_code.tar.gz ($(ls -lh ctr_code.tar.gz | awk '{print $5}'))"

# 验证 setup.py 被包含
if tar -tzf ctr_code.tar.gz | grep -q "setup.py"; then
    echo "   ✓ setup.py included"
else
    echo "   ❌ WARNING: setup.py NOT found in archive!"
    exit 1
fi

echo ""

# 3. 列出要上传的文件
echo "=" * 60
echo "📋 Files ready for upload to Google Drive:"
echo "=" * 60
ls -lh criteo_processed.tar.gz ctr_code.tar.gz

echo ""
echo "📁 Please upload these files to:"
echo "   Google Drive → CTR-Prediction/"
echo ""
echo "   criteo_processed.tar.gz  → for data"
echo "   ctr_code.tar.gz          → for source code"
echo ""

# 4. 验证压缩包内容
echo "=" * 60
echo "🔍 Verifying archive contents..."
echo "=" * 60

echo ""
echo "📦 criteo_processed.tar.gz contains:"
tar -tzf criteo_processed.tar.gz | head -5
echo "   ... (total $(tar -tzf criteo_processed.tar.gz | wc -l | tr -d ' ') files)"

echo ""
echo "📦 ctr_code.tar.gz contains:"
tar -tzf ctr_code.tar.gz | grep -E "\.(py|txt|md)$" | head -10
echo "   ... (total $(tar -tzf ctr_code.tar.gz | wc -l | tr -d ' ') files)"

echo ""
echo "=" * 60
echo "✅ All files prepared successfully!"
echo "=" * 60
echo ""
echo "Next steps:"
echo "  1. Upload criteo_processed.tar.gz to Google Drive"
echo "  2. Upload ctr_code.tar.gz to Google Drive"
echo "  3. Open colab_training.ipynb in Google Colab"
echo "  4. Update file paths in the notebook to match your Drive structure"
echo "  5. Run the notebook!"
echo ""
