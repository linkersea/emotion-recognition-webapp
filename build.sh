#!/bin/bash

# Render构建脚本
echo "🔧 Building Emotion Recognition App..."

# 安装Python依赖
pip install -r requirements.txt

# 创建必要的目录
mkdir -p models/saved_models

echo "✅ Build completed successfully!"
