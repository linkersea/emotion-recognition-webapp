#!/bin/bash

# Render部署启动脚本
echo "🚀 Starting Emotion Recognition App on Render..."

# 设置环境变量
export PYTHONPATH="/opt/render/project/src:/opt/render/project"

# 启动Flask应用
cd /opt/render/project
python web/backend/app.py
