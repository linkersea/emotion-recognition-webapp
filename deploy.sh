#!/bin/bash

echo "========================================"
echo "情感识别应用 - GitHub + Railway 部署助手"
echo "========================================"
echo

echo "请确保你已经："
echo "1. 注册了GitHub账号"
echo "2. 注册了Railway账号（使用GitHub登录）"
echo "3. 准备好了API密钥（可选）"
echo

read -p "按Enter继续..."

echo "第一步：创建GitHub仓库"
echo "请在浏览器中完成以下操作："
echo "1. 访问 https://github.com/new"
echo "2. Repository name: emotion-recognition-webapp"
echo "3. Description: 基于深度学习的情感识别Web应用"
echo "4. 选择 Public"
echo "5. 不勾选任何初始化选项"
echo "6. 点击 Create repository"
echo

read -p "完成后按Enter继续..."

read -p "请输入你的GitHub用户名: " username

echo
echo "正在添加远程仓库..."
git remote add origin https://github.com/$username/emotion-recognition-webapp.git

echo
echo "正在推送代码到GitHub..."
git branch -M main
git push -u origin main

echo
echo "========================================"
echo "GitHub上传完成！"
echo "========================================"
echo

echo "第二步：部署到Railway"
echo "请在浏览器中完成以下操作："
echo "1. 访问 https://railway.app"
echo "2. 使用GitHub账号登录"
echo "3. 点击 New Project"
echo "4. 选择 Deploy from GitHub repo"
echo "5. 选择 emotion-recognition-webapp 仓库"
echo "6. 在Variables标签页添加环境变量："
echo "   - PORT=8080"
echo "   - PYTHONPATH=/app:/app/src"
echo "   - SILICONFLOW_API_KEY=你的API密钥（可选）"
echo

echo "部署完成后，Railway会提供访问链接！"
echo

read -p "按Enter退出..."
