# PowerShell部署脚本 - 支持中文显示
# 设置控制台编码为UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "情感识别应用 - GitHub + Railway 部署助手" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "请确保你已经：" -ForegroundColor Yellow
Write-Host "1. 注册了GitHub账号"
Write-Host "2. 注册了Railway账号（使用GitHub登录）"
Write-Host "3. 准备好了API密钥（可选）"
Write-Host ""

Read-Host "按Enter继续"

Write-Host "第一步：创建GitHub仓库" -ForegroundColor Green
Write-Host "请在浏览器中完成以下操作："
Write-Host "1. 访问 https://github.com/new"
Write-Host "2. Repository name: emotion-recognition-webapp"
Write-Host "3. Description: 基于深度学习的情感识别Web应用"
Write-Host "4. 选择 Public"
Write-Host "5. 不勾选任何初始化选项"
Write-Host "6. 点击 Create repository"
Write-Host ""

Read-Host "完成后按Enter继续"

$username = Read-Host "请输入你的GitHub用户名"

Write-Host ""
Write-Host "正在添加远程仓库..." -ForegroundColor Yellow
git remote add origin "https://github.com/$username/emotion-recognition-webapp.git"

Write-Host ""
Write-Host "正在推送代码到GitHub..." -ForegroundColor Yellow
git branch -M main
git push -u origin main

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "GitHub上传完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "第二步：部署到Railway" -ForegroundColor Green
Write-Host "请在浏览器中完成以下操作："
Write-Host "1. 访问 https://railway.app"
Write-Host "2. 使用GitHub账号登录"
Write-Host "3. 点击 New Project"
Write-Host "4. 选择 Deploy from GitHub repo"
Write-Host "5. 选择 emotion-recognition-webapp 仓库"
Write-Host "6. 在Variables标签页添加环境变量："
Write-Host "   - PORT=8080"
Write-Host "   - PYTHONPATH=/app:/app/src"
Write-Host "   - SILICONFLOW_API_KEY=你的API密钥（可选）"
Write-Host ""

Write-Host "部署完成后，Railway会提供访问链接！" -ForegroundColor Green
Write-Host ""

Read-Host "按Enter退出"
