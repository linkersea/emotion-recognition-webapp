# 🚀 GitHub + Railway 部署指南

## 📋 准备工作

1. **注册账号**：
   - [GitHub账号](https://github.com)
   - [Railway账号](https://railway.app)（使用GitHub登录）

2. **准备API密钥**（可选，不设置将使用离线回复）：
   - [硅基流动API](https://siliconflow.cn) - 免费额度充足
   - [DeepSeek API](https://platform.deepseek.com) - 备用选择

## 📤 第一步：上传到GitHub

### 1. 初始化Git仓库

```bash
# 进入项目目录
cd "f:\emotion recognize\emotion_recognize\emotion_recognize"

# 初始化Git
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: Emotion Recognition Web App"
```

### 2. 创建GitHub仓库

1. 登录 [GitHub](https://github.com)
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息：
   - Repository name: `emotion-recognition-webapp`
   - Description: `基于深度学习的情感识别Web应用`
   - 选择 "Public"（免费部署）
   - 不勾选任何初始化选项

### 3. 推送代码到GitHub

```bash
# 添加远程仓库（替换YOUR_USERNAME为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/emotion-recognition-webapp.git

# 推送代码
git branch -M main
git push -u origin main
```

## 🚂 第二步：部署到Railway

### 1. 连接GitHub

1. 访问 [Railway](https://railway.app)
2. 使用GitHub账号登录
3. 点击 "New Project"
4. 选择 "Deploy from GitHub repo"
5. 选择你刚才创建的仓库 `emotion-recognition-webapp`

### 2. 配置环境变量

在Railway项目设置中添加环境变量：

1. 点击项目 → "Variables" 标签
2. 添加以下变量：

```
PORT=8080
PYTHONPATH=/app:/app/src
SILICONFLOW_API_KEY=你的硅基流动API密钥（可选）
DEEPSEEK_API_KEY=你的DeepSeek API密钥（可选）
```

### 3. 等待部署完成

- Railway会自动检测到Python项目
- 安装依赖（requirements.txt）
- 启动应用（使用Procfile配置）
- 部署通常需要5-10分钟

### 4. 获取访问链接

部署成功后，Railway会提供一个公网访问链接，类似：
```
https://your-app-name.up.railway.app
```

## 🔧 常见问题解决

### 1. 模型文件过大
如果遇到模型文件太大无法上传的问题：

```bash
# 使用Git LFS管理大文件
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add models/saved_models/emotion_resnet_VGG16_best.pth
git commit -m "Add model file with LFS"
git push
```

### 2. 内存不足
如果Railway内存不足，可以：
- 升级到Pro计划（$5/月）
- 或者移除大模型文件，使用预训练模型

### 3. API密钥问题
- 确保在Railway环境变量中正确设置了API密钥
- 如果不设置API密钥，系统会使用离线回复模式

## 📝 更新代码

后续修改代码后，推送更新：

```bash
git add .
git commit -m "Update: 描述你的修改"
git push
```

Railway会自动重新部署。

## 🎉 完成！

部署完成后，你的情感识别应用就可以在全球访问了！

分享你的应用链接：`https://your-app-name.up.railway.app`
