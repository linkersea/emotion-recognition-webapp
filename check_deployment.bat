@echo off
echo ========================================
echo 部署前检查 - 确保所有文件就绪
echo ========================================
echo.

echo 检查必要文件...
echo.

if exist "Procfile" (
    echo ✅ Procfile - Railway部署配置
) else (
    echo ❌ 缺少 Procfile
)

if exist "requirements.txt" (
    echo ✅ requirements.txt - Python依赖
) else (
    echo ❌ 缺少 requirements.txt
)

if exist "runtime.txt" (
    echo ✅ runtime.txt - Python版本
) else (
    echo ❌ 缺少 runtime.txt
)

if exist "railway.toml" (
    echo ✅ railway.toml - Railway配置
) else (
    echo ❌ 缺少 railway.toml
)

if exist ".gitignore" (
    echo ✅ .gitignore - Git忽略文件
) else (
    echo ❌ 缺少 .gitignore
)

if exist ".gitattributes" (
    echo ✅ .gitattributes - Git LFS配置
) else (
    echo ❌ 缺少 .gitattributes
)

if exist "models\saved_models\emotion_resnet_VGG16_best.pth" (
    echo ✅ 模型文件存在
) else (
    echo ❌ 缺少模型文件
)

if exist "web\backend\app.py" (
    echo ✅ 后端应用
) else (
    echo ❌ 缺少后端应用
)

if exist "web\frontend\index.html" (
    echo ✅ 前端页面
) else (
    echo ❌ 缺少前端页面
)

echo.
echo 检查Git LFS状态...
git lfs ls-files

echo.
echo 检查Git状态...
git status --short

echo.
echo ========================================
echo 检查完成！如果所有项目都显示 ✅，
echo 那么你的项目已经准备好部署了！
echo ========================================
echo.
pause
