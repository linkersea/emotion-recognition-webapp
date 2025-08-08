@echo off
chcp 65001 >nul
echo ========================================
echo Pre-deployment Check - Ensure all files ready
echo ========================================
echo.

echo Checking required files...
echo.

if exist "Procfile" (
    echo [OK] Procfile - Railway deployment config
) else (
    echo [MISSING] Procfile
)

if exist "requirements.txt" (
    echo [OK] requirements.txt - Python dependencies
) else (
    echo [MISSING] requirements.txt
)

if exist "runtime.txt" (
    echo [OK] runtime.txt - Python version
) else (
    echo [MISSING] runtime.txt
)

if exist "railway.toml" (
    echo [OK] railway.toml - Railway config
) else (
    echo [MISSING] railway.toml
)

if exist ".gitignore" (
    echo [OK] .gitignore - Git ignore file
) else (
    echo [MISSING] .gitignore
)

if exist ".gitattributes" (
    echo [OK] .gitattributes - Git LFS config
) else (
    echo [MISSING] .gitattributes
)

if exist "models\saved_models\emotion_resnet_VGG16_best.pth" (
    echo [OK] Model file exists
) else (
    echo [MISSING] Model file
)

if exist "web\backend\app.py" (
    echo [OK] Backend application
) else (
    echo [MISSING] Backend application
)

if exist "web\frontend\index.html" (
    echo [OK] Frontend page
) else (
    echo [MISSING] Frontend page
)

echo.
echo Checking Git LFS status...
git lfs ls-files

echo.
echo Checking Git status...
git status --short

echo.
echo ========================================
echo Check completed! If all items show [OK],
echo your project is ready for deployment!
echo ========================================
echo.
pause
