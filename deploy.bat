@echo off
chcp 65001 >nul
echo ========================================
echo Emotion Recognition App - GitHub + Railway Deploy Helper
echo ========================================
echo.

echo Please make sure you have:
echo 1. Registered GitHub account
echo 2. Registered Railway account (login with GitHub)
echo 3. Prepared API keys (optional)
echo.

pause

echo Step 1: Create GitHub Repository
echo Please complete the following in your browser:
echo 1. Visit https://github.com/new
echo 2. Repository name: emotion-recognition-webapp
echo 3. Description: AI-powered emotion recognition web app
echo 4. Select Public
echo 5. Do not check any initialization options
echo 6. Click Create repository
echo.

pause

set /p username=Please enter your GitHub username: 

echo.
echo Adding remote repository...
git remote add origin https://github.com/%username%/emotion-recognition-webapp.git

echo.
echo Pushing code to GitHub...
git branch -M main
git push -u origin main

echo.
echo ========================================
echo GitHub upload completed!
echo ========================================
echo.

echo Step 2: Deploy to Railway
echo Please complete the following in your browser:
echo 1. Visit https://railway.app
echo 2. Login with GitHub account
echo 3. Click New Project
echo 4. Select Deploy from GitHub repo
echo 5. Select emotion-recognition-webapp repository
echo 6. Add environment variables in Variables tab:
echo    - PORT=8080
echo    - PYTHONPATH=/app:/app/src
echo    - SILICONFLOW_API_KEY=your_api_key (optional)
echo.

echo After deployment, Railway will provide access link!
echo.

pause
