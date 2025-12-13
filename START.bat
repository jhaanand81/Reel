@echo off
title Reel Sense AI
color 0A

echo.
echo  ============================================
echo       REEL SENSE AI - Video Ad Generator
echo  ============================================
echo.

cd /d "%~dp0"

echo  [OK] Working directory set
echo  [..] Checking Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found!
    echo  Install from: https://python.org/downloads
    start https://www.python.org/downloads/
    pause
    exit /b
)

for /f "tokens=*" %%i in ('python --version') do echo  [OK] %%i

echo  [..] Checking backend files...
if not exist "backend\main.py" (
    echo  [ERROR] backend\main.py not found!
    pause
    exit /b
)
echo  [OK] Backend files found

echo  [..] Checking API keys...
if not exist "backend\.env" (
    echo  [ERROR] backend\.env not found!
    pause
    exit /b
)
echo  [OK] API keys configured

echo.
echo  ============================================
echo   STARTING SERVER
echo  ============================================
echo.
echo   Login: any email + code REELSENSE2025
echo.
echo   Keep this window open!
echo   Press Ctrl+C to stop.
echo  ============================================
echo.

start "" http://localhost:5000/auth.html

cd backend
python main.py

echo.
echo  Server stopped.
pause
