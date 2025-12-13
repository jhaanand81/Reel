@echo off
REM ============================================
REM ReelSense AI - Push to Live Environment
REM One-click deployment script
REM ============================================

echo.
echo ========================================
echo   ReelSense AI - Deploy to Live
echo ========================================
echo.

set PYTHON_PATH=C:\Users\Anand Jha\AppData\Local\Programs\Python\Python311\python.exe
set SCRIPT_PATH=%~dp0deploy.py

REM Check Python
if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Ask for commit message
set /p MESSAGE="Enter deployment message (or press Enter for default): "

if "%MESSAGE%"=="" (
    set MESSAGE=Quick deploy via push_to_live.bat
)

echo.
echo Deploying with message: %MESSAGE%
echo.

REM Run deployment
"%PYTHON_PATH%" "%SCRIPT_PATH%" --deploy --message "%MESSAGE%"

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo   DEPLOYMENT SUCCESSFUL!
    echo ========================================
) else (
    echo ========================================
    echo   DEPLOYMENT FAILED!
    echo   Check deploy.log for details
    echo ========================================
)

echo.
pause
