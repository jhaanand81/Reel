@echo on
title DEBUG - Reel Sense AI

echo ========================================
echo DEBUG MODE - All commands visible
echo ========================================
echo.

cd /d "%~dp0"
echo Current directory: %CD%
echo.

echo Checking for Python...
where python
echo.

echo Checking backend folder...
dir backend\*.py
echo.

echo Checking .env file...
if exist "backend\.env" (
    echo .env file EXISTS
) else (
    echo .env file MISSING!
)
echo.

echo Trying to run Python...
python --version
echo.

echo Trying to import flask...
python -c "import flask; print('Flask OK')"
echo.

echo Starting server...
cd backend
python main.py

echo.
echo ========================================
echo Server exited. Check errors above.
echo ========================================
pause
