@echo off
REM ReelSense AI - Daily Backup Script
REM Run this via Windows Task Scheduler for automated daily backups

echo ========================================
echo ReelSense AI Backup - %date% %time%
echo ========================================

REM Set the paths
set PYTHON_PATH=C:\Users\Anand Jha\AppData\Local\Programs\Python\Python311\python.exe
set SCRIPT_PATH=%~dp0backend\backup_manager.py

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found at %PYTHON_PATH%
    echo Please update PYTHON_PATH in this script
    pause
    exit /b 1
)

REM Check if backup script exists
if not exist "%SCRIPT_PATH%" (
    echo ERROR: Backup script not found at %SCRIPT_PATH%
    pause
    exit /b 1
)

REM Run the backup
echo Running backup...
"%PYTHON_PATH%" "%SCRIPT_PATH%" --backup

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Backup completed successfully!
) else (
    echo.
    echo ERROR: Backup failed with error code %ERRORLEVEL%
)

echo ========================================
echo Backup process finished
echo ========================================
