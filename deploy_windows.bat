@echo off
REM Soccer Film Analysis - Windows Deployment Script
REM Run this script in the project root directory

echo ============================================
echo Soccer Film Analysis - Deployment Script
echo ============================================
echo.

REM Check Python version
echo [1/8] Checking Python version...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo [2/8] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo.
echo [3/8] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/8] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo.
echo [5/8] Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Install Roboflow Sports
echo.
echo [6/8] Installing Roboflow Sports...
pip install git+https://github.com/roboflow/sports.git

REM Create .env if not exists
echo.
echo [7/8] Checking configuration...
if not exist .env (
    echo Creating .env from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Edit .env file with your settings:
    echo   - DB_PASSWORD: Your PostgreSQL password
    echo   - ROBOFLOW_API_KEY: Get from https://app.roboflow.com/
    echo.
)

REM Create directories
echo.
echo [8/8] Creating data directories...
if not exist data\videos mkdir data\videos
if not exist data\outputs mkdir data\outputs
if not exist data\models mkdir data\models
if not exist logs mkdir logs

echo.
echo ============================================
echo Deployment Complete!
echo ============================================
echo.
echo Next steps:
echo 1. Edit .env with your PostgreSQL password and Roboflow API key
echo 2. Run: python scripts/setup_database.py
echo 3. Place a test video in data\videos\
echo 4. Run: python -m src.gui.main_window
echo.
echo Or for CLI analysis:
echo    python scripts/run_analysis.py data\videos\your_video.mp4 --depth quick
echo.
pause
