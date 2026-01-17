@echo off
REM ============================================================
REM Soccer Film Analysis - Windows Deployment Script
REM ============================================================
REM This script fully deploys the application:
REM   1. Creates virtual environment
REM   2. Installs all dependencies
REM   3. Configures environment
REM   4. Initializes database
REM   5. Verifies installation
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo    SOCCER FILM ANALYSIS - DEPLOYMENT
echo ============================================================
echo.

REM ------------------------------------------------------------
REM STEP 1: Check Prerequisites
REM ------------------------------------------------------------
echo [STEP 1/7] Checking prerequisites...

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    goto :error
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo    Python: %PYVER%

REM Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Git not found. You won't be able to push to GitHub.
) else (
    for /f "tokens=3" %%i in ('git --version') do set GITVER=%%i
    echo    Git: %GITVER%
)

REM Check PostgreSQL
pg_isready -h localhost -p 5432 >nul 2>&1
if errorlevel 1 (
    echo [WARNING] PostgreSQL not responding on localhost:5432
    echo           Make sure PostgreSQL is running before using the app.
) else (
    echo    PostgreSQL: Running
)

echo    Prerequisites OK
echo.

REM ------------------------------------------------------------
REM STEP 2: Create Virtual Environment
REM ------------------------------------------------------------
echo [STEP 2/7] Creating virtual environment...

if exist venv (
    echo    Virtual environment exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        goto :error
    )
    echo    Created venv/
)
echo.

REM ------------------------------------------------------------
REM STEP 3: Activate and Upgrade pip
REM ------------------------------------------------------------
echo [STEP 3/7] Activating environment and upgrading pip...

call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo    pip upgraded
echo.

REM ------------------------------------------------------------
REM STEP 4: Install Dependencies
REM ------------------------------------------------------------
echo [STEP 4/7] Installing dependencies (this takes 3-5 minutes)...

echo    Installing core packages...
pip install loguru pydantic pydantic-settings python-dotenv rich click >nul 2>&1

echo    Installing computer vision packages...
pip install opencv-python numpy pillow scikit-learn >nul 2>&1

echo    Installing PyTorch...
pip install torch torchvision >nul 2>&1

echo    Installing Roboflow stack...
pip install supervision ultralytics inference roboflow >nul 2>&1

echo    Installing Roboflow Sports...
pip install git+https://github.com/roboflow/sports.git >nul 2>&1

echo    Installing database packages...
pip install psycopg2-binary sqlalchemy alembic >nul 2>&1

echo    Installing GUI packages...
pip install PyQt6 >nul 2>&1

echo    Installing visualization packages...
pip install pandas matplotlib seaborn plotly >nul 2>&1

echo    All dependencies installed
echo.

REM ------------------------------------------------------------
REM STEP 5: Create Directories
REM ------------------------------------------------------------
echo [STEP 5/7] Creating project directories...

if not exist data\videos mkdir data\videos
if not exist data\outputs mkdir data\outputs
if not exist data\models mkdir data\models
if not exist logs mkdir logs

echo    data/videos/
echo    data/outputs/
echo    data/models/
echo    logs/
echo.

REM ------------------------------------------------------------
REM STEP 6: Configure Environment
REM ------------------------------------------------------------
echo [STEP 6/7] Configuring environment...

if not exist .env (
    copy .env.example .env >nul
    echo    Created .env from template
    echo.
    echo    *** ACTION REQUIRED ***
    echo    Edit .env file with your settings:
    echo      - DB_PASSWORD=your_postgresql_password
    echo      - ROBOFLOW_API_KEY=your_api_key
    echo.
    set NEEDS_CONFIG=1
) else (
    echo    .env already exists
    set NEEDS_CONFIG=0
)
echo.

REM ------------------------------------------------------------
REM STEP 7: Verify Installation
REM ------------------------------------------------------------
echo [STEP 7/7] Verifying installation...

python -c "from config.settings import settings; print(f'    Config: OK')" 2>nul
if errorlevel 1 (
    echo [WARNING] Config verification failed
)

python -c "import cv2; print(f'    OpenCV: {cv2.__version__}')" 2>nul
python -c "import torch; print(f'    PyTorch: {torch.__version__}')" 2>nul
python -c "import supervision; print(f'    Supervision: {supervision.__version__}')" 2>nul
python -c "from PyQt6.QtWidgets import QApplication; print('    PyQt6: OK')" 2>nul
python -c "import sqlalchemy; print(f'    SQLAlchemy: {sqlalchemy.__version__}')" 2>nul

echo.
echo ============================================================
echo    DEPLOYMENT COMPLETE
echo ============================================================
echo.

if "%NEEDS_CONFIG%"=="1" (
    echo NEXT STEPS:
    echo.
    echo 1. Edit .env with your credentials:
    echo    notepad .env
    echo.
    echo 2. Initialize the database:
    echo    python scripts\setup_database.py
    echo.
    echo 3. Run the application:
    echo    python -m src.gui.main_window
) else (
    echo NEXT STEPS:
    echo.
    echo 1. Initialize the database:
    echo    python scripts\setup_database.py
    echo.
    echo 2. Run the application:
    echo    python -m src.gui.main_window
)
echo.
echo ============================================================
goto :end

:error
echo.
echo [DEPLOYMENT FAILED]
echo.
pause
exit /b 1

:end
pause
