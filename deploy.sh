#!/bin/bash
# ============================================================
# Soccer Film Analysis - Deployment Script (macOS/Linux)
# ============================================================
# This script fully deploys the application:
#   1. Creates virtual environment
#   2. Installs all dependencies
#   3. Configures environment
#   4. Initializes database
#   5. Verifies installation
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo "   SOCCER FILM ANALYSIS - DEPLOYMENT"
echo "============================================================"
echo ""

# ------------------------------------------------------------
# STEP 1: Check Prerequisites
# ------------------------------------------------------------
echo -e "${BLUE}[STEP 1/7]${NC} Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    echo -e "${RED}[ERROR]${NC} Python not found. Install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "   Python: $PYTHON_VERSION"

# Check Git
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version | cut -d' ' -f3)
    echo "   Git: $GIT_VERSION"
else
    echo -e "${YELLOW}[WARNING]${NC} Git not found. You won't be able to push to GitHub."
fi

# Check PostgreSQL
if command -v psql &> /dev/null; then
    if pg_isready -h localhost -p 5432 &> /dev/null; then
        echo "   PostgreSQL: Running"
    else
        echo -e "${YELLOW}[WARNING]${NC} PostgreSQL not responding on localhost:5432"
    fi
else
    echo -e "${YELLOW}[WARNING]${NC} PostgreSQL client not found"
fi

echo "   Prerequisites OK"
echo ""

# ------------------------------------------------------------
# STEP 2: Create Virtual Environment
# ------------------------------------------------------------
echo -e "${BLUE}[STEP 2/7]${NC} Creating virtual environment..."

if [ -d "venv" ]; then
    echo "   Virtual environment exists, skipping..."
else
    $PYTHON_CMD -m venv venv
    echo "   Created venv/"
fi
echo ""

# ------------------------------------------------------------
# STEP 3: Activate and Upgrade pip
# ------------------------------------------------------------
echo -e "${BLUE}[STEP 3/7]${NC} Activating environment and upgrading pip..."

source venv/bin/activate
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "   pip upgraded"
echo ""

# ------------------------------------------------------------
# STEP 4: Install Dependencies
# ------------------------------------------------------------
echo -e "${BLUE}[STEP 4/7]${NC} Installing dependencies (this takes 3-5 minutes)..."

echo "   Installing core packages..."
pip install loguru pydantic pydantic-settings python-dotenv rich click > /dev/null 2>&1

echo "   Installing computer vision packages..."
pip install opencv-python numpy pillow scikit-learn > /dev/null 2>&1

echo "   Installing PyTorch..."
pip install torch torchvision > /dev/null 2>&1

echo "   Installing Roboflow stack..."
pip install supervision ultralytics inference roboflow > /dev/null 2>&1

echo "   Installing Roboflow Sports..."
pip install git+https://github.com/roboflow/sports.git > /dev/null 2>&1

echo "   Installing database packages..."
pip install psycopg2-binary sqlalchemy alembic > /dev/null 2>&1

echo "   Installing GUI packages..."
pip install PyQt6 > /dev/null 2>&1

echo "   Installing visualization packages..."
pip install pandas matplotlib seaborn plotly > /dev/null 2>&1

echo -e "   ${GREEN}All dependencies installed${NC}"
echo ""

# ------------------------------------------------------------
# STEP 5: Create Directories
# ------------------------------------------------------------
echo -e "${BLUE}[STEP 5/7]${NC} Creating project directories..."

mkdir -p data/videos data/outputs data/models logs

echo "   data/videos/"
echo "   data/outputs/"
echo "   data/models/"
echo "   logs/"
echo ""

# ------------------------------------------------------------
# STEP 6: Configure Environment
# ------------------------------------------------------------
echo -e "${BLUE}[STEP 6/7]${NC} Configuring environment..."

NEEDS_CONFIG=0
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "   Created .env from template"
    echo ""
    echo -e "   ${YELLOW}*** ACTION REQUIRED ***${NC}"
    echo "   Edit .env file with your settings:"
    echo "     - DB_PASSWORD=your_postgresql_password"
    echo "     - ROBOFLOW_API_KEY=your_api_key"
    NEEDS_CONFIG=1
else
    echo "   .env already exists"
fi
echo ""

# ------------------------------------------------------------
# STEP 7: Verify Installation
# ------------------------------------------------------------
echo -e "${BLUE}[STEP 7/7]${NC} Verifying installation..."

python -c "from config.settings import settings; print('   Config: OK')" 2>/dev/null || echo -e "   ${YELLOW}Config: Needs configuration${NC}"
python -c "import cv2; print(f'   OpenCV: {cv2.__version__}')" 2>/dev/null || true
python -c "import torch; print(f'   PyTorch: {torch.__version__}')" 2>/dev/null || true
python -c "import supervision; print(f'   Supervision: {supervision.__version__}')" 2>/dev/null || true
python -c "from PyQt6.QtWidgets import QApplication; print('   PyQt6: OK')" 2>/dev/null || true
python -c "import sqlalchemy; print(f'   SQLAlchemy: {sqlalchemy.__version__}')" 2>/dev/null || true

echo ""
echo "============================================================"
echo -e "   ${GREEN}DEPLOYMENT COMPLETE${NC}"
echo "============================================================"
echo ""

if [ "$NEEDS_CONFIG" -eq 1 ]; then
    echo "NEXT STEPS:"
    echo ""
    echo "1. Edit .env with your credentials:"
    echo "   nano .env"
    echo ""
    echo "2. Initialize the database:"
    echo "   python scripts/setup_database.py"
    echo ""
    echo "3. Run the application:"
    echo "   python -m src.gui.main_window"
else
    echo "NEXT STEPS:"
    echo ""
    echo "1. Initialize the database:"
    echo "   python scripts/setup_database.py"
    echo ""
    echo "2. Run the application:"
    echo "   python -m src.gui.main_window"
fi
echo ""
echo "============================================================"
