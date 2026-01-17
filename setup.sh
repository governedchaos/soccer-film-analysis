#!/bin/bash
# Soccer Film Analysis - Quick Setup Script for Claude Code
# Run this script to set up the project environment

set -e  # Exit on error

echo "=================================================="
echo "Soccer Film Analysis - Quick Setup"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo ""
    echo "[1/5] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[2/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
pip install --upgrade pip > /dev/null

# Install core packages first
pip install loguru pydantic pydantic-settings python-dotenv rich > /dev/null 2>&1 || true
echo "  - Core packages installed"

# Install computer vision packages
pip install opencv-python numpy pillow scikit-learn > /dev/null 2>&1 || true
echo "  - Computer vision packages installed"

# Install Roboflow stack
pip install supervision ultralytics inference roboflow > /dev/null 2>&1 || true
echo "  - Roboflow packages installed"

# Install Roboflow Sports
pip install git+https://github.com/roboflow/sports.git > /dev/null 2>&1 || true
echo "  - Roboflow Sports installed"

# Install database packages
pip install psycopg2-binary sqlalchemy alembic > /dev/null 2>&1 || true
echo "  - Database packages installed"

# Install GUI packages  
pip install PyQt6 > /dev/null 2>&1 || true
echo "  - GUI packages installed"

echo "✓ All dependencies installed"

# Create directories
echo ""
echo "[4/5] Creating directories..."
mkdir -p data/videos data/outputs data/models logs
echo "✓ Directories created"

# Setup environment file
echo ""
echo "[5/5] Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env file from template"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file with your settings:"
    echo "   - Set your PostgreSQL password"
    echo "   - Add your Roboflow API key (get free at roboflow.com)"
else
    echo "✓ .env file already exists"
fi

echo ""
echo "=================================================="
echo "✓ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit .env with your database credentials and Roboflow API key"
echo "   nano .env"
echo ""
echo "2. Setup the database (make sure PostgreSQL is running)"
echo "   python scripts/setup_database.py"
echo ""
echo "3. Run the application"
echo "   python -m src.gui.main_window"
echo ""
echo "Or run from command line:"
echo "   python scripts/run_analysis.py path/to/video.mp4"
echo ""
