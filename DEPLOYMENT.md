# Soccer Film Analysis - Deployment Guide

Complete guide for deploying the soccer film analysis application. This application runs entirely offline using local YOLOv8 models - **no API keys required**.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Full Installation](#full-installation)
4. [PostgreSQL Setup](#postgresql-setup)
5. [Configuration](#configuration)
6. [Running the Application](#running-the-application)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- **Python 3.10+**
- **Git**

### Optional (for database features)
- **PostgreSQL 14+** - For persistent game/player data

### Verify Prerequisites

```bash
# Check Python version (need 3.10+)
python --version

# Check Git
git --version

# Check PostgreSQL (optional)
pg_isready
```

---

## Quick Start

For the fastest setup without database features:

```bash
# Clone repository
git clone https://github.com/governedchaos/soccer-film-analysis.git
cd soccer-film-analysis

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m src.gui.main_window
```

The first run will automatically download YOLOv8 models (~25MB).

---

## Full Installation

### 1. Clone Repository

```bash
git clone https://github.com/governedchaos/soccer-film-analysis.git
cd soccer-film-analysis
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install all requirements
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "
import cv2
print(f'OpenCV: {cv2.__version__}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import supervision
print(f'Supervision: {supervision.__version__}')

from PyQt6.QtWidgets import QApplication
print('PyQt6: OK')

from ultralytics import YOLO
print('Ultralytics: OK')

print()
print('All core packages installed successfully!')
"
```

---

## PostgreSQL Setup

PostgreSQL is **optional** but recommended for:
- Saving game analysis across sessions
- Building a team/player database
- Tracking player statistics over time

### 1. Install PostgreSQL

**Windows**: Download from https://www.postgresql.org/download/windows/

**macOS**:
```bash
brew install postgresql@14
brew services start postgresql@14
```

**Linux**:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### 2. Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE soccer_analysis;

# Verify
\l

# Exit
\q
```

### 3. Configure Connection

Create `.env` file in project root:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=soccer_analysis
DB_USER=postgres
DB_PASSWORD=your_password
```

### 4. Initialize Tables

```bash
python scripts/setup_database.py
```

---

## Configuration

### Environment Variables

All settings can be configured via `.env` file:

| Setting | Description | Default |
|---------|-------------|---------|
| `DB_HOST` | PostgreSQL host | localhost |
| `DB_PORT` | PostgreSQL port | 5432 |
| `DB_NAME` | Database name | soccer_analysis |
| `DB_USER` | Database user | postgres |
| `DB_PASSWORD` | Database password | (empty) |
| `YOLO_MODEL_SIZE` | Model: nano/small/medium/large/xlarge | small |
| `ENABLE_GPU` | Use GPU acceleration | true |
| `LOG_LEVEL` | DEBUG/INFO/WARNING/ERROR | INFO |
| `PLAYER_CONFIDENCE_THRESHOLD` | Player detection threshold | 0.3 |
| `BALL_CONFIDENCE_THRESHOLD` | Ball detection threshold | 0.15 |

### Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| nano | ~3MB | Fastest | Basic |
| small | ~25MB | Fast | Good |
| medium | ~50MB | Medium | Better |
| large | ~85MB | Slow | High |
| xlarge | ~130MB | Slowest | Best |

Models are downloaded automatically on first use.

---

## Running the Application

### GUI Mode (Recommended)

```bash
# Using Python module
python -m src.gui.main_window

# Or using run script (Windows)
run.bat gui

# Or using run script (Linux/Mac)
./run.sh gui
```

### Command Line Mode

```bash
# Quick analysis (fastest)
python scripts/run_analysis.py video.mp4 --depth quick

# Standard analysis
python scripts/run_analysis.py video.mp4 --depth standard

# Deep analysis (most detailed)
python scripts/run_analysis.py video.mp4 --depth deep
```

### Test Mode

```bash
# Quick system test
run.bat test  # Windows
./run.sh test # Linux/Mac
```

---

## Using the Application

### 1. Load Video
- Click "Open Video" or drag and drop
- Supported formats: MP4, AVI, MOV, MKV

### 2. Set Team Colors
- Use the color picker to set home/away jersey colors
- Or let auto-detection work (processes first few seconds)

### 3. Start Analysis
- Select analysis depth (Quick/Standard/Deep)
- Click "Start Analysis"
- Watch real-time detection and statistics

### 4. Review Results
- View detection overlays on video
- Check stats panel for possession, player counts
- Export reports as PDF

---

## Troubleshooting

### Application Won't Start

```bash
# Check Python version
python --version  # Need 3.10+

# Verify virtual environment is active
which python  # Should show venv path

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### No GPU Detected

```bash
# Check CUDA (NVIDIA)
python -c "import torch; print(torch.cuda.is_available())"

# Check MPS (Apple Silicon)
python -c "import torch; print(torch.backends.mps.is_available())"

# Force CPU mode in .env
ENABLE_GPU=false
```

### Database Connection Failed

```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Verify credentials
psql -h localhost -U postgres -d soccer_analysis

# Check .env file
cat .env | grep DB_
```

### PyQt6 Display Issues (Linux)

```bash
# Install system dependencies
sudo apt-get install libxcb-xinerama0 libxcb-cursor0

# Try platform override
QT_QPA_PLATFORM=xcb python -m src.gui.main_window
```

### Video Won't Load

```bash
# Check OpenCV can read the video
python -c "
import cv2
cap = cv2.VideoCapture('path/to/video.mp4')
print(f'Opened: {cap.isOpened()}')
print(f'Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
cap.release()
"

# Try different video codec
# Convert video to H.264 MP4 if needed
```

### Detection Quality Issues

1. **Ball not detected**: Lower `BALL_CONFIDENCE_THRESHOLD` to 0.1
2. **Too many false positives**: Raise `PLAYER_CONFIDENCE_THRESHOLD` to 0.4
3. **Players switching teams**: Set team colors manually
4. **Referees counted as players**: They should auto-filter now

---

## Project Structure

```
soccer_film_analysis/
├── config/
│   └── settings.py              # Configuration management
├── src/
│   ├── core/
│   │   └── video_processor.py   # Main processing pipeline
│   ├── detection/
│   │   ├── detector.py          # Base YOLO detection
│   │   ├── enhanced_detector.py # Improved detection
│   │   └── pitch_detector.py    # Field boundary detection
│   ├── analysis/
│   │   ├── tactical_analytics.py
│   │   ├── formation_detection.py
│   │   ├── expected_goals.py
│   │   └── space_analysis.py
│   ├── data/
│   │   ├── team_database.py     # PostgreSQL database
│   │   └── cloud_sync.py        # Backup functionality
│   ├── gui/
│   │   ├── main_window.py       # Main application
│   │   └── stats_widget.py      # Statistics panel
│   └── database/
│       └── models.py            # ORM models
├── scripts/
│   ├── setup_database.py        # DB initialization
│   └── run_analysis.py          # CLI tool
├── data/
│   ├── videos/                  # Input videos
│   └── outputs/                 # Analysis results
├── requirements.txt
├── .env.example
├── run.bat / run.sh             # Run scripts
└── README.md
```

---

## Quick Reference

```bash
# Activate environment
venv\Scripts\activate            # Windows
source venv/bin/activate         # Linux/Mac

# Run GUI
python -m src.gui.main_window

# Run CLI analysis
python scripts/run_analysis.py VIDEO --depth quick

# Check database tables
psql -h localhost -U postgres -d soccer_analysis -c "\dt"

# Git commands
git status
git add .
git commit -m "Description"
git push
```

---

## Support

For issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in `logs/` directory
3. Open an issue on GitHub: https://github.com/governedchaos/soccer-film-analysis/issues
