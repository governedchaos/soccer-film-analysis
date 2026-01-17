# Soccer Film Analysis - Deployment Guide

This guide covers deploying the soccer film analysis application from Claude Desktop to your local environment, pushing to GitHub, and using Claude Code for testing and improvements.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Get the Code](#get-the-code)
3. [Local Environment Setup](#local-environment-setup)
4. [PostgreSQL Configuration](#postgresql-configuration)
5. [Install Dependencies](#install-dependencies)
6. [Configure Environment](#configure-environment)
7. [Initialize Database](#initialize-database)
8. [Test the Application](#test-the-application)
9. [Push to GitHub](#push-to-github)
10. [Claude Code Workflow](#claude-code-workflow)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.10+** installed
- **PostgreSQL 14+** installed and running
- **Git** installed
- **GitHub account** with repository created
- **Roboflow account** (free tier) for API key: https://app.roboflow.com/

### Verify Prerequisites

```bash
# Check Python version (need 3.10+)
python --version

# Check PostgreSQL is running
pg_isready

# Check Git
git --version
```

---

## Get the Code

### Option A: Download from Claude Desktop Session

If you're in Claude Desktop, ask Claude to create a zip of the project:

```
Please create a zip file of the soccer_film_analysis project
```

Then download and extract to your preferred location.

### Option B: Clone from GitHub (after initial push)

```bash
git clone https://github.com/YOUR_USERNAME/soccer_film_analysis.git
cd soccer_film_analysis
```

### Option C: Manual File Creation

Create the project structure manually and copy files from the Claude Desktop conversation.

---

## Local Environment Setup

### 1. Create Project Directory

```bash
# Navigate to your projects folder
cd ~/projects  # or wherever you keep projects

# Create project directory (if not cloning)
mkdir soccer_film_analysis
cd soccer_film_analysis
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Verify Virtual Environment

```bash
# Should show path to venv Python
which python  # macOS/Linux
where python  # Windows
```

---

## PostgreSQL Configuration

### 1. Create Database

```bash
# Connect to PostgreSQL as superuser
psql -U postgres

# In PostgreSQL prompt:
CREATE DATABASE soccer_analysis;

# Verify database was created
\l

# Exit
\q
```

### 2. Create Database User (Optional but Recommended)

```sql
-- In PostgreSQL prompt:
CREATE USER soccer_app WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE soccer_analysis TO soccer_app;

-- Connect to the database and grant schema permissions
\c soccer_analysis
GRANT ALL ON SCHEMA public TO soccer_app;
```

### 3. Test Connection

```bash
# Test connection with your credentials
psql -h localhost -U soccer_app -d soccer_analysis

# If successful, you'll see the database prompt
# Exit with \q
```

---

## Install Dependencies

### 1. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 2. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Roboflow Sports (Additional)

```bash
pip install git+https://github.com/roboflow/sports.git
```

### 4. Verify Key Packages

```bash
# Test imports
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import supervision; print(f'Supervision: {supervision.__version__}')"
python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6: OK')"
python -c "import sqlalchemy; print(f'SQLAlchemy: {sqlalchemy.__version__}')"
```

### 5. Check GPU Availability (Optional)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

## Configure Environment

### 1. Create .env File

```bash
# Copy example file
cp .env.example .env
```

### 2. Edit .env with Your Settings

Open `.env` in your editor and update:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=soccer_analysis
DB_USER=soccer_app          # or postgres
DB_PASSWORD=your_password   # your actual password

# Roboflow API Key (get from https://app.roboflow.com/)
ROBOFLOW_API_KEY=your_actual_api_key

# Application Settings
LOG_LEVEL=INFO
DEFAULT_ANALYSIS_DEPTH=standard
ENABLE_GPU=true
```

### 3. Verify Configuration

```bash
# Test that settings load correctly
python -c "from config.settings import settings; print(f'DB: {settings.db_name}'); print(f'Device: {settings.get_device()}')"
```

---

## Initialize Database

### 1. Run Database Setup Script

```bash
# Check connection first
python scripts/setup_database.py --check

# Create tables
python scripts/setup_database.py
```

### 2. Verify Tables Created

```bash
# Connect to database and list tables
psql -h localhost -U soccer_app -d soccer_analysis -c "\dt"
```

Expected tables:
- games
- teams
- players
- tracking_data
- events
- player_metrics
- team_metrics
- formations
- analysis_sessions

---

## Test the Application

### 1. Quick Smoke Test

```bash
# Test imports and configuration
python -c "
from config.settings import settings
from src.database.models import Game, Team, Player
from src.detection.detector import SoccerDetector
print('All imports successful!')
print(f'Database: {settings.db_name}')
print(f'Device: {settings.get_device()}')
"
```

### 2. Test with Sample Video (CLI)

```bash
# Place a test video in data/videos/
# Then run analysis
python scripts/run_analysis.py data/videos/test_clip.mp4 --depth quick
```

### 3. Launch GUI Application

```bash
# Run the main GUI application
python -m src.gui.main_window
```

### 4. Test Checklist

- [ ] Application window opens
- [ ] Can load a video file
- [ ] Video displays in player
- [ ] Can start analysis (even if short)
- [ ] Detection boxes appear on video
- [ ] Stats panel updates
- [ ] No database connection errors in console

---

## Push to GitHub

### 1. Initialize Git Repository

```bash
# Initialize repo (if not already done)
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Soccer film analysis application

Features:
- Player detection with Roboflow Sports models
- Team classification by jersey color
- Real-time visualization during analysis
- PostgreSQL database for persistence
- PyQt6 GUI with video player
- CLI tool for batch processing
- Configurable analysis depth levels"
```

### 2. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `soccer_film_analysis`
3. Description: "High school soccer film analysis application with AI-powered player detection"
4. Set to Private (recommended for now)
5. Do NOT initialize with README (we already have one)
6. Click "Create repository"

### 3. Push to GitHub

```bash
# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/soccer_film_analysis.git

# Push to main branch
git branch -M main
git push -u origin main
```

### 4. Verify Push

- Go to your GitHub repository URL
- Confirm all files are present
- Check that .env is NOT pushed (should be in .gitignore)

---

## Claude Code Workflow

Once the code is on GitHub, you can use Claude Code to test, run, and improve the application.

### 1. Open Claude Code

Launch Claude Code and navigate to your project:

```bash
cd ~/projects/soccer_film_analysis
```

### 2. Common Claude Code Tasks

#### Run the Application
```
Run the GUI application and help me test it with a video file
```

#### Debug Issues
```
I'm getting this error when running the app: [paste error]
Help me debug and fix it
```

#### Add New Features
```
Add jersey number detection using EasyOCR to the detection pipeline
```

#### Improve Detection Accuracy
```
The team classification is showing too many false positives. 
Help me add better filtering for referees and spectators
```

#### Generate Reports
```
Add HTML report generation that shows player heatmaps and team statistics
```

#### Code Quality
```
Review the video_processor.py file and suggest improvements for error handling and performance
```

### 3. Testing Workflow in Claude Code

1. **Load a test video**: Place a short clip (30-60 seconds) in `data/videos/`
2. **Run quick analysis**: `python scripts/run_analysis.py data/videos/test.mp4 --depth quick`
3. **Check results**: Review console output and database records
4. **Launch GUI**: `python -m src.gui.main_window`
5. **Iterate**: Make changes, test again

### 4. Suggested Improvement Tasks

Once basic functionality works, ask Claude Code to help with:

1. **Event Detection**: Add goal, shot, pass detection algorithms
2. **Formation Analysis**: Detect 4-4-2, 4-3-3, etc. formations
3. **Heatmap Generation**: Create player movement heatmaps
4. **Report Export**: Generate PDF/HTML match reports
5. **Performance Optimization**: Batch database writes, frame caching
6. **UI Enhancements**: Event timeline, formation visualization

---

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Test connection manually
psql -h localhost -U postgres -d soccer_analysis

# Check .env has correct credentials
cat .env | grep DB_
```

### Import Errors

```bash
# Ensure virtual environment is activated
which python  # Should show venv path

# Reinstall problematic package
pip uninstall package_name
pip install package_name
```

### PyQt6 Display Issues (Linux)

```bash
# Install system dependencies
sudo apt-get install libxcb-xinerama0 libxcb-cursor0

# Or try with platform override
QT_QPA_PLATFORM=xcb python -m src.gui.main_window
```

### GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Roboflow API Errors

```bash
# Verify API key
python -c "from config.settings import settings; print(f'API Key set: {bool(settings.roboflow_api_key)}')"

# Test Roboflow connection
python -c "
from inference import get_model
model = get_model('football-players-detection-3zvbc/10')
print('Roboflow connection successful!')
"
```

### Video Processing Errors

```bash
# Check OpenCV can read the video
python -c "
import cv2
cap = cv2.VideoCapture('data/videos/test.mp4')
print(f'Opened: {cap.isOpened()}')
print(f'Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
cap.release()
"
```

---

## Quick Reference Commands

```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Run GUI
python -m src.gui.main_window

# Run CLI analysis
python scripts/run_analysis.py VIDEO_PATH --depth quick|standard|deep

# Reset database
python scripts/setup_database.py --reset

# Check database tables
psql -h localhost -U soccer_app -d soccer_analysis -c "\dt"

# Run tests (when added)
pytest tests/

# Git status
git status

# Push changes
git add .
git commit -m "Description of changes"
git push
```

---

## Project Structure Reference

```
soccer_film_analysis/
├── config/
│   ├── __init__.py
│   └── settings.py           # Pydantic configuration
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   └── video_processor.py  # Main processing pipeline
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py        # Roboflow detection
│   ├── database/
│   │   ├── __init__.py
│   │   └── models.py          # SQLAlchemy models
│   └── gui/
│       ├── __init__.py
│       └── main_window.py     # PyQt6 interface
├── scripts/
│   ├── setup_database.py      # DB initialization
│   └── run_analysis.py        # CLI tool
├── data/
│   ├── videos/                # Input videos
│   ├── outputs/               # Results
│   └── models/                # Downloaded models
├── logs/                      # Application logs
├── requirements.txt
├── .env                       # Your configuration (not in git)
├── .env.example              # Configuration template
├── .gitignore
├── README.md
├── DEPLOYMENT.md             # This file
└── setup.sh                  # Quick setup script
```

---

## Next Steps After Deployment

1. **Test with real game footage** - Use a short clip first (1-2 minutes)
2. **Calibrate team colors** - Run auto-calibration or set manually
3. **Verify detection accuracy** - Check player counts match reality
4. **Tune confidence thresholds** - Adjust in .env if needed
5. **Add event detection** - Goals, passes, shots
6. **Build reports** - HTML/PDF generation
7. **Optimize performance** - Profile and improve slow areas

---

## Support

For issues with this deployment:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review application logs in `logs/` directory
3. Use Claude Code to debug specific errors
4. Check GitHub Issues for known problems

Happy analyzing! ⚽
