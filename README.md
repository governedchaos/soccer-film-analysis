# Soccer Film Analysis

A comprehensive high school soccer game film analysis application that uses computer vision to detect players, track movements, identify events, and generate detailed analytics.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## Features

### Core Detection
- **Player Detection & Tracking** - Identify and track all players with persistent IDs
- **Team Classification** - Automatically differentiate teams by jersey color
- **Ball Tracking** - Real-time ball detection and possession analysis
- **Referee Detection** - Identify and exclude referees from team statistics
- **Pitch Detection** - Detect field boundaries for coordinate transformation

### Analytics
- **Event Detection** - Goals, shots, passes, turnovers, tackles, saves
- **Team Metrics** - Possession, passing accuracy, formations, pressing intensity
- **Player Metrics** - Distance, speed, sprints, passes, shots, performance rating
- **Advanced Stats** - Expected Goals (xG), passing networks, heatmaps

### Visualization
- **Real-time Overlay** - Live detection boxes during playback
- **Heatmaps** - Player and team movement visualization
- **Formation Display** - 2D pitch representation with player positions
- **Event Timeline** - Navigate to specific game events

### Application
- **Modern GUI** - PyQt6-based interface with dark theme
- **Analysis Depth Levels** - Quick (5-10 min), Standard (20-30 min), Deep (45-60 min)
- **Report Generation** - HTML, PDF, Excel exports
- **Database Storage** - PostgreSQL for persistent data

## Technology Stack

| Component | Technology |
|-----------|------------|
| Detection | Roboflow Sports, YOLOv8, Supervision |
| Tracking | ByteTrack via Supervision |
| GUI | PyQt6 |
| Database | PostgreSQL with SQLAlchemy |
| Visualization | OpenCV, matplotlib, mplsoccer |
| Reporting | Jinja2, WeasyPrint, openpyxl |

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- GPU recommended (CUDA for NVIDIA, MPS for Apple Silicon)
- [Roboflow API Key](https://app.roboflow.com/) (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/soccer_film_analysis.git
cd soccer_film_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Roboflow Sports (if not in requirements)
pip install git+https://github.com/roboflow/sports.git

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings (database credentials, Roboflow API key)
```

### Database Setup

```bash
# Make sure PostgreSQL is running, then:
python scripts/setup_database.py
```

### Running the Application

```bash
# GUI Application
python -m src.gui.main_window

# Command Line Analysis
python scripts/run_analysis.py path/to/video.mp4 --depth standard
```

## Running in Claude Code

This project is designed to work seamlessly with Claude Code. Here's how to set it up:

### 1. Clone and Setup

In Claude Code terminal:

```bash
# Navigate to your workspace
cd /path/to/your/projects

# Clone or create the project
git clone https://github.com/yourusername/soccer_film_analysis.git
cd soccer_film_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# The key packages are:
pip install supervision ultralytics inference roboflow
pip install git+https://github.com/roboflow/sports.git
pip install PyQt6 psycopg2-binary sqlalchemy loguru rich
```

### 3. Configure Environment

Create `.env` file:

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=soccer_analysis
DB_USER=postgres
DB_PASSWORD=your_password

# Roboflow API Key (get free at roboflow.com)
ROBOFLOW_API_KEY=your_api_key

# Settings
LOG_LEVEL=INFO
ENABLE_GPU=true
```

### 4. Setup Database

```bash
# Make sure PostgreSQL is installed and running
# On Mac: brew install postgresql && brew services start postgresql
# On Linux: sudo apt install postgresql && sudo systemctl start postgresql
# On Windows: Download from postgresql.org

# Create the database
python scripts/setup_database.py
```

### 5. Run Analysis

```bash
# GUI Mode
python -m src.gui.main_window

# CLI Mode
python scripts/run_analysis.py data/videos/game.mp4 --depth standard

# Quick test with sample frames
python scripts/run_analysis.py game.mp4 --depth quick --start 0 --end 1000
```

### 6. Push to GitHub

```bash
# Initialize git if not already done
git init

# Add remote
git remote add origin https://github.com/yourusername/soccer_film_analysis.git

# Commit and push
git add .
git commit -m "Initial commit: Soccer film analysis app"
git push -u origin main
```

## Project Structure

```
soccer_film_analysis/
├── config/
│   ├── __init__.py
│   └── settings.py          # Pydantic settings management
├── src/
│   ├── core/
│   │   └── video_processor.py    # Main processing pipeline
│   ├── detection/
│   │   └── detector.py           # Roboflow Sports integration
│   ├── tracking/                 # Object tracking
│   ├── analytics/                # Event detection & metrics
│   ├── visualization/            # Overlays & heatmaps
│   ├── database/
│   │   └── models.py             # SQLAlchemy models
│   ├── gui/
│   │   └── main_window.py        # PyQt6 interface
│   └── reports/                  # Report generation
├── scripts/
│   ├── setup_database.py         # Database initialization
│   └── run_analysis.py           # CLI analysis tool
├── tests/                        # Unit tests
├── data/
│   ├── videos/                   # Input videos
│   ├── outputs/                  # Analysis outputs
│   └── models/                   # Downloaded models
├── logs/                         # Application logs
├── requirements.txt
├── .env.example
└── README.md
```

## Usage Examples

### Basic Analysis

```python
from src.core.video_processor import VideoProcessor
from config import AnalysisDepth

# Initialize
processor = VideoProcessor()

# Load video
video_info = processor.load_video("game.mp4")

# Calibrate team colors (or provide manually)
processor.calibrate_teams()

# Run analysis
result = processor.process_video(
    analysis_depth=AnalysisDepth.STANDARD,
    progress_callback=lambda p: print(f"Progress: {p.percentage:.1f}%")
)
```

### With Manual Team Colors

```python
# If you know the jersey colors
processor.team_classifier.set_team_colors(
    home_color=(255, 255, 0),  # Yellow
    away_color=(0, 0, 255)     # Blue
)
```

### Real-time Frame Display

```python
import cv2

def on_frame(frame, detections):
    cv2.imshow("Analysis", frame)
    cv2.waitKey(1)

processor.process_video(
    analysis_depth=AnalysisDepth.QUICK,
    frame_callback=on_frame
)
```

## Configuration

All settings can be configured via environment variables or `.env` file:

| Setting | Description | Default |
|---------|-------------|---------|
| `DB_HOST` | PostgreSQL host | localhost |
| `DB_PORT` | PostgreSQL port | 5432 |
| `DB_NAME` | Database name | soccer_analysis |
| `ROBOFLOW_API_KEY` | Roboflow API key | required |
| `LOG_LEVEL` | Logging level | INFO |
| `ENABLE_GPU` | Use GPU acceleration | true |
| `PLAYER_CONFIDENCE_THRESHOLD` | Detection confidence | 0.3 |

## Analysis Depth Levels

| Level | Frame Sample | Processing Time | Features |
|-------|--------------|-----------------|----------|
| Quick | 1/10 frames | 5-10 min | Basic detection only |
| Standard | 1/5 frames | 20-30 min | Detection + events |
| Deep | 1/2 frames | 45-60 min | Full analytics |

## Troubleshooting

### Common Issues

**PostgreSQL Connection Failed**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Check credentials in .env
```

**Roboflow Model Not Loading**
```bash
# Verify API key
python -c "from inference import get_model; print('OK')"

# Check internet connection
```

**GUI Not Displaying**
```bash
# Make sure PyQt6 is installed correctly
pip install --force-reinstall PyQt6 PyQt6-Qt6

# On headless servers, use CLI mode instead
```

**GPU Not Detected**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# For Apple Silicon
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Roboflow Sports](https://github.com/roboflow/sports) - Detection models and utilities
- [Supervision](https://github.com/roboflow/supervision) - Computer vision utilities
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) - Soccer visualizations

## Roadmap

- [ ] Jersey number OCR recognition
- [ ] Advanced event detection (tactical fouls, set pieces)
- [ ] Multi-camera support
- [ ] Cloud deployment (Azure/GCP)
- [ ] Mobile companion app API
- [ ] Custom model fine-tuning
- [ ] Season-long player tracking
