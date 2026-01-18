# Soccer Film Analysis

A comprehensive high school soccer game film analysis application using local computer vision models. No API keys required - runs entirely offline using YOLOv8 for detection.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## Features

### Core Detection (Runs Offline - No API Required)
- **Player Detection & Tracking** - YOLOv8 with ByteTrack for persistent player IDs
- **Team Classification** - K-means clustering on jersey colors with persistent tracking
- **Ball Tracking** - Multi-method detection (YOLO + color-based fallback + interpolation)
- **Referee Detection** - Automatic classification based on color patterns (black, bright colors)
- **Pitch Boundary Detection** - Grass segmentation and line detection to filter out-of-bounds
- **Enhanced Tracking** - Stable team assignments that don't flicker between frames

### Tactical Analytics
- **Formation Detection** - Auto-detect 10+ formations (4-4-2, 4-3-3, 3-5-2, etc.)
- **Expected Goals (xG)** - Shot quality model with distance, angle, pressure factors
- **Expected Threat (xT)** - Ball progression value using 12x8 zone grid
- **Pressing Analysis** - PPDA calculation and pressing intensity metrics
- **Team Shape Analysis** - Compactness, width, defensive line tracking
- **Space Creation** - Off-ball movement and third-man run detection
- **Counter-Attack Detection** - Automatic identification of counter-attacking plays
- **Duel Statistics** - Aerial and ground duel tracking with win rates

### Advanced Features
- **Goalkeeper Analysis** - Distribution patterns, saves, positioning
- **Set Piece Analysis** - Corner and free kick pattern detection
- **Passing Lane Analysis** - Lane blocking and available passing options
- **Fatigue Detection** - Speed/sprint decline tracking per player
- **Opponent Tendency Prediction** - Pattern analysis for scouting
- **Bird's Eye View** - Homography transformation for tactical view

### Application
- **Modern GUI** - PyQt6-based interface with dark theme
- **Real-time Stats Panel** - Live possession, shots, passes, detection quality
- **Video Caching** - Background preprocessing for faster analysis
- **PDF Reports** - Comprehensive match reports with charts and diagrams
- **Database Storage** - PostgreSQL for persistent game/player data
- **Cloud Sync** - Backup to local folders or Google Drive

## Technology Stack

| Component | Technology |
|-----------|------------|
| Detection | YOLOv8 (local models, no API needed) |
| Tracking | ByteTrack via Supervision |
| GUI | PyQt6 |
| Database | PostgreSQL with psycopg2 |
| Analytics | NumPy, SciPy, scikit-learn |
| Visualization | OpenCV, matplotlib |
| Reporting | ReportLab (PDF), Jinja2 (HTML) |

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (for database features)
- GPU recommended but not required (CUDA for NVIDIA, MPS for Apple Silicon)
- **No API keys required** - all detection runs locally

### Installation

```bash
# Clone the repository
git clone https://github.com/governedchaos/soccer-film-analysis.git
cd soccer-film-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment (optional - for database)
cp .env.example .env
# Edit .env with your PostgreSQL credentials if using database features
```

### First Run

The first time you run the app, YOLOv8 models will be downloaded automatically (~25MB).

```bash
# Launch GUI Application
python -m src.gui.main_window

# Or use the run script
./run.sh gui      # Linux/Mac
run.bat gui       # Windows
```

### Database Setup (Optional)

If you want to persist game data across sessions:

```bash
# Make sure PostgreSQL is running
python scripts/setup_database.py
```

## Usage

### GUI Mode

1. Launch the application: `python -m src.gui.main_window`
2. Click "Open Video" to load a game recording
3. Set team colors using the color picker (or let auto-detection work)
4. Click "Start Analysis" to begin processing
5. View real-time detection and statistics
6. Export results as PDF or save to database

### Command Line Mode

```bash
# Quick analysis (fastest, samples frames)
python scripts/run_analysis.py path/to/game.mp4 --depth quick

# Standard analysis (balanced)
python scripts/run_analysis.py path/to/game.mp4 --depth standard

# Deep analysis (most detailed)
python scripts/run_analysis.py path/to/game.mp4 --depth deep
```

### Programmatic Usage

```python
from src.core.video_processor import VideoProcessor

# Initialize (uses EnhancedDetector by default)
processor = VideoProcessor()

# Load video
video_info = processor.load_video("game.mp4")

# Set team colors (optional - auto-detection available)
processor.team_classifier.set_team_colors(
    home_color=(255, 255, 0),  # Yellow jerseys
    away_color=(0, 0, 255)     # Blue jerseys
)

# Process with progress callback
result = processor.process_video(
    progress_callback=lambda p: print(f"Progress: {p.percentage:.1f}%")
)
```

## Project Structure

```
soccer_film_analysis/
├── config/
│   └── settings.py              # Pydantic settings management
├── src/
│   ├── core/
│   │   └── video_processor.py   # Main processing pipeline
│   ├── detection/
│   │   ├── detector.py          # Base YOLO detection
│   │   ├── enhanced_detector.py # Improved detection with filtering
│   │   └── pitch_detector.py    # Pitch boundary detection
│   ├── analysis/
│   │   ├── tactical_analytics.py    # Pressing, zones, set pieces
│   │   ├── advanced_tactical.py     # xT, duels, fatigue, shape
│   │   ├── formation_detection.py   # Formation auto-detection
│   │   ├── expected_goals.py        # xG model
│   │   ├── space_analysis.py        # Space creation, third-man runs
│   │   └── pdf_report.py            # PDF report generation
│   ├── data/
│   │   ├── team_database.py     # PostgreSQL team/player database
│   │   └── cloud_sync.py        # Cloud backup functionality
│   ├── processing/
│   │   └── video_cache.py       # Video preprocessing cache
│   ├── gui/
│   │   ├── main_window.py       # Main PyQt6 interface
│   │   └── stats_widget.py      # Real-time statistics panel
│   └── database/
│       └── models.py            # SQLAlchemy ORM models
├── scripts/
│   ├── setup_database.py        # Database initialization
│   └── run_analysis.py          # CLI analysis tool
├── data/
│   ├── videos/                  # Input videos
│   └── outputs/                 # Analysis outputs
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

Settings via environment variables or `.env` file:

| Setting | Description | Default |
|---------|-------------|---------|
| `DB_HOST` | PostgreSQL host | localhost |
| `DB_PORT` | PostgreSQL port | 5432 |
| `DB_NAME` | Database name | soccer_analysis |
| `DB_USER` | Database user | postgres |
| `DB_PASSWORD` | Database password | (empty) |
| `LOG_LEVEL` | Logging level | INFO |
| `ENABLE_GPU` | Use GPU acceleration | true |
| `PLAYER_CONFIDENCE_THRESHOLD` | Detection confidence | 0.3 |
| `BALL_CONFIDENCE_THRESHOLD` | Ball detection confidence | 0.15 |

## Detection Features

### Enhanced Detection Pipeline

The `EnhancedDetector` provides several improvements over basic YOLO:

1. **Pitch Boundary Filtering** - Detects the playing field and filters out:
   - Ball boys
   - Substitutes on the sideline
   - Coaches
   - Spectators near the field

2. **Referee Classification** - Automatically identifies referees by:
   - Black kit detection
   - Bright color detection (yellow, pink for assistant refs)
   - Referees are NOT assigned to home/away teams

3. **Stable Team Assignment** - Uses tracking history to prevent flickering:
   - Maintains color history per tracked player
   - Uses majority voting over 30 frames
   - Players maintain team assignment consistently

4. **Ball Detection Fallback** - When YOLO misses the ball:
   - Color-based detection for white/orange balls
   - Position interpolation from recent frames
   - Works even in crowded scenes

## Analysis Modules

### Formation Detection
Automatically identifies team formations using template matching:
- 4-4-2, 4-3-3, 4-2-3-1, 3-5-2, 3-4-3, 5-3-2, 5-4-1, and more
- Tracks formation changes throughout the match
- Uses Hungarian algorithm for optimal player-position matching

### Expected Goals (xG)
Shot quality model considering:
- Distance and angle to goal
- Shot type (foot, header)
- Defensive pressure
- Goalkeeper positioning
- Game situation (open play, counter, set piece)

### Space Analysis
Tracks off-ball movement:
- Decoy runs, overlap/underlap runs
- Third-man run pattern detection
- Space creation quantification
- Player run classification

## Troubleshooting

### Common Issues

**Detection not working well**
- Ensure good video quality (720p+ recommended)
- Set team colors manually if auto-detection struggles
- Check that the pitch is clearly visible

**Ball not being detected**
- Ball detection improves with higher resolution video
- The fallback color detection helps with white/orange balls
- Position interpolation fills short gaps

**Players switching teams**
- The enhanced detector uses tracking history to stabilize assignments
- Allow a few seconds for the tracker to learn consistent assignments
- Manually setting team colors helps significantly

**PostgreSQL connection failed**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Verify credentials in .env file
```

**GPU not detected**
```bash
# Check CUDA (NVIDIA)
python -c "import torch; print(torch.cuda.is_available())"

# Check MPS (Apple Silicon)
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

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [Supervision](https://github.com/roboflow/supervision) - Computer vision utilities and tracking
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework

## Roadmap

- [x] Local YOLO detection (no API required)
- [x] Enhanced referee detection
- [x] Pitch boundary filtering
- [x] Formation auto-detection
- [x] Expected Goals (xG) model
- [x] Space creation analysis
- [x] Third-man run detection
- [x] Video preprocessing cache
- [x] PostgreSQL database
- [x] PDF report generation
- [ ] Jersey number OCR recognition
- [ ] Multi-camera synchronization
- [ ] Live streaming analysis
- [ ] Mobile companion app
- [ ] Season-long player tracking
