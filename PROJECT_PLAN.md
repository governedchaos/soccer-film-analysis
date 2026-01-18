# Soccer Film Analysis App - Project Plan

## Project Overview

A high school soccer film analysis application that processes game footage to provide comprehensive team and player analytics. Uses local YOLOv8 models for detection - **no API keys required**.

## Current Status: Active Development

### Implemented Features

#### Core Detection (100% Complete)
- [x] **Player Detection & Tracking** - YOLOv8 with ByteTrack
- [x] **Team Classification** - K-means clustering on jersey colors
- [x] **Referee Detection** - Color-based classification (black, bright colors)
- [x] **Ball Tracking** - YOLO + color fallback + interpolation
- [x] **Pitch Boundary Detection** - Grass segmentation and line detection
- [x] **Enhanced Tracking** - Persistent team assignments across frames
- [x] **Out-of-bounds Filtering** - Excludes ball boys, coaches, substitutes

#### Tactical Analytics (100% Complete)
- [x] **Formation Detection** - Auto-detect 10+ formations (4-4-2, 4-3-3, etc.)
- [x] **Expected Goals (xG)** - Shot quality model
- [x] **Expected Threat (xT)** - Ball progression value
- [x] **Pressing Analysis** - PPDA calculation
- [x] **Team Shape Analysis** - Compactness, width, defensive line
- [x] **Space Creation** - Off-ball movement tracking
- [x] **Third-Man Run Detection** - Pattern identification
- [x] **Counter-Attack Detection** - Automatic identification
- [x] **Duel Statistics** - Aerial and ground duels

#### Advanced Features (100% Complete)
- [x] **Goalkeeper Analysis** - Distribution patterns
- [x] **Set Piece Analysis** - Corner and free kick patterns
- [x] **Passing Lane Analysis** - Lane blocking detection
- [x] **Fatigue Detection** - Speed/sprint decline tracking
- [x] **Opponent Tendency Prediction** - Pattern analysis
- [x] **Bird's Eye View** - Homography transformation

#### Application (90% Complete)
- [x] **PyQt6 GUI** - Modern dark theme interface
- [x] **Real-time Stats Panel** - Live possession, detections
- [x] **Video Caching** - Background preprocessing
- [x] **PostgreSQL Database** - Team/player persistence
- [x] **PDF Reports** - Match reports with charts
- [x] **Cloud Sync** - Local folder and Google Drive backup
- [ ] **Analysis Modules in GUI** - Pending integration

#### Pending Features
- [ ] Jersey number OCR recognition
- [ ] Multi-camera synchronization
- [ ] Live streaming analysis
- [ ] Mobile companion app

---

## Technology Stack

| Component | Technology | Status |
|-----------|------------|--------|
| Detection | YOLOv8 (local, no API) | Complete |
| Tracking | ByteTrack via Supervision | Complete |
| GUI | PyQt6 | Complete |
| Database | PostgreSQL + psycopg2 | Complete |
| Analytics | NumPy, SciPy, scikit-learn | Complete |
| Visualization | OpenCV, matplotlib | Complete |
| Reporting | ReportLab (PDF), Jinja2 (HTML) | Complete |

**Note:** All detection runs locally using YOLOv8 models. No Roboflow API key or external services required.

---

## Project Structure

```
soccer_film_analysis/
├── config/
│   └── settings.py              # Pydantic settings
├── src/
│   ├── core/
│   │   └── video_processor.py   # Main pipeline
│   ├── detection/
│   │   ├── detector.py          # Base YOLO detection
│   │   ├── enhanced_detector.py # Improved detection
│   │   └── pitch_detector.py    # Pitch boundaries
│   ├── analysis/
│   │   ├── tactical_analytics.py
│   │   ├── advanced_tactical.py
│   │   ├── formation_detection.py
│   │   ├── expected_goals.py
│   │   ├── space_analysis.py
│   │   └── pdf_report.py
│   ├── data/
│   │   ├── team_database.py     # PostgreSQL
│   │   └── cloud_sync.py        # Backup
│   ├── processing/
│   │   └── video_cache.py       # Preprocessing
│   ├── gui/
│   │   ├── main_window.py
│   │   └── stats_widget.py
│   └── database/
│       └── models.py            # SQLAlchemy ORM
├── scripts/
│   ├── setup_database.py
│   └── run_analysis.py
└── data/
    ├── videos/
    └── outputs/
```

---

## Detection Pipeline

### EnhancedDetector Features

1. **Pitch Boundary Detection**
   - Grass color segmentation (HSV)
   - White line detection
   - Filters out-of-bounds detections

2. **Referee Classification**
   - Auto-detects black kits
   - Detects bright colors (assistant refs)
   - Separate from team assignments

3. **Stable Team Assignment**
   - TrackedPerson class with history
   - Majority voting over 30 frames
   - Prevents frame-to-frame flickering

4. **Ball Detection Fallback**
   - YOLO detection primary
   - Color-based fallback (white/orange)
   - Position interpolation

---

## Analytics Modules

### Formation Detection
- Template matching for 10+ formations
- Hungarian algorithm for player assignment
- Confidence scoring and smoothing
- Formation change detection

### Expected Goals (xG)
- Distance and angle to goal
- Shot type (foot, header)
- Defensive pressure
- Goalkeeper position
- Game situation modifiers

### Space Analysis
- Run type classification
- Space creation quantification
- Third-man run pattern detection
- Off-ball movement tracking

---

## Database Schema

### Core Tables
- **games** - Match metadata
- **teams** - Team information with colors
- **players** - Player data with jersey numbers
- **game_records** - Per-game player stats
- **events** - Detected game events

### Analytics Tables
- **formations** - Formation snapshots
- **possession_data** - Frame-by-frame possession
- **tracking_data** - Player positions

---

## Running the Application

### Quick Start
```bash
# Clone and setup
git clone https://github.com/governedchaos/soccer-film-analysis.git
cd soccer-film-analysis
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run GUI
python -m src.gui.main_window
```

### Database Setup (Optional)
```bash
# Configure .env with PostgreSQL credentials
python scripts/setup_database.py
```

### Command Line
```bash
python scripts/run_analysis.py video.mp4 --depth standard
```

---

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Player detection | 90%+ | Testing |
| Team classification | 85%+ | Testing |
| Ball tracking | <20% lost | Improved |
| Referee detection | 95%+ | New |
| Out-of-bounds filtering | 95%+ | New |
| Processing time | <60 min | Met |
| Real-time display | 15+ FPS | Met |

---

## Next Steps

1. **Integrate analysis modules into GUI**
   - Formation display panel
   - xG shot map visualization
   - Space analysis overlays

2. **Jersey Number OCR**
   - EasyOCR integration
   - Number tracking per player

3. **Performance Optimization**
   - GPU acceleration
   - Batch processing
   - Frame skipping options

4. **Export Improvements**
   - Interactive HTML reports
   - Excel workbooks
   - Video clips export
