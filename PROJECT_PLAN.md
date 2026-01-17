# Soccer Film Analysis App - Complete Rebuild Plan

## Project Overview

A high school soccer film analysis application that processes game footage to provide comprehensive team and player analytics. Built from scratch using modern open-source tools and designed to run in Claude Code.

## Consolidated Requirements (From All Previous Chats)

### Core Detection Features
1. **Player Detection & Tracking**
   - Identify all players on the field with persistent IDs
   - Track player movement throughout the game
   - Filter out non-players (coaches, spectators, ball boys)

2. **Team Differentiation**
   - Identify jersey colors using K-means clustering
   - Assign players to home/away teams
   - Manual color override option for accuracy
   - Support for goalkeeper differentiation

3. **Referee Detection**
   - Identify referees separately from players
   - Exclude from team statistics

4. **Ball Tracking**
   - Real-time ball detection and tracking
   - Handle occlusion and fast movement
   - Ball possession determination

5. **Jersey Number Recognition (OCR)**
   - Read jersey numbers when visible
   - Associate numbers with tracked players
   - Handle partial visibility and motion blur

6. **Field/Pitch Detection**
   - Detect field boundaries and lines
   - Perspective transformation to top-down view
   - Support for varying camera angles

### Analytics Features
1. **Event Detection**
   - Goals
   - Assists
   - Shots (on target, off target)
   - Passes (successful, unsuccessful)
   - Turnovers/steals
   - Tackles
   - Goalkeeper saves
   - Fouls (if detectable)

2. **Team Metrics**
   - Possession percentage
   - Pass completion rate
   - Total distance covered
   - Formation detection (4-4-2, 4-3-3, etc.)
   - Formation changes over time
   - Pressing intensity
   - Defensive actions

3. **Player Metrics**
   - Distance covered
   - Speed (max, average)
   - Sprints count
   - Passes made/received
   - Shots taken
   - Goals/assists
   - Performance rating
   - Position heatmaps

4. **Advanced Analytics**
   - Expected Goals (xG)
   - Passing networks
   - Pressure maps
   - Risk/opportunity zones
   - Problem areas identification

### Visualization Features
1. **Real-Time Display**
   - Bounding boxes on detected objects
   - Team color-coded overlays
   - Statistics panel during playback
   - Confidence scores

2. **Heatmaps**
   - Player movement heatmaps
   - Team activity heatmaps
   - Shot location maps
   - Pass origin/destination maps

3. **Formation Visualization**
   - 2D pitch representation
   - Radar-style player positioning
   - Voronoi diagrams for control zones

4. **Event Annotations**
   - Timestamped event markers
   - Click-to-navigate to events
   - Event filtering by type/player

### Application Features
1. **Video Processing**
   - Upload MP4 video files
   - Variable playback speed (0.25x - 4x)
   - Frame-by-frame navigation
   - **Processing time under 60 minutes**

2. **Analysis Depth Levels**
   - Quick (5-10 min) - Basic detection only
   - Standard (20-30 min) - Detection + core events
   - Deep (45-60 min) - Full analytics

3. **Pre-Analysis Setup**
   - Team color selection
   - Roster upload (optional)
   - Game timing controls (select game period)
   - Field boundary marking

4. **Feedback System**
   - Correct misclassified objects
   - Manual event tagging
   - Model improvement through feedback

5. **Reporting**
   - Team summary reports
   - Individual player reports
   - HTML/PDF export
   - Excel workbooks with multiple sheets
   - Interactive charts (Plotly)

### Technical Requirements
1. **Stack**
   - Python 3.10+
   - PostgreSQL database
   - Modern GUI (PyQt6 recommended over tkinter)
   - GitHub integration

2. **Performance**
   - GPU acceleration (CUDA)
   - Batch processing
   - Multi-threading
   - Chunked video processing for reliability

3. **Logging & Debugging**
   - Comprehensive logging
   - Error recovery
   - Progress tracking

---

## Technology Stack

### Open-Source Tools to Leverage

1. **Roboflow Sports** (MIT License)
   - `pip install git+https://github.com/roboflow/sports.git`
   - Pre-trained soccer models for player/ball/pitch detection
   - Team classification using SigLIP + UMAP + KMeans
   - Pitch keypoint detection for perspective transformation
   - Radar visualization

2. **Supervision** (MIT License)
   - `pip install supervision`
   - ByteTrack for object tracking
   - Built-in annotators for visualization
   - Video frame processing utilities

3. **Ultralytics YOLOv8**
   - `pip install ultralytics`
   - Object detection backbone
   - Fast inference with GPU support

4. **Roboflow Inference**
   - `pip install inference`
   - Access to pre-trained soccer detection models
   - No training required initially

5. **Additional Libraries**
   - `psycopg2-binary` - PostgreSQL
   - `PyQt6` - Modern GUI
   - `opencv-python` - Video processing
   - `mplsoccer` - Soccer visualizations
   - `pandas` - Data analysis
   - `plotly` - Interactive charts
   - `easyocr` - Jersey number recognition

---

## Project Structure

```
soccer_film_analysis/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
│
├── config/
│   ├── __init__.py
│   ├── settings.py          # App configuration
│   └── database.py          # DB connection settings
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── video_processor.py    # Main video processing pipeline
│   │   ├── analysis_engine.py    # Orchestrates all analysis
│   │   └── progress_tracker.py   # Track and report progress
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── player_detector.py    # Player detection using Roboflow
│   │   ├── ball_detector.py      # Ball tracking
│   │   ├── pitch_detector.py     # Field line detection
│   │   └── team_classifier.py    # Team assignment
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── object_tracker.py     # ByteTrack integration
│   │   ├── player_tracker.py     # Player-specific tracking
│   │   └── ball_tracker.py       # Ball-specific tracking
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── event_detector.py     # Detect game events
│   │   ├── possession_analyzer.py
│   │   ├── formation_analyzer.py
│   │   ├── player_metrics.py
│   │   ├── team_metrics.py
│   │   └── xg_calculator.py      # Expected goals
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── real_time_overlay.py  # Live detection display
│   │   ├── heatmap_generator.py
│   │   ├── formation_visualizer.py
│   │   ├── passing_network.py
│   │   └── pitch_renderer.py     # 2D pitch views
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py             # SQLAlchemy models
│   │   ├── repository.py         # Data access layer
│   │   └── migrations/           # Database migrations
│   │
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py        # Main application window
│   │   ├── video_player.py       # Video playback widget
│   │   ├── analysis_panel.py     # Analysis controls
│   │   ├── results_panel.py      # Display results
│   │   ├── setup_wizard.py       # Pre-analysis setup
│   │   └── feedback_dialog.py    # Correction interface
│   │
│   └── reports/
│       ├── __init__.py
│       ├── report_generator.py
│       ├── templates/            # HTML templates
│       └── exporters/
│           ├── html_exporter.py
│           ├── pdf_exporter.py
│           └── excel_exporter.py
│
├── tests/
│   ├── __init__.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_analytics.py
│   └── test_database.py
│
├── scripts/
│   ├── setup_database.py         # Initialize PostgreSQL
│   ├── download_models.py        # Download required models
│   └── run_analysis.py           # CLI analysis tool
│
├── data/
│   ├── videos/                   # Input videos
│   ├── outputs/                  # Analysis outputs
│   └── models/                   # Downloaded models
│
└── logs/                         # Application logs
```

---

## Implementation Phases

### Phase 1: Foundation (Days 1-2)
- Project structure setup
- Configuration system
- Database schema and connection
- Basic logging
- Environment setup

### Phase 2: Detection Pipeline (Days 3-5)
- Integrate Roboflow Sports
- Player detection
- Ball detection
- Team classification
- Pitch detection

### Phase 3: Tracking System (Days 6-7)
- ByteTrack integration
- Persistent player IDs
- Ball tracking with occlusion handling

### Phase 4: Analytics Engine (Days 8-10)
- Event detection
- Possession calculation
- Formation analysis
- Player/team metrics

### Phase 5: Visualization (Days 11-12)
- Real-time overlay display
- Heatmaps
- Pitch visualizations

### Phase 6: GUI Application (Days 13-15)
- Main window
- Video player
- Analysis controls
- Results display

### Phase 7: Reports & Polish (Days 16-17)
- Report generation
- Export formats
- Bug fixes
- Performance optimization

---

## Claude Code Instructions

To run this project in Claude Code:

1. **Clone/Initialize Repository**
   ```bash
   cd /path/to/your/workspace
   git init soccer_film_analysis
   cd soccer_film_analysis
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Database**
   ```bash
   # Make sure PostgreSQL is running
   python scripts/setup_database.py
   ```

5. **Download Models**
   ```bash
   python scripts/download_models.py
   ```

6. **Run Application**
   ```bash
   python -m src.gui.main_window
   ```

---

## Database Schema

### Tables

1. **games** - Game metadata
2. **teams** - Team information
3. **players** - Player data with jersey numbers
4. **tracking_data** - Frame-by-frame player positions
5. **events** - Detected game events
6. **player_metrics** - Per-game player statistics
7. **team_metrics** - Per-game team statistics
8. **formations** - Formation snapshots
9. **analysis_sessions** - Analysis run metadata

---

## Key Design Decisions

1. **Use Roboflow Sports as foundation** - Proven soccer detection models
2. **Supervision for tracking** - ByteTrack is industry standard
3. **PyQt6 over tkinter** - More modern, better threading support
4. **SQLAlchemy ORM** - Cleaner database code
5. **Modular architecture** - Easy to extend and test
6. **Progress tracking** - User knows what's happening
7. **Chunked processing** - Handle long videos reliably

---

## Success Criteria

- [ ] Correctly identifies 90%+ of players
- [ ] Team classification accuracy >85%
- [ ] Ball tracking with <20% lost frames
- [ ] Event detection accuracy >70%
- [ ] Processing time <60 minutes for 90-min game
- [ ] Real-time visualization at 15+ FPS
- [ ] Clean, usable reports

