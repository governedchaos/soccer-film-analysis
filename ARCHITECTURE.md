# Soccer Film Analysis - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GUI Layer (PyQt6)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │ MainWindow  │  │ VideoWidget │  │ ControlPanel│  │ StatsPanel│  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘  │
└─────────┼────────────────┼────────────────┼───────────────┼────────┘
          │                │                │               │
          ▼                ▼                ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Controller Layer                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ VideoController  │  │AnalysisController│  │  StatsController │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
└───────────┼─────────────────────┼─────────────────────┼────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Core Processing                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    VideoProcessor                            │   │
│  │  ┌───────────┐  ┌────────────┐  ┌────────────────────────┐  │   │
│  │  │ load_video│  │process_sync│  │  process_video_async   │  │   │
│  │  └───────────┘  └────────────┘  └────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Detection Pipeline                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  SoccerDetector │  │EnhancedDetector │  │  PitchDetector  │     │
│  │  (YOLOv8)       │  │  (Filtering)    │  │  (Boundaries)   │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │               │
│           ▼                    ▼                    ▼               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ TeamClassifier  │  │PossessionCalc   │  │  IDStabilizer   │     │
│  │  (K-means)      │  │                 │  │  (Tracking)     │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Analysis Pipeline                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │
│  │ FormationDet  │  │ SpaceAnalysis │  │ TacticalAnalytics     │   │
│  └───────────────┘  └───────────────┘  └───────────────────────┘   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │
│  │  ExpectedGoals│  │  HeatmapGen   │  │ AdvancedAnalytics     │   │
│  └───────────────┘  └───────────────┘  └───────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Data Persistence                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   PostgreSQL    │  │  JSON Export    │  │   Video Cache   │     │
│  │   (SQLAlchemy)  │  │                 │  │                 │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### GUI Layer (`src/gui/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| MainWindow | `main_window.py` | Main application window, menu bar, layout |
| VideoWidget | `widgets/video_widget.py` | Video frame display with detection overlays |
| ControlPanel | `widgets/control_panel.py` | Play/pause, analysis controls, depth selector |
| StatsPanel | `widgets/stats_panel.py` | Real-time statistics display |
| LogPanel | `widgets/log_panel.py` | Debug log viewer |

### Core Processing (`src/core/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| VideoProcessor | `video_processor.py` | Main orchestrator for video analysis |
| BatchFrameReader | `video_processor.py` | Efficient batched frame reading |
| ThreadedVideoProcessor | `video_processor.py` | Async wrapper with callbacks |

### Detection Pipeline (`src/detection/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| SoccerDetector | `detector.py` | YOLOv8-based person/ball detection |
| EnhancedDetector | `enhanced_detector.py` | Pitch filtering, referee detection |
| PitchDetector | `pitch_detector.py` | Field boundary and goal detection |
| TeamClassifier | `detector.py` | K-means team color classification |
| PossessionCalculator | `detector.py` | Ball possession tracking |
| IDStabilizer | `tracking_persistence.py` | Tracking ID consistency |

### Analysis Pipeline (`src/analysis/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| AnalysisPipeline | `pipeline.py` | Main analysis orchestrator |
| FormationDetector | `formation_detection.py` | Auto-detect team formations |
| SpaceAnalyzer | `space_analysis.py` | Space creation analysis |
| TacticalAnalytics | `tactical_analytics.py` | Pressing, zones, set pieces |
| ExpectedGoals | `expected_goals.py` | xG model |
| HeatmapGenerator | `advanced_analytics.py` | Player heatmaps |
| GamePeriodDetector | `game_periods.py` | Detect halves, stoppages |

### Data Layer (`src/database/`, `src/data/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| Models | `database/models.py` | SQLAlchemy ORM (Game, Team, Player, etc.) |
| TeamDatabase | `data/team_database.py` | Team/player info management |
| CloudSync | `data/cloud_sync.py` | Google Drive backup |

## Data Flow

### Video Analysis Flow

```
1. User loads video
   └─> VideoController.load_video()
       └─> VideoProcessor.load_video()
           └─> cv2.VideoCapture opens video
           └─> VideoInfo extracted

2. User starts analysis
   └─> AnalysisController.start_analysis()
       └─> VideoProcessor.process_video_async()
           └─> Creates worker thread
           └─> For each frame batch:
               └─> EnhancedDetector.detect_frame()
                   └─> YOLO inference
                   └─> PitchDetector filters bounds
                   └─> TeamClassifier assigns teams
               └─> AnalysisPipeline.process_frame()
               └─> Emit frame_analyzed signal
               └─> Save to database (batched)

3. UI updates
   └─> frame_analyzed signal received
       └─> draw_detections() renders frame
       └─> VideoWidget.display_frame()
       └─> StatsPanel.update_stats()
```

### Detection Data Structures

```python
FrameDetections
├── frame_number: int
├── timestamp_seconds: float
├── players: List[PlayerDetection]
│   ├── bbox: (x1, y1, x2, y2)
│   ├── confidence: float
│   ├── tracker_id: int
│   ├── team_id: int (0=home, 1=away, -1=unknown)
│   └── dominant_color: (r, g, b)
├── referees: List[PlayerDetection]
├── goalkeepers: List[PlayerDetection]
├── ball: BallDetection
│   ├── bbox: (x1, y1, x2, y2)
│   ├── confidence: float
│   └── center: (x, y)
└── possession_team: int
```

## Key Algorithms

### Color Extraction (Histogram-based)
```
1. Extract jersey ROI from detection bbox
2. Convert to HSV, filter grass/skin/shadows
3. Quantize to 8x8x8 color bins (512 total)
4. Find most common bin
5. Return mean color of dominant bin
```

### Team Classification (K-means)
```
1. Fit K-means on calibration colors
2. For new detections:
   a. Check cache by tracker_id
   b. If miss, predict cluster
   c. Cache result for future frames
```

### Pitch Boundary Detection
```
1. HSV grass segmentation (H: 30-90, S > 40)
2. Find largest contour
3. Compute convex hull
4. Optional: Hough lines for field markings
5. Estimate goal areas (outer 12% of width)
```

## Configuration

### Settings (`config/settings.py`)

| Category | Key Settings |
|----------|--------------|
| Detection | `player_confidence_threshold`, `ball_confidence_threshold` |
| Processing | `max_processing_threads`, `enable_gpu`, `gpu_memory_limit_gb` |
| Analysis | `frame_sample_quick`, `frame_sample_standard`, `frame_sample_deep` |
| Database | `database_url`, `db_host`, `db_port` |

### Analysis Depth Levels

| Level | Frame Rate | Features |
|-------|------------|----------|
| Quick | Every 5th | Basic detection, possession |
| Standard | Every 2nd | + Formations, pressing |
| Deep | Every frame | + Space analysis, xG, fatigue |

## Threading Model

```
Main Thread (GUI)
├── PyQt6 event loop
├── Signal/slot connections
└── UI updates

Worker Thread (Analysis)
├── Frame reading
├── YOLO inference
├── Analysis pipeline
└── Emits signals to main thread

GPU (if available)
├── YOLO batch inference
└── CUDA memory management
```

## Error Handling

### Exception Hierarchy (`src/exceptions.py`)

```
SoccerAnalysisError (base)
├── VideoError
│   ├── VideoLoadError
│   ├── VideoFrameError
│   └── VideoProcessingError
├── DetectionError
│   ├── ModelLoadError
│   ├── DetectionFailedError
│   └── ColorExtractionError
├── AnalysisError
│   ├── CalibrationError
│   └── FormationDetectionError
├── DatabaseError
│   ├── SessionNotFoundError
│   └── PersistenceError
├── ConfigurationError
│   └── InvalidConfigError
└── ExportError
    └── ExportFailedError
```

## Performance Optimizations

1. **Color Cache**: LRU cache (max 30 entries) for tracked players
2. **Team Cache**: Avoids re-classification for same tracker IDs
3. **Histogram Color**: O(n) vs K-means O(n*k*i) per frame
4. **Ball History**: Deque with maxlen for O(1) trimming
5. **Batch Inference**: GPU processes multiple frames together
6. **Database Batching**: Bulk inserts every 100 frames

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_integration.py -v
```

## Directory Structure

```
soccer_film_analysis/
├── config/                 # Configuration
│   ├── settings.py        # Pydantic settings
│   └── user/              # User preferences
├── src/
│   ├── core/              # Video processing
│   ├── detection/         # YOLO + classification
│   ├── analysis/          # Tactical analysis
│   ├── gui/               # PyQt6 interface
│   ├── database/          # SQLAlchemy models
│   ├── data/              # Team database, cloud
│   └── exceptions.py      # Custom exceptions
├── tests/                 # Pytest suite
├── docs/wiki/             # Documentation
├── scripts/               # CLI tools
└── main.py               # Entry point
```
