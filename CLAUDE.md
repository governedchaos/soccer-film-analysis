# CLAUDE.md - Soccer Film Analysis

## Project Overview
Computer vision application for soccer match analysis. Processes game video to detect players/ball, classify teams, and compute tactical analytics. PyQt6 desktop GUI with PostgreSQL persistence.

## Quick Start
```bash
pip install -r requirements.txt
python -m pytest tests/ -v           # Run tests
python scripts/run_analysis.py       # CLI analysis
python -c "from src.gui.main_window import SoccerAnalysisApp"  # GUI (needs display)
```

## Architecture
```
config/settings.py          — Pydantic BaseSettings (env, GPU, thresholds, paths)
src/
  exceptions.py             — Custom exception hierarchy (SoccerAnalysisError base)
  core/video_processor.py   — VideoProcessor (load, batch read, orchestrate pipeline)
  detection/
    detector.py             — SoccerDetector (YOLOv8), TeamClassifier (K-means), PossessionCalculator
    enhanced_detector.py    — EnhancedDetector (pitch filtering, referee detection, tracking)
    pitch_detector.py       — PitchDetector (HSV segmentation, goal area estimation)
    tracking_persistence.py — IDStabilizer (consistent player IDs across frames)
    jersey_ocr.py           — JerseyOCRAnalyzer (EasyOCR/PaddleOCR jersey number reading)
  analysis/
    pipeline.py             — AnalysisPipeline (frame-by-frame orchestrator)
    formation_detection.py  — FormationDetector (10+ templates, Hungarian matching)
    tactical_analytics.py   — PPDA, pressing, zones, set pieces, defensive line
    advanced_tactical.py    — Duels, fatigue, counter-attacks, xT
    expected_goals.py       — ExpectedGoalsModel (shot quality → xG)
    space_analysis.py       — SpaceCreationAnalyzer, ThirdManRunDetector
    advanced_analytics.py   — HeatmapGenerator, PassNetworkAnalyzer, SpeedDistanceTracker
    pdf_report.py           — PDFReportGenerator
    export.py               — AnalysisExporter (JSON, CSV, Excel, Word)
    auto_save.py            — Auto-save functionality
    video_export.py         — Video clip export
    game_periods.py         — Half detection
    game_comparison.py      — Match comparison
    hudl_import.py          — Hudl data import
  gui/
    main_window.py          — SoccerAnalysisApp (QMainWindow)
    controllers/            — VideoController, AnalysisController (MVC pattern)
    widgets/                — VideoWidget, ControlPanel, TimelineWidget, StatsPanel, LogPanel
  database/models.py        — SQLAlchemy ORM (Game, Team, Player, TrackingData, Event, etc.)
  data/
    team_database.py        — Team/player database management
    cloud_sync.py           — Google Drive and local folder backup
  processing/video_cache.py — Frame caching/preprocessing
tests/                      — Pytest suite (conftest fixtures, unit + integration)
```

## Key Patterns
- **MVC**: Controllers separate GUI from business logic
- **Pipeline**: Detection → Analysis → Persistence (frame-by-frame)
- **Async**: Threading for non-blocking video analysis
- **GPU**: YOLOv8 batch inference with CUDA/MPS support; GPUMemoryManager handles cache clearing
- **Caching**: LRU color cache, team cache, video frame cache
- **Config**: Pydantic BaseSettings loads from `.env`, type-validated
- **Exceptions**: Structured hierarchy with context dicts for debugging

## Database
- **Engine**: PostgreSQL (production), SQLite (testing)
- **ORM**: SQLAlchemy 2.0 declarative with connection pooling (QueuePool)
- **Tables**: games, teams, players, tracking_data, events, player_metrics, team_metrics, formations, analysis_sessions
- **Indexes**: Composite indexes on (game_id, frame_number), (player_id, frame_number), (game_id, event_type)

## Detection Pipeline
- **Model**: YOLOv8 (nano/small/medium/large/xlarge) — local .pt files, no API key required
- **Tracking**: ByteTrack via Supervision library
- **Team Classification**: K-means on dominant jersey colors
- **Pitch Detection**: HSV grass segmentation + contour analysis
- **Enhanced**: Pitch boundary filtering, referee detection (black/bright), out-of-bounds exclusion

## Analysis Depth Levels
| Level | Frame Rate | Features |
|-------|-----------|----------|
| MINIMAL | Every 10th | Basic detection, possession |
| STANDARD | Every 5th | + Formations, pressing, zones |
| COMPREHENSIVE | Every 2nd | + Space creation, xG, fatigue, duels |

## Testing
```bash
python -m pytest tests/ -v                    # All tests
python -m pytest tests/test_core.py -v        # Core tests only
python -m pytest tests/test_integration.py -v # Integration tests
python -m pytest tests/ --cov=src --cov-report=html  # With coverage
```

## Conventions
- All settings via Pydantic (type-validated, .env file)
- Custom exceptions with context dicts for debugging
- Logging via Python `logging` module (loguru available)
- Version tracked in `src/__init__.py`
- Tests use synthetic frames/detections (no real video required)
- `pytest.ini` configured with `-v --tb=short`

## Dependencies (Major)
- `ultralytics` — YOLOv8 detection
- `supervision` — ByteTrack tracking
- `opencv-python` — Video/frame processing
- `PyQt6` — Desktop GUI
- `sqlalchemy` + `psycopg2-binary` — Database
- `pydantic-settings` — Configuration
- `torch` — GPU acceleration
- `scikit-learn` — K-means clustering
- `matplotlib`, `plotly`, `mplsoccer` — Visualizations

## Known Limitations
- Single-machine deployment (Phase 1 — local Windows)
- PostgreSQL required for production (SQLite for tests)
- No user authentication
- GPU recommended but not required (CPU fallback)
- Large video files not stored in git (data/videos/ gitignored)
