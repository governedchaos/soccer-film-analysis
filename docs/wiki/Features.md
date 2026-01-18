# Features Overview

Soccer Film Analysis provides comprehensive game analysis capabilities. All detection runs locally using YOLOv8 models - no API keys or internet required.

## Core Detection

### Player Detection & Tracking
- **YOLOv8** for initial detection
- **ByteTrack** for persistent tracking across frames
- Automatic ID assignment maintained throughout video
- Handles occlusions and re-appearances

### Team Classification
- **K-means clustering** on jersey colors
- Automatic home/away team separation
- **Persistent assignment** using majority voting over 30 frames
- Prevents frame-to-frame team flickering

### Referee Detection
- Automatically identifies referees by jersey color
- Detects black kits (main referee)
- Detects bright colors (assistant referees)
- Excluded from team player counts

### Ball Tracking
- Primary: YOLO detection
- Fallback: Color-based detection (white/orange balls)
- Position interpolation for brief occlusions
- Ball possession determination

### Pitch Boundary Detection
- Grass color segmentation (HSV)
- White line detection
- Filters out-of-bounds detections
- Excludes ball boys, coaches, substitutes

## Tactical Analytics

### Formation Detection
Automatically detects 10+ formations:
- 4-4-2, 4-4-1-1
- 4-3-3, 4-3-3 (holding)
- 4-2-3-1
- 3-5-2, 3-4-3
- 5-3-2, 5-4-1
- 4-1-4-1, 4-5-1

Features:
- Template matching with Hungarian algorithm
- Confidence scoring
- Formation change detection
- Smoothing over time

### Expected Goals (xG)
Shot quality model considering:
- Distance to goal
- Angle to goal
- Shot type (foot/header)
- Defensive pressure
- Goalkeeper position
- Game situation (open play, counter, set piece)

### Expected Threat (xT)
Ball progression value:
- Grid-based pitch zones
- Value of moving ball between zones
- Identifies high-value progressions

### Pressing Analysis
- **PPDA** (Passes Per Defensive Action)
- High press detection
- Press trigger identification
- Pressing intensity over time

### Team Shape Analysis
- **Compactness** - How tight the team is
- **Width** - Horizontal spread
- **Defensive line height** - How high/deep
- **Length** - Distance between lines

## Advanced Analytics

### Space Creation
- Off-ball movement tracking
- Run type classification
- Space created quantification
- Movement quality scoring

### Third-Man Run Detection
- Identifies A → B → C passing patterns
- Detects combination play
- Movement timing analysis

### Counter-Attack Detection
- Automatic identification
- Transition speed analysis
- Counter success tracking

### Goalkeeper Analysis
- Distribution patterns
- Kick types (short/long)
- Distribution success rate

### Set Piece Analysis
- Corner patterns
- Free kick routines
- Delivery type classification

### Passing Lane Analysis
- Lane blocking detection
- Available passing options
- Lane quality scoring

### Fatigue Detection
- Player speed tracking over time
- Sprint frequency decline
- Substitution timing suggestions

### Opponent Tendency Prediction
- Historical pattern analysis
- Formation preferences
- Set piece tendencies

## Visualization

### Bird's Eye View
- Homography transformation
- 2D tactical view
- Player position mapping

### Real-time Overlays
- Detection bounding boxes
- Team color coding
- Ball position marker
- Formation lines

### Statistics Panel
- Live possession percentage
- Player counts per team
- Shot/save counters
- Pass statistics

## Data Management

### PostgreSQL Database
- Game metadata storage
- Team/player persistence
- Historical statistics
- Event logging

### Export Options
- **PDF Reports** - Match summary with charts
- **Data Export** - CSV/JSON formats
- **Video Clips** - Key moment extraction (planned)

### Cloud Sync
- Local folder backup
- Google Drive integration (optional)

## Performance

| Feature | Performance |
|---------|-------------|
| Processing | 15+ FPS real-time display |
| Full analysis | <60 min for 90-min game |
| Memory | ~2GB for standard analysis |
| GPU | Optional, significantly faster |
