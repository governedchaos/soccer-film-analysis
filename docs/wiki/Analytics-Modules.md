# Analytics Modules

Detailed documentation for each analytics module.

## Formation Detection

**File:** `src/analysis/formation_detection.py`

### Supported Formations

| Formation | Description |
|-----------|-------------|
| 4-4-2 | Classic two banks of four |
| 4-4-1-1 | 4-4-2 with second striker |
| 4-3-3 | Wide attacking formation |
| 4-3-3 Holding | 4-3-3 with defensive mid |
| 4-2-3-1 | Modern balanced formation |
| 3-5-2 | Three at back, wing backs |
| 3-4-3 | Attacking three at back |
| 5-3-2 | Defensive with wing backs |
| 5-4-1 | Very defensive |
| 4-1-4-1 | Single pivot midfield |
| 4-5-1 | Packed midfield |

### How It Works

1. **Position Normalization**
   - Player positions normalized to 0-1 range
   - Accounts for attacking direction

2. **Template Matching**
   - Each formation has predefined positions
   - Hungarian algorithm finds best player-to-position assignment

3. **Confidence Scoring**
   - Based on average distance to template positions
   - Higher confidence = better match

4. **Smoothing**
   - Formation tracked over time
   - Prevents rapid switching
   - Reports formation changes

### Usage

```python
from src.analysis.formation_detection import FormationDetector

detector = FormationDetector()
result = detector.detect_formation(
    frame=1,
    team_positions=positions,  # List of (x, y) tuples
    team_id=1,
    attacking_direction='right'
)

print(f"Formation: {result.formation.value}")
print(f"Confidence: {result.confidence:.2f}")
```

---

## Expected Goals (xG)

**File:** `src/analysis/expected_goals.py`

### Model Factors

| Factor | Impact |
|--------|--------|
| Distance | Closer = higher xG |
| Angle | Central = higher xG |
| Shot type | Foot > Header |
| Pressure | More defenders = lower xG |
| GK position | Off-line = higher xG |
| Situation | Counter > Open play > Set piece |

### Typical xG Values

| Shot Location | xG |
|---------------|-----|
| 6 yards, central | ~0.49 |
| Penalty spot | ~0.76 |
| 18 yards, central | ~0.15 |
| 25 yards, central | ~0.05 |
| Tight angle, 10 yards | ~0.08 |

### Usage

```python
from src.analysis.expected_goals import ExpectedGoalsModel, ShotContext

model = ExpectedGoalsModel()

context = ShotContext(
    distance_to_goal=11.0,  # meters
    angle_to_goal=45.0,     # degrees
    shot_type='foot',
    defenders_in_path=1,
    goalkeeper_position=(0, 0),
    is_counter_attack=False,
    is_set_piece=False
)

xg = model.calculate_xg(context)
print(f"xG: {xg:.3f}")
```

---

## Space Analysis

**File:** `src/analysis/space_analysis.py`

### Components

#### SpaceCreationAnalyzer
Tracks off-ball movement and space creation.

**Metrics:**
- Run distance
- Space created (area opened)
- Run type (diagonal, vertical, lateral)
- Quality score

#### ThirdManRunDetector
Identifies combination play patterns.

**Pattern:** A passes to B, B passes to C (who made a run)

**Detection criteria:**
- Run started before/during first pass
- Run into space
- Timing synchronization

#### OffBallMovementAnalyzer
Comprehensive movement tracking.

**Tracks:**
- All player movements
- Movement quality
- Team movement patterns

### Usage

```python
from src.analysis.space_analysis import SpaceCreationAnalyzer

analyzer = SpaceCreationAnalyzer()

events = analyzer.analyze_frame(
    frame=frame_number,
    timestamp=time,
    team_positions={'team1': [...], 'team2': [...]},
    ball_position=(x, y),
    team_in_possession=1
)

for event in events:
    print(f"Player {event.player_id}: {event.run_type}")
    print(f"Space created: {event.space_created:.1f} sq units")
```

---

## Tactical Analytics

**File:** `src/analysis/tactical_analytics.py`

### Team Shape

```python
from src.analysis.tactical_analytics import TacticalAnalyzer

analyzer = TacticalAnalyzer()
shape = analyzer.calculate_team_shape(positions)

print(f"Compactness: {shape['compactness']:.1f}")
print(f"Width: {shape['width']:.1f}")
print(f"Length: {shape['length']:.1f}")
print(f"Defensive line: {shape['defensive_line']:.1f}")
```

### Pressing Intensity (PPDA)

```python
ppda = analyzer.calculate_ppda(
    defensive_actions=15,
    opponent_passes=45
)
# PPDA = 45/15 = 3.0 (high pressing)
# Lower PPDA = more intense pressing
```

### Possession

```python
possession = analyzer.calculate_possession(
    team1_frames=540,
    team2_frames=360
)
# Returns: {'team1': 60.0, 'team2': 40.0}
```

---

## Advanced Tactical

**File:** `src/analysis/advanced_tactical.py`

### Duel Statistics

```python
from src.analysis.advanced_tactical import DuelTracker

tracker = DuelTracker()
tracker.record_duel(
    frame=100,
    duel_type='ground',
    winner_team=1,
    location=(50, 30)
)

stats = tracker.get_statistics()
# {'team1': {'ground_won': 5, 'aerial_won': 2}, ...}
```

### Passing Lanes

```python
from src.analysis.advanced_tactical import PassingLaneAnalyzer

analyzer = PassingLaneAnalyzer()
lanes = analyzer.find_open_lanes(
    passer_pos=(30, 40),
    teammates=[(50, 30), (60, 50)],
    opponents=[(40, 35), (55, 45)]
)
# Returns list of open passing lanes with quality scores
```

### Fatigue Detection

```python
from src.analysis.advanced_tactical import FatigueTracker

tracker = FatigueTracker()
# Update with player speeds throughout match
tracker.update(player_id=7, timestamp=2700, speed=8.5)

fatigue = tracker.get_fatigue_level(player_id=7)
# Returns fatigue score 0-1 (1 = very fatigued)
```

---

## Bird's Eye View

**File:** `src/analysis/advanced_tactical.py`

### Homography Transformation

Converts camera view to 2D tactical view.

```python
from src.analysis.advanced_tactical import BirdsEyeTransformer

transformer = BirdsEyeTransformer()

# Set up with corner points from video
transformer.set_source_points([
    (100, 200),  # Top-left corner
    (1800, 200), # Top-right corner
    (1900, 900), # Bottom-right corner
    (50, 900)    # Bottom-left corner
])

# Transform player position
tactical_pos = transformer.transform_point((500, 400))
```

---

## Integration Example

```python
from src.analysis.formation_detection import FormationDetector
from src.analysis.expected_goals import ExpectedGoalsModel
from src.analysis.space_analysis import SpaceCreationAnalyzer
from src.analysis.tactical_analytics import TacticalAnalyzer

# Initialize all analyzers
formation_detector = FormationDetector()
xg_model = ExpectedGoalsModel()
space_analyzer = SpaceCreationAnalyzer()
tactical = TacticalAnalyzer()

# Process each frame
for frame_data in game_frames:
    # Detect formation
    formation = formation_detector.detect_formation(...)

    # Analyze space creation
    space_events = space_analyzer.analyze_frame(...)

    # Calculate team shape
    shape = tactical.calculate_team_shape(...)

    # If shot detected, calculate xG
    if shot_detected:
        xg = xg_model.calculate_xg(shot_context)
```
