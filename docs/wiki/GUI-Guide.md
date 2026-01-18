# GUI Guide

Complete guide to using the Soccer Film Analysis graphical interface.

## Starting the Application

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Launch GUI
python -m src.gui.main_window
```

## Main Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Menu Bar                                                    │
├─────────────────────────────────────────────────────────────┤
│  Toolbar: [Open] [Play/Pause] [Stop] [Settings]             │
├───────────────────────────────────┬─────────────────────────┤
│                                   │                         │
│                                   │    Statistics Panel     │
│        Video Display              │    - Possession         │
│        with Overlays              │    - Player counts      │
│                                   │    - Shots/Saves        │
│                                   │    - Pass stats         │
│                                   │                         │
├───────────────────────────────────┴─────────────────────────┤
│  Timeline / Progress Bar                                     │
├─────────────────────────────────────────────────────────────┤
│  Status Bar                                                  │
└─────────────────────────────────────────────────────────────┘
```

## Loading a Video

### Method 1: File Menu
1. Click **File** → **Open Video**
2. Navigate to your video file
3. Select and click **Open**

### Method 2: Drag and Drop
- Drag a video file directly onto the application window

### Method 3: Toolbar
- Click the **Open** button in the toolbar

### Supported Formats
- MP4 (recommended)
- AVI
- MOV
- MKV

## Setting Team Colors

For best team classification results, set team colors manually:

1. Click **Settings** → **Team Colors**
2. Use the color picker for **Home Team**
3. Use the color picker for **Away Team**
4. Click **Apply**

**Tip:** Pick the dominant jersey color, avoiding logos and numbers.

## Starting Analysis

### Quick Analysis
- Fastest processing
- Basic detection only
- Good for preview

### Standard Analysis (Recommended)
- Balanced speed and detail
- Formation detection
- Possession tracking

### Deep Analysis
- Most detailed
- All analytics modules
- Takes longest

**To start:**
1. Select analysis depth from dropdown
2. Click **Start Analysis** button
3. Watch progress in status bar

## Video Controls

| Control | Action |
|---------|--------|
| Space | Play/Pause |
| ← / → | Frame step |
| Home | Go to start |
| End | Go to end |
| Mouse wheel | Timeline scrub |

## Statistics Panel

### Possession Bar
- Visual bar showing possession percentage
- Updates in real-time during analysis

### Player Counts
- Home team players detected
- Away team players detected
- Referees detected

### Ball Status
- Ball detected: Yes/No
- Detection confidence
- Position on field

### Shots & Saves
- Shots by each team
- Saves by each team
- xG totals (if calculated)

### Pass Statistics
- Completed passes
- Failed passes
- Pass accuracy percentage

## Detection Overlays

### Bounding Boxes
- **Blue/Red** - Team players (colors match team settings)
- **Yellow** - Referees
- **White** - Ball

### Information Display
- Player track ID
- Team assignment
- Confidence score (optional)

### Toggle Overlays
- **View** → **Show Detections** - Toggle boxes
- **View** → **Show IDs** - Toggle track IDs
- **View** → **Show Ball Trail** - Toggle ball path

## Exporting Results

### PDF Report
1. Complete analysis first
2. Click **File** → **Export Report**
3. Choose save location
4. Report includes:
   - Match summary
   - Possession stats
   - Formation analysis
   - Key events
   - Charts and visualizations

### Data Export
1. Click **File** → **Export Data**
2. Choose format (CSV/JSON)
3. Select data to export:
   - Detections
   - Formations
   - Events
   - Statistics

## Database Features

If PostgreSQL is configured:

### Save Game
- **File** → **Save to Database**
- Enter game metadata (teams, date, etc.)
- Analysis saved for future reference

### Load Previous
- **File** → **Load from Database**
- Browse saved games
- Resume analysis or review results

### Player Database
- Track players across games
- Build season statistics
- Player identification by number (when available)

## Settings

### Detection Settings
- Model size (nano → xlarge)
- Confidence thresholds
- GPU enable/disable

### Display Settings
- Theme (light/dark)
- Overlay colors
- Font sizes

### Performance Settings
- Processing threads
- Frame skip interval
- Cache settings

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open video |
| Ctrl+S | Save to database |
| Ctrl+E | Export report |
| Space | Play/Pause |
| F | Toggle fullscreen |
| D | Toggle detections |
| Esc | Stop analysis |

## Tips for Best Results

1. **Video Quality**
   - Higher resolution = better detection
   - Stable camera = better tracking
   - Good lighting = better color classification

2. **Team Colors**
   - Set manually for best accuracy
   - Ensure teams have contrasting colors
   - Avoid similar colors to referee kit

3. **Processing Speed**
   - Use GPU if available
   - Lower model size for faster preview
   - Use frame skipping for long videos

4. **Memory Usage**
   - Close other applications
   - Use video cache for repeated analysis
   - Consider video compression for very long games
