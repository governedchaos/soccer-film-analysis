# Frequently Asked Questions

## General

### Do I need an API key?
**No.** All detection runs locally using YOLOv8 models. No API keys, subscriptions, or internet connection required after initial setup.

### What video formats are supported?
MP4, AVI, MOV, and MKV. MP4 with H.264 encoding is recommended for best compatibility.

### How long does analysis take?
Depends on video length, model size, and hardware:
- Quick analysis: ~10-15 min per 90-min game
- Standard analysis: ~30-45 min per 90-min game
- Deep analysis: ~45-60 min per 90-min game
- GPU acceleration makes it 3-5x faster

### Does it work offline?
Yes, completely offline after initial setup. Models are downloaded once and stored locally.

---

## Detection

### How accurate is player detection?
With good video quality (1080p+, stable camera), expect:
- Player detection: 90%+ accuracy
- Team classification: 85%+ accuracy
- Ball tracking: 80%+ (varies with ball visibility)

### Why isn't the ball detected?
Ball detection is challenging because:
- Balls are small relative to frame
- Fast movement causes blur
- Similar colors to field markings

Tips:
- Lower `BALL_CONFIDENCE_THRESHOLD` to 0.1
- White/orange balls work best
- Higher resolution video helps

### Can it detect jersey numbers?
Not yet. Jersey number OCR is planned for a future release.

### Does it work with any camera angle?
Works best with:
- Wide/tactical view (sees most of field)
- Stable camera (tripod or broadcast quality)
- Elevated position

Challenging angles:
- Close-up/player-following shots
- Very low angles
- Rapidly moving cameras

---

## Teams & Players

### How does team classification work?
1. Extracts jersey color from each detected player
2. Uses K-means clustering to group similar colors
3. Assigns to home/away based on color clusters
4. Maintains consistent assignment using 30-frame majority voting

### Why do players sometimes switch teams?
This can happen when:
- Jerseys have similar colors
- Poor lighting causes color shifts
- Player partially occluded

Solution: Set team colors manually in Settings.

### How does it identify referees?
Looks for typical referee colors:
- Black (main referee)
- Bright yellow (assistant)
- Pink/magenta (assistant)

If referees wear non-standard colors, they might be misclassified.

### Does it track individual players across the whole game?
Yes, using ByteTrack. However:
- IDs may change if player leaves frame for extended time
- Substitutes get new IDs
- Without jersey number OCR, can't link to specific known players

---

## Analytics

### How is formation detected?
1. Player positions normalized to field coordinates
2. Compared against templates for 10+ formations
3. Hungarian algorithm finds best position-to-player mapping
4. Confidence score indicates match quality
5. Smoothed over time to prevent rapid changes

### What is xG (Expected Goals)?
A measure of shot quality. Each shot is assigned a probability (0-1) of becoming a goal based on:
- Distance and angle to goal
- Shot type (foot/header)
- Defensive pressure
- Goalkeeper position

### How accurate is the xG model?
The model uses standard xG factors calibrated to typical values. For high school soccer, actual conversion rates may differ from professional benchmarks.

### What is PPDA?
Passes Per Defensive Action. Measures pressing intensity:
- Low PPDA (3-6): High pressing
- Medium PPDA (7-10): Moderate pressing
- High PPDA (11+): Low pressing

---

## Database

### Do I need PostgreSQL?
No, it's optional. Without it:
- Analysis works normally
- Data isn't saved between sessions
- Can still export PDF reports

With PostgreSQL:
- Save game analysis permanently
- Build player/team database
- Track statistics over time

### Can I use a different database?
Currently only PostgreSQL is supported. SQLite support may be added in the future.

### How do I back up my data?
1. **Database backup:**
```bash
pg_dump soccer_analysis > backup.sql
```

2. **Cloud sync:** Configure Google Drive folder in `.env`

3. **Local backup:** Set `BACKUP_DIR` in `.env`

---

## Hardware

### Do I need a GPU?
No, but it helps significantly:
- **Without GPU:** Works on CPU, slower processing
- **With GPU:** 3-5x faster processing

### What GPU is supported?
- **NVIDIA:** CUDA-capable GPUs (most GeForce/Quadro)
- **Apple Silicon:** M1/M2/M3 via MPS
- **AMD:** Limited support through ROCm

### Minimum system requirements?
- **CPU:** 4+ cores recommended
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 500MB for app, more for videos
- **GPU:** Optional but recommended

### What about Apple Silicon (M1/M2)?
Fully supported! PyTorch MPS backend provides GPU acceleration on Apple Silicon Macs.

---

## Video Quality

### What resolution should my video be?
- **Minimum:** 720p
- **Recommended:** 1080p
- **Ideal:** 1080p or higher

Higher resolution = better detection accuracy

### Does frame rate matter?
- **30 fps:** Works well for most analysis
- **60 fps:** Better for fast action, but slower to process

### Can I use phone recordings?
Yes, with caveats:
- Stabilization helps (use tripod or gimbal)
- Wide angle captures more of the field
- Avoid zooming in and out frequently

---

## Export & Reports

### What can I export?
- **PDF Reports:** Match summary, stats, charts
- **CSV/JSON:** Raw detection and analysis data
- **Screenshots:** Current frame with overlays

### Can I export video clips?
Video clip export is planned for a future release.

### Are reports customizable?
Currently reports use a standard template. Custom templates may be added later.

---

## Troubleshooting

### Where are the logs?
In the `logs/` directory. Set `LOG_LEVEL=DEBUG` in `.env` for verbose logging.

### How do I reset settings?
Delete the `.env` file and copy from `.env.example` again.

### Application crashes on startup
1. Check Python version (3.10+ required)
2. Verify virtual environment is active
3. Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
4. Check logs for specific error

### How do I report a bug?
Open an issue at: https://github.com/governedchaos/soccer-film-analysis/issues

Include:
- Operating system
- Python version
- Error message / stack trace
- Steps to reproduce
