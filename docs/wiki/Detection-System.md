# Detection System

The detection system uses YOLOv8 models running locally. No API keys or internet connection required.

## Architecture

```
Video Frame
    │
    ▼
┌─────────────────┐
│  YOLOv8 Model   │ ◄── Local model (~25MB)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ByteTrack     │ ◄── Persistent tracking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EnhancedDetector│ ◄── Post-processing
│  - Pitch filter │
│  - Team assign  │
│  - Referee ID   │
│  - Ball fallback│
└────────┬────────┘
         │
         ▼
    Detections
```

## YOLOv8 Models

Models are downloaded automatically on first run.

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| nano | ~3MB | Fastest | Basic | Quick preview |
| small | ~25MB | Fast | Good | **Default** |
| medium | ~50MB | Medium | Better | Better accuracy |
| large | ~85MB | Slow | High | High quality |
| xlarge | ~130MB | Slowest | Best | Maximum accuracy |

Configure in `.env`:
```env
YOLO_MODEL_SIZE=small
```

## Detection Classes

The system detects:
- **Person** (class 0) - Players, referees, others
- **Sports Ball** (class 32) - Soccer ball

## EnhancedDetector Features

### 1. Pitch Boundary Detection

Filters out-of-bounds detections (ball boys, coaches, substitutes).

**How it works:**
1. Grass segmentation using HSV color space
2. White line detection for field boundaries
3. Morphological operations to clean mask
4. Bounding box validation against pitch area

**Configuration:**
```python
GRASS_HSV_LOWER = [30, 30, 30]  # H, S, V
GRASS_HSV_UPPER = [90, 255, 255]
```

### 2. Team Classification

Assigns players to teams based on jersey color.

**Process:**
1. Extract jersey region (upper body)
2. Convert to LAB color space
3. K-means clustering with k=2
4. Assign to home/away based on dominant colors

**Stable Assignment:**
- `TrackedPerson` class maintains history
- Majority voting over 30 frames
- Prevents frame-to-frame flickering

### 3. Referee Detection

Automatically identifies referees.

**Detection criteria:**
- Black jersey (main referee)
- Bright yellow/pink (assistant referees)
- Color distance from team colors

**Auto-detected colors:**
```python
REFEREE_COLORS = [
    (30, 30, 30),    # Black
    (50, 50, 50),    # Dark gray
    (255, 230, 0),   # Bright yellow
    (255, 100, 150)  # Pink
]
```

### 4. Ball Detection Fallback

When YOLO doesn't detect the ball:

**Fallback 1: Color Detection**
- White ball detection in HSV
- Orange ball detection
- Circle detection with Hough transform
- Size validation

**Fallback 2: Interpolation**
- Uses last known position
- Velocity estimation
- Short-term prediction (5 frames max)

## Confidence Thresholds

Adjust detection sensitivity in `.env`:

```env
# Player detection (0.0 - 1.0)
# Lower = more detections, more false positives
PLAYER_CONFIDENCE_THRESHOLD=0.3

# Ball detection (typically lower than players)
BALL_CONFIDENCE_THRESHOLD=0.15

# Pitch line detection
PITCH_CONFIDENCE_THRESHOLD=0.5
```

## Troubleshooting Detection

### Ball Not Detected
1. Lower `BALL_CONFIDENCE_THRESHOLD` to 0.1
2. Check ball color - works best with white/orange
3. High-contrast balls easier to detect
4. Small balls in wide shots are challenging

### Too Many False Positives
1. Raise `PLAYER_CONFIDENCE_THRESHOLD` to 0.4-0.5
2. Enable pitch boundary filtering
3. Check video quality

### Players Switching Teams
1. Set team colors manually in GUI
2. Ensure good jersey color contrast
3. Avoid teams with similar colors

### Referees Counted as Players
- Should auto-filter with EnhancedDetector
- If not working, check referee jersey isn't close to team color

### Ball Boys Detected
- Pitch boundary detection should filter these
- Ensure grass is visible in frame
- Check pitch detection is working (green overlay)

## GPU Acceleration

The system auto-detects GPU availability.

**Check GPU:**
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {torch.backends.mps.is_available()}")
```

**Force CPU:**
```env
ENABLE_GPU=false
```

## Performance Tips

1. **Use appropriate model size** - Small is good balance
2. **Enable GPU** if available - 3-5x faster
3. **Lower resolution** for faster processing
4. **Frame skipping** - Process every nth frame for speed
5. **Video preprocessing** - Use video cache for repeated analysis
