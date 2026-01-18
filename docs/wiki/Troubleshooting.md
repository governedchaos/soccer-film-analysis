# Troubleshooting

Solutions for common issues with Soccer Film Analysis.

## Installation Issues

### Python Version Error

**Problem:** `Python 3.10+ required`

**Solution:**
```bash
# Check your version
python --version

# Install Python 3.10+ from python.org
# Or use pyenv:
pyenv install 3.10.0
pyenv local 3.10.0
```

### Package Installation Fails

**Problem:** `pip install -r requirements.txt` fails

**Solutions:**

1. **Upgrade pip:**
```bash
pip install --upgrade pip setuptools wheel
```

2. **Install Visual C++ Build Tools (Windows):**
   - Download from Microsoft
   - Required for some packages

3. **Install individually:**
```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install PyQt6
```

### PyQt6 Won't Install

**Problem:** PyQt6 installation errors

**Solutions:**

**Linux:**
```bash
sudo apt-get install python3-pyqt6
# Or
pip install PyQt6 --no-cache-dir
```

**macOS:**
```bash
brew install pyqt6
pip install PyQt6
```

---

## Application Won't Start

### Import Error

**Problem:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
```bash
# Ensure virtual environment is active
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Verify
which python  # Should show venv path

# Reinstall
pip install -r requirements.txt
```

### Qt Platform Error (Linux)

**Problem:** `qt.qpa.plugin: Could not load the Qt platform plugin`

**Solution:**
```bash
# Install dependencies
sudo apt-get install libxcb-xinerama0 libxcb-cursor0

# Set platform
export QT_QPA_PLATFORM=xcb
python -m src.gui.main_window
```

### Display Error (Headless Server)

**Problem:** `Cannot connect to X server`

**Solution:**
```bash
# Use CLI mode instead
python scripts/run_analysis.py video.mp4 --depth standard

# Or set up virtual display
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99
```

---

## Video Issues

### Video Won't Load

**Problem:** Video file doesn't open

**Solutions:**

1. **Check format:**
```bash
# Verify OpenCV can read it
python -c "
import cv2
cap = cv2.VideoCapture('video.mp4')
print(f'Opened: {cap.isOpened()}')
print(f'Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
cap.release()
"
```

2. **Convert video:**
```bash
# Convert to H.264 MP4
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
```

3. **Install codecs:**
```bash
# Linux
sudo apt-get install ffmpeg libavcodec-extra

# Windows - install K-Lite Codec Pack
```

### Video Plays But No Detection

**Problem:** Video displays but nothing detected

**Solutions:**

1. **Lower confidence threshold:**
```env
PLAYER_CONFIDENCE_THRESHOLD=0.2
BALL_CONFIDENCE_THRESHOLD=0.1
```

2. **Check model download:**
```bash
# Models should be in data/models or ~/.cache/ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
```

---

## Detection Issues

### Ball Not Detected

**Problem:** Ball rarely or never detected

**Solutions:**

1. Lower threshold in `.env`:
```env
BALL_CONFIDENCE_THRESHOLD=0.1
```

2. Ball detection works best with:
   - White or orange balls
   - Good contrast with grass
   - Ball not too small in frame

3. The fallback system will try:
   - Color-based detection
   - Position interpolation

### Too Many False Detections

**Problem:** Detecting non-players (crowd, coaches, etc.)

**Solutions:**

1. Raise threshold:
```env
PLAYER_CONFIDENCE_THRESHOLD=0.4
```

2. Enable pitch boundary filtering:
   - Should be automatic with EnhancedDetector
   - Requires visible grass/lines

### Players Constantly Switching Teams

**Problem:** Player colors flickering between teams

**Solutions:**

1. **Set team colors manually** in Settings
2. Ensure jerseys have distinct colors
3. The system uses 30-frame majority voting
4. Check video quality - blur affects color detection

### Referees Counted as Players

**Problem:** Referee showing as team player

**Solutions:**

- EnhancedDetector should auto-filter
- Check referee isn't wearing team-similar colors
- Verify referee colors are detected:
  - Black
  - Bright yellow
  - Pink/magenta

---

## Performance Issues

### Slow Processing

**Problem:** Analysis taking too long

**Solutions:**

1. **Enable GPU:**
```env
ENABLE_GPU=true
```

Check if working:
```python
import torch
print(torch.cuda.is_available())  # NVIDIA
print(torch.backends.mps.is_available())  # Apple Silicon
```

2. **Use smaller model:**
```env
YOLO_MODEL_SIZE=nano
```

3. **Reduce threads:**
```env
MAX_PROCESSING_THREADS=2
```

4. **Use frame skipping** (Quick analysis mode)

### High Memory Usage

**Problem:** Application using too much RAM

**Solutions:**

1. Close other applications
2. Use smaller model size
3. Process shorter video segments
4. Enable video caching (preprocesses in background)

### GPU Not Detected

**Problem:** `CUDA available: False`

**Solutions:**

**NVIDIA:**
```bash
# Check driver
nvidia-smi

# Install CUDA toolkit
# Then reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Apple Silicon:**
```bash
# MPS should work automatically with PyTorch 1.12+
pip install --upgrade torch torchvision
```

---

## Database Issues

### Connection Failed

**Problem:** `Could not connect to PostgreSQL`

**Solutions:**

1. **Check PostgreSQL running:**
```bash
pg_isready -h localhost -p 5432
```

2. **Verify credentials:**
```bash
psql -h localhost -U postgres -d soccer_analysis
```

3. **Check .env file:**
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=soccer_analysis
DB_USER=postgres
DB_PASSWORD=your_password
```

### Tables Don't Exist

**Problem:** `relation "games" does not exist`

**Solution:**
```bash
python scripts/setup_database.py
```

### Permission Denied

**Problem:** Database permission error

**Solution:**
```sql
-- In psql as superuser
GRANT ALL PRIVILEGES ON DATABASE soccer_analysis TO postgres;
GRANT ALL ON ALL TABLES IN SCHEMA public TO postgres;
```

---

## Getting More Help

1. **Check logs:**
   - Located in `logs/` directory
   - Set `LOG_LEVEL=DEBUG` for verbose output

2. **Run diagnostics:**
```bash
python -c "
import cv2
import torch
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication

print(f'OpenCV: {cv2.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print('All OK!')
"
```

3. **Open an issue:**
   - https://github.com/governedchaos/soccer-film-analysis/issues
   - Include: OS, Python version, error message, steps to reproduce
