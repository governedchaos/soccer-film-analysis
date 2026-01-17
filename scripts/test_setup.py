#!/usr/bin/env python3
"""
Soccer Film Analysis - Setup & Installation Test
Tests that all dependencies are installed and configured correctly.
Run this script to verify your installation before running the full app.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_status(name, success, details=""):
    """Print status with color indicators"""
    status = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{status}{reset} {name}" + (f" - {details}" if details else ""))

def test_core_imports():
    """Test core Python package imports"""
    print_header("Testing Core Python Imports")
    
    packages = [
        ("numpy", "NumPy - Numerical computing"),
        ("cv2", "OpenCV - Computer vision"),
        ("PIL", "Pillow - Image processing"),
        ("torch", "PyTorch - Deep learning"),
        ("sklearn", "Scikit-learn - Machine learning"),
    ]
    
    all_success = True
    for module, desc in packages:
        try:
            __import__(module)
            print_status(desc, True)
        except ImportError as e:
            print_status(desc, False, str(e))
            all_success = False
    
    return all_success

def test_detection_imports():
    """Test detection-specific imports"""
    print_header("Testing Detection Libraries")
    
    packages = [
        ("ultralytics", "Ultralytics - YOLOv8 detection"),
        ("supervision", "Supervision - Object tracking"),
    ]
    
    all_success = True
    for module, desc in packages:
        try:
            __import__(module)
            print_status(desc, True)
        except ImportError as e:
            print_status(desc, False, str(e))
            all_success = False
    
    # Special check for roboflow/inference (optional)
    try:
        import inference
        print_status("Roboflow Inference - Soccer detection models", True)
    except ImportError:
        print_status("Roboflow Inference (optional)", False, "pip install inference")
        # Not critical for basic functionality
    
    return all_success

def test_database():
    """Test database connectivity"""
    print_header("Testing Database")
    
    try:
        from config import settings
        print_status("Configuration loaded", True, f"DB type: {settings.db_type}")
        print(f"       Connection: {settings.db_connection_string[:50]}...")
        
        from src.database.models import init_database, get_engine
        
        # Test engine creation
        engine = get_engine()
        print_status("Database engine created", True)
        
        # Initialize tables
        init_database()
        print_status("Database tables initialized", True)
        
        return True
        
    except Exception as e:
        print_status("Database setup", False, str(e))
        return False

def test_config():
    """Test configuration loading"""
    print_header("Testing Configuration")
    
    try:
        from config import settings, AnalysisDepth
        
        print_status("Settings loaded", True)
        print(f"       Log level: {settings.log_level}")
        print(f"       Analysis depth: {settings.default_analysis_depth}")
        print(f"       GPU enabled: {settings.enable_gpu}")
        print(f"       Device: {settings.get_device()}")
        
        # Check directories
        dirs = [
            ("Video dir", settings.get_video_dir()),
            ("Output dir", settings.get_output_dir()),
            ("Models dir", settings.get_models_dir()),
            ("Logs dir", settings.get_logs_dir()),
        ]
        
        for name, path in dirs:
            exists = path.exists()
            print_status(f"{name}: {path}", exists)
        
        return True
        
    except Exception as e:
        print_status("Configuration", False, str(e))
        return False

def test_gpu():
    """Test GPU availability"""
    print_header("Testing GPU Availability")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print_status("CUDA (NVIDIA GPU)", cuda_available, 
                     f"Device: {torch.cuda.get_device_name(0)}" if cuda_available else "Not available")
        
        try:
            mps_available = torch.backends.mps.is_available()
            print_status("MPS (Apple Silicon)", mps_available)
        except:
            print_status("MPS (Apple Silicon)", False, "Not available")
        
        device = "cuda" if cuda_available else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"       Using device: {device}")
        
        return True
        
    except Exception as e:
        print_status("GPU detection", False, str(e))
        return False

def test_yolo_detection():
    """Test basic YOLO detection capability"""
    print_header("Testing YOLO Detection")
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Load a pretrained model
        print("       Loading YOLOv8n model (first time may download)...")
        model = YOLO("yolov8n.pt")  # Nano model for quick testing
        print_status("YOLOv8 model loaded", True)
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run detection
        results = model(dummy_image, verbose=False)
        print_status("Detection test passed", True, f"Results: {len(results)} frame(s)")
        
        return True
        
    except Exception as e:
        print_status("YOLO detection", False, str(e))
        return False

def test_supervision_tracking():
    """Test Supervision library tracking"""
    print_header("Testing Supervision Tracking")
    
    try:
        import supervision as sv
        import numpy as np
        
        # Create dummy detections
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 200, 200], [300, 300, 400, 400]]),
            confidence=np.array([0.8, 0.9]),
            class_id=np.array([0, 0])
        )
        print_status("Detections created", True)
        
        # Create tracker
        tracker = sv.ByteTrack()
        tracked = tracker.update_with_detections(detections)
        print_status("ByteTrack tracker working", True, f"Tracked {len(tracked)} objects")
        
        # Test annotators
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box_annotator = sv.BoxAnnotator()
        annotated = box_annotator.annotate(frame.copy(), detections)
        print_status("Box annotation working", True)
        
        return True
        
    except Exception as e:
        print_status("Supervision tracking", False, str(e))
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 60)
    print("       SOCCER FILM ANALYSIS - INSTALLATION TEST")
    print("=" * 60)
    
    results = {}
    
    results["Core imports"] = test_core_imports()
    results["Detection imports"] = test_detection_imports()
    results["Configuration"] = test_config()
    results["Database"] = test_database()
    results["GPU"] = test_gpu()
    results["YOLO detection"] = test_yolo_detection()
    results["Supervision tracking"] = test_supervision_tracking()
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        print_status(name, success)
    
    print("\n" + "-" * 60)
    
    if passed == total:
        print(f"  \033[92mAll {total} tests passed! Your installation is ready.\033[0m")
        print("\n  Next steps:")
        print("  1. Place a video in: data/videos/")
        print("  2. Run: python scripts/run_analysis.py data/videos/your_video.mp4")
        return 0
    else:
        print(f"  \033[91m{passed}/{total} tests passed. Please fix the issues above.\033[0m")
        print("\n  Common fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
