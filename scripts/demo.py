#!/usr/bin/env python3
"""
Soccer Film Analysis - Quick Demo Script
Demonstrates the detection pipeline on a single frame or short video clip.
Works without GUI - perfect for testing in Claude Code.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from datetime import datetime


def demo_synthetic_detection():
    """
    Demonstrate detection on a synthetic soccer field image.
    Useful when no real video is available.
    """
    print("\n=== Demo: Synthetic Detection ===\n")
    
    try:
        from ultralytics import YOLO
        import supervision as sv
        
        # Create a synthetic soccer field image
        print("Creating synthetic soccer field image...")
        field = create_synthetic_field()
        
        # Load YOLO model (uses general model since no soccer-specific)
        print("Loading YOLOv8 model...")
        model = YOLO("yolov8n.pt")
        
        # Run detection
        print("Running detection...")
        results = model(field, verbose=False)
        
        # Process results with supervision
        detections = sv.Detections.from_ultralytics(results[0])
        print(f"Detected {len(detections)} objects")
        
        # Annotate image
        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated = box_annotator.annotate(field.copy(), detections)
        
        # Save result
        output_path = project_root / "data/outputs/demo_detection.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated image to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_synthetic_field():
    """Create a synthetic soccer field image"""
    # Create green field
    width, height = 1280, 720
    field = np.zeros((height, width, 3), dtype=np.uint8)
    field[:] = (34, 139, 34)  # Forest green (BGR)
    
    # Draw field lines (white)
    white = (255, 255, 255)
    
    # Outer boundary
    cv2.rectangle(field, (50, 50), (width-50, height-50), white, 3)
    
    # Center line
    cv2.line(field, (width//2, 50), (width//2, height-50), white, 2)
    
    # Center circle
    cv2.circle(field, (width//2, height//2), 80, white, 2)
    cv2.circle(field, (width//2, height//2), 5, white, -1)
    
    # Penalty boxes
    # Left penalty box
    cv2.rectangle(field, (50, height//2 - 150), (200, height//2 + 150), white, 2)
    # Right penalty box
    cv2.rectangle(field, (width-200, height//2 - 150), (width-50, height//2 + 150), white, 2)
    
    # Goal boxes
    cv2.rectangle(field, (50, height//2 - 60), (100, height//2 + 60), white, 2)
    cv2.rectangle(field, (width-100, height//2 - 60), (width-50, height//2 + 60), white, 2)
    
    # Add some "players" (colored rectangles as simple representations)
    np.random.seed(42)
    
    # Team 1 (yellow jerseys)
    for _ in range(11):
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)
        cv2.rectangle(field, (x-15, y-30), (x+15, y+30), (0, 255, 255), -1)  # Yellow
        cv2.rectangle(field, (x-15, y-30), (x+15, y+30), (0, 0, 0), 1)  # Border
    
    # Team 2 (red jerseys)
    for _ in range(11):
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)
        cv2.rectangle(field, (x-15, y-30), (x+15, y+30), (0, 0, 255), -1)  # Red
        cv2.rectangle(field, (x-15, y-30), (x+15, y+30), (0, 0, 0), 1)  # Border
    
    # Ball (orange)
    ball_x, ball_y = width // 2 + 50, height // 2 - 30
    cv2.circle(field, (ball_x, ball_y), 10, (0, 165, 255), -1)
    cv2.circle(field, (ball_x, ball_y), 10, (0, 0, 0), 2)
    
    return field


def demo_video_detection(video_path: str, max_frames: int = 30):
    """
    Demonstrate detection on a video file.
    """
    print(f"\n=== Demo: Video Detection ({max_frames} frames) ===\n")
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return False
    
    try:
        from ultralytics import YOLO
        import supervision as sv
        
        # Load model
        print("Loading YOLOv8 model...")
        model = YOLO("yolov8n.pt")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {video_path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total frames: {total_frames}")
        
        # Initialize tracker
        tracker = sv.ByteTrack()
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator()
        
        # Setup output video
        output_path = project_root / "data/outputs/demo_output.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        print(f"\nProcessing {min(max_frames, total_frames)} frames...")
        frame_count = 0
        total_detections = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Track objects
            tracked = tracker.update_with_detections(detections)
            
            # Create labels
            labels = [f"#{t}" for t in tracked.tracker_id] if tracked.tracker_id is not None else []
            
            # Annotate
            annotated = box_annotator.annotate(frame.copy(), tracked)
            annotated = label_annotator.annotate(annotated, tracked, labels=labels)
            
            # Add frame info
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated, f"Objects: {len(tracked)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(annotated)
            
            total_detections += len(tracked)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"  Processed {frame_count}/{max_frames} frames...")
        
        cap.release()
        out.release()
        
        print(f"\nDone! Processed {frame_count} frames")
        print(f"Average detections per frame: {total_detections / frame_count:.1f}")
        print(f"Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_team_classification():
    """
    Demonstrate team color classification using K-means clustering.
    """
    print("\n=== Demo: Team Color Classification ===\n")
    
    try:
        from sklearn.cluster import KMeans
        
        # Create sample jersey colors
        print("Creating sample jersey colors...")
        
        # Simulate extracted colors from players
        # Team 1 (yellow/gold jerseys)
        team1_colors = [
            [255, 255, 0],   # Yellow
            [255, 230, 50],
            [250, 250, 80],
            [245, 240, 30],
            [255, 220, 40],
        ]
        
        # Team 2 (red jerseys)
        team2_colors = [
            [255, 0, 0],     # Red
            [230, 20, 30],
            [255, 50, 40],
            [200, 10, 20],
            [240, 30, 25],
        ]
        
        # Mix them up (as they would be detected)
        all_colors = np.array(team1_colors + team2_colors)
        np.random.shuffle(all_colors)
        
        # Apply K-means clustering
        print("Running K-means clustering...")
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_colors)
        centroids = kmeans.cluster_centers_.astype(int)
        
        print(f"\nDetected team colors:")
        for i, centroid in enumerate(centroids):
            print(f"  Team {i+1}: RGB({centroid[0]}, {centroid[1]}, {centroid[2]})")
        
        # Visualize
        viz_height, viz_width = 100, 400
        viz = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
        
        # Draw team color bars
        viz[:, :viz_width//2] = centroids[0][::-1]  # RGB to BGR
        viz[:, viz_width//2:] = centroids[1][::-1]
        
        # Add labels
        cv2.putText(viz, "Team 1", (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(viz, "Team 2", (280, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save visualization
        output_path = project_root / "data/outputs/team_colors.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), viz)
        print(f"\nSaved team color visualization to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Soccer Film Analysis - Quick Demo"
    )
    parser.add_argument(
        "--demo",
        type=str,
        choices=["synthetic", "video", "teams", "all"],
        default="all",
        help="Which demo to run (default: all)"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file (for video demo)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Max frames to process for video demo (default: 30)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("       SOCCER FILM ANALYSIS - QUICK DEMO")
    print("=" * 60)
    print(f"       Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    success = True
    
    if args.demo in ["synthetic", "all"]:
        success = demo_synthetic_detection() and success
    
    if args.demo in ["teams", "all"]:
        success = demo_team_classification() and success
    
    if args.demo in ["video", "all"]:
        if args.video:
            success = demo_video_detection(args.video, args.frames) and success
        elif args.demo == "video":
            print("\nError: --video path required for video demo")
            success = False
        else:
            print("\n[Skipping video demo - no video path provided]")
            print("  To test with video: --demo video --video path/to/video.mp4")
    
    print("\n" + "=" * 60)
    if success:
        print("  Demo completed successfully!")
    else:
        print("  Some demos failed. Check errors above.")
    print("=" * 60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
