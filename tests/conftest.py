"""
Pytest configuration and shared fixtures for Soccer Film Analysis tests.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing (640x480 BGR)"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some green for grass
    frame[:, :] = [34, 139, 34]  # BGR for forest green
    return frame


@pytest.fixture
def sample_player_bbox():
    """Sample player bounding box (x1, y1, x2, y2)"""
    return (100, 100, 150, 250)


@pytest.fixture
def sample_ball_bbox():
    """Sample ball bounding box"""
    return (300, 350, 320, 370)


@pytest.fixture
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_video_path(project_root):
    """Path to sample video (if exists)"""
    video_dir = project_root / "data" / "videos"
    videos = list(video_dir.glob("*.mp4"))
    if videos:
        return videos[0]
    return None


@pytest.fixture
def synthetic_detections():
    """Create synthetic frame detections for testing"""
    from src.detection.detector import FrameDetections, PlayerDetection, BallDetection

    detections = FrameDetections(
        frame_number=0,
        timestamp_seconds=0.0
    )

    # Add some players
    for i in range(5):
        player = PlayerDetection(
            bbox=(100 + i * 100, 200, 150 + i * 100, 400),
            confidence=0.85,
            class_id=0,
            class_name="person",
            tracker_id=i,
            dominant_color=(255, 0, 0),
            team_id=0
        )
        detections.players.append(player)

    for i in range(5):
        player = PlayerDetection(
            bbox=(100 + i * 100, 500, 150 + i * 100, 700),
            confidence=0.82,
            class_id=0,
            class_name="person",
            tracker_id=10 + i,
            dominant_color=(0, 0, 255),
            team_id=1
        )
        detections.players.append(player)

    # Add ball
    detections.ball = BallDetection(
        bbox=(500, 400, 520, 420),
        confidence=0.75,
        class_id=32,
        class_name="sports ball"
    )

    return detections


@pytest.fixture
def team_classifier():
    """Create a pre-configured team classifier"""
    from src.detection.detector import TeamClassifier

    classifier = TeamClassifier()
    classifier.set_team_colors((255, 0, 0), (0, 0, 255))
    return classifier


@pytest.fixture
def analysis_pipeline():
    """Create an analysis pipeline for testing"""
    from src.analysis.pipeline import AnalysisPipeline

    return AnalysisPipeline()
