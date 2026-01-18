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
