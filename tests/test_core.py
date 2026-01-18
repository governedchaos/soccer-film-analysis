"""
Tests for core video processing module.
"""

import pytest
from pathlib import Path


class TestVideoProcessor:
    """Tests for VideoProcessor class"""

    def test_video_processor_import(self):
        """Test that video processor can be imported"""
        from src.core import VideoProcessor, VideoInfo, AnalysisProgress
        assert VideoProcessor is not None
        assert VideoInfo is not None
        assert AnalysisProgress is not None

    def test_video_processor_initialization(self):
        """Test video processor initialization"""
        from src.core import VideoProcessor
        processor = VideoProcessor()
        assert processor.video_info is None
        assert processor._detection_stats is not None

    def test_detection_stats_initial(self):
        """Test initial detection statistics"""
        from src.core import VideoProcessor
        processor = VideoProcessor()
        stats = processor.get_detection_stats()

        assert stats["frames_processed"] == 0
        assert stats["ball_detection_rate"] == 0.0
        assert "home_team_detections" in stats
        assert "away_team_detections" in stats

    def test_detection_stats_reset(self):
        """Test detection stats reset"""
        from src.core import VideoProcessor
        processor = VideoProcessor()
        processor._detection_stats["frames_processed"] = 100
        processor.reset_detection_stats()
        assert processor._detection_stats["frames_processed"] == 0


class TestSettings:
    """Tests for settings configuration"""

    def test_settings_import(self):
        """Test that settings can be imported"""
        from config import settings
        assert settings is not None

    def test_device_detection(self):
        """Test device detection"""
        from config import settings
        device = settings.get_device()
        assert device in ["cpu", "cuda", "mps"]

    def test_device_info(self):
        """Test device info retrieval"""
        from config import settings
        info = settings.get_device_info()

        assert "device" in info
        assert "cuda_available" in info
        assert "torch_version" in info
        assert info["device"] in ["cpu", "cuda", "mps"]

    def test_frame_sample_rates(self):
        """Test frame sample rate configuration"""
        from config import settings, AnalysisDepth

        quick_rate = settings.get_frame_sample_rate(AnalysisDepth.QUICK)
        standard_rate = settings.get_frame_sample_rate(AnalysisDepth.STANDARD)
        deep_rate = settings.get_frame_sample_rate(AnalysisDepth.DEEP)

        # Quick should have highest skip rate (process fewer frames)
        assert quick_rate > standard_rate
        assert standard_rate > deep_rate
