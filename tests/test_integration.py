"""
Integration tests for the Soccer Film Analysis pipeline.

Tests the full flow from video loading through detection to analysis,
using synthetic data when real video is not available.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock, patch, MagicMock

from src.detection.detector import (
    SoccerDetector, TeamClassifier, PossessionCalculator,
    FrameDetections, PlayerDetection, BallDetection
)
from src.core.video_processor import VideoProcessor, BatchFrameReader
from src.analysis.pipeline import (
    AnalysisPipeline, AnalysisPipelineConfig, AnalysisDepthLevel,
    FrameAnalysisResult
)


class SyntheticFrameGenerator:
    """
    Generates synthetic video frames for testing.
    Creates frames with colored rectangles simulating players and ball.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        num_frames: int = 100,
        fps: float = 30.0
    ):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps

        # Team colors (RGB)
        self.home_color = (255, 0, 0)  # Red
        self.away_color = (0, 0, 255)  # Blue

    def generate_frame(self, frame_num: int) -> np.ndarray:
        """Generate a single synthetic frame with grass background"""
        # Green grass background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Forest green (BGR)
        return frame

    def generate_detections(self, frame_num: int) -> FrameDetections:
        """Generate synthetic detections for a frame"""
        detections = FrameDetections(
            frame_number=frame_num,
            timestamp_seconds=frame_num / self.fps
        )

        # Generate 5 home players
        for i in range(5):
            player = PlayerDetection(
                bbox=(100 + i * 150, 200, 150 + i * 150, 400),
                confidence=0.85,
                class_id=0,
                class_name="person",
                tracker_id=i,
                dominant_color=self.home_color,
                team_id=0
            )
            detections.players.append(player)

        # Generate 5 away players
        for i in range(5):
            player = PlayerDetection(
                bbox=(100 + i * 150, 500, 150 + i * 150, 700),
                confidence=0.82,
                class_id=0,
                class_name="person",
                tracker_id=10 + i,
                dominant_color=self.away_color,
                team_id=1
            )
            detections.players.append(player)

        # Generate ball (alternates position based on frame for possession changes)
        ball_x = 500 + (frame_num % 50) * 10
        ball_y = 300 if frame_num % 100 < 50 else 600  # Near home or away
        detections.ball = BallDetection(
            bbox=(ball_x, ball_y, ball_x + 20, ball_y + 20),
            confidence=0.75,
            class_id=32,
            class_name="sports ball"
        )

        # Set possession based on ball position
        detections.possession_team = 0 if frame_num % 100 < 50 else 1

        return detections


class TestBatchFrameReader:
    """Tests for BatchFrameReader class"""

    def test_batch_reader_initialization(self):
        """Test batch reader initialization"""
        mock_cap = Mock()
        reader = BatchFrameReader(mock_cap, batch_size=8, frame_sample_rate=2)
        assert reader.batch_size == 8
        assert reader.frame_sample_rate == 2

    def test_read_batch_returns_correct_count(self):
        """Test that read_batch returns correct number of frames"""
        mock_cap = Mock()

        # Mock successful frame reads
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = True

        reader = BatchFrameReader(mock_cap, batch_size=4, frame_sample_rate=1)
        frames, frame_numbers = reader.read_batch(0, 10)

        assert len(frames) == 4
        assert len(frame_numbers) == 4
        assert frame_numbers == [0, 1, 2, 3]

    def test_read_batch_respects_sample_rate(self):
        """Test that batch reader respects frame sample rate"""
        mock_cap = Mock()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.set.return_value = True

        reader = BatchFrameReader(mock_cap, batch_size=4, frame_sample_rate=3)
        frames, frame_numbers = reader.read_batch(0, 20)

        assert len(frames) == 4
        assert frame_numbers == [0, 3, 6, 9]  # Every 3rd frame


class TestAnalysisPipeline:
    """Tests for AnalysisPipeline class"""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with default config"""
        pipeline = AnalysisPipeline()
        assert pipeline.config is not None
        assert pipeline._frame_count == 0

    def test_pipeline_config_from_depth(self):
        """Test pipeline config creation from depth level"""
        # MINIMAL disables most features
        minimal = AnalysisPipelineConfig.from_depth(AnalysisDepthLevel.MINIMAL)
        assert minimal.detect_formations is False
        assert minimal.analyze_pressing is False
        assert minimal.track_possession is True  # Always on

        # STANDARD enables more features
        standard = AnalysisPipelineConfig.from_depth(AnalysisDepthLevel.STANDARD)
        assert standard.detect_formations is True
        assert standard.analyze_pressing is True
        assert standard.analyze_space_creation is False

        # COMPREHENSIVE enables everything
        comprehensive = AnalysisPipelineConfig.from_depth(AnalysisDepthLevel.COMPREHENSIVE)
        assert comprehensive.detect_formations is True
        assert comprehensive.analyze_space_creation is True
        assert comprehensive.detect_fatigue is True

    def test_pipeline_process_frame(self):
        """Test pipeline processing a single frame"""
        generator = SyntheticFrameGenerator()
        pipeline = AnalysisPipeline()

        detections = generator.generate_detections(0)
        result = pipeline.process_frame(detections, fps=30.0)

        assert isinstance(result, FrameAnalysisResult)
        assert result.frame_number == 0

    def test_pipeline_tracks_possession(self):
        """Test that pipeline tracks possession changes"""
        generator = SyntheticFrameGenerator()
        pipeline = AnalysisPipeline()

        # Process multiple frames
        for i in range(50):
            detections = generator.generate_detections(i)
            pipeline.process_frame(detections, fps=30.0)

        summary = pipeline.get_summary()
        # Should have tracked some possession
        assert summary.frames_analyzed == 50

    def test_pipeline_get_summary(self):
        """Test getting analysis summary"""
        generator = SyntheticFrameGenerator()
        pipeline = AnalysisPipeline()

        for i in range(20):
            detections = generator.generate_detections(i)
            pipeline.process_frame(detections, fps=30.0)

        summary = pipeline.get_summary()
        assert summary.frames_analyzed == 20
        assert hasattr(summary, 'possession_home')
        assert hasattr(summary, 'possession_away')


class TestColorCache:
    """Tests for color and team classification caching"""

    def test_color_cache_initialization(self):
        """Test that color cache is initialized"""
        detector = SoccerDetector(model_size="nano")
        stats = detector.get_color_cache_stats()

        assert stats['cache_size'] == 0
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0

    def test_color_cache_reset_on_tracker_reset(self):
        """Test that color cache is cleared when tracker resets"""
        detector = SoccerDetector(model_size="nano")

        # Simulate adding to cache
        detector._color_cache[1] = (255, 0, 0)
        detector._color_cache[2] = (0, 0, 255)
        assert len(detector._color_cache) == 2

        detector.reset_tracker()
        assert len(detector._color_cache) == 0

    def test_team_cache_initialization(self):
        """Test that team cache is initialized"""
        classifier = TeamClassifier()
        stats = classifier.get_cache_stats()

        assert stats['cache_size'] == 0
        assert stats['cache_hits'] == 0

    def test_team_cache_reset(self):
        """Test team cache reset"""
        classifier = TeamClassifier()
        classifier._team_cache[1] = 0
        classifier._team_cache[2] = 1

        classifier.reset_cache()
        assert len(classifier._team_cache) == 0


class TestDetectorBatch:
    """Tests for batch detection functionality"""

    def test_detect_batch_method_exists(self):
        """Test that detect_batch method exists"""
        detector = SoccerDetector(model_size="nano")
        assert hasattr(detector, 'detect_batch')
        assert callable(getattr(detector, 'detect_batch'))

    def test_detect_batch_empty_input(self):
        """Test detect_batch with empty input"""
        detector = SoccerDetector(model_size="nano")
        results = detector.detect_batch([], [], fps=30.0)
        assert results == []

    def test_detect_batch_mismatched_lengths_raises(self):
        """Test that mismatched frame/number lengths raises error"""
        detector = SoccerDetector(model_size="nano")
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        frame_numbers = [0, 1]  # Mismatched length

        with pytest.raises(ValueError):
            detector.detect_batch(frames, frame_numbers, fps=30.0)


class TestVideoProcessorIntegration:
    """Integration tests for VideoProcessor"""

    def test_processor_initialization(self):
        """Test processor initialization creates all components"""
        processor = VideoProcessor()

        assert processor.detector is not None
        assert processor.team_classifier is not None
        assert processor.possession_calculator is not None
        assert processor.frame_detections == {}

    def test_processor_detection_stats(self):
        """Test detection statistics tracking"""
        processor = VideoProcessor()

        # Simulate updating stats
        generator = SyntheticFrameGenerator()
        detections = generator.generate_detections(0)
        processor._update_detection_stats(detections)

        stats = processor.get_detection_stats()
        assert stats['frames_processed'] == 1
        assert stats['total_player_detections'] == 10

    def test_processor_stats_include_cache(self):
        """Test that detection stats include cache information"""
        processor = VideoProcessor()
        stats = processor.get_detection_stats()

        assert 'color_cache' in stats
        assert 'team_cache' in stats


class TestPossessionCalculator:
    """Tests for possession calculation"""

    def test_possession_initialization(self):
        """Test possession calculator initialization"""
        calc = PossessionCalculator()
        home, away = calc.get_possession_percentage()
        assert home == 50.0
        assert away == 50.0

    def test_possession_updates(self):
        """Test possession tracking updates"""
        calc = PossessionCalculator()
        generator = SyntheticFrameGenerator()

        # Process several frames
        for i in range(10):
            detections = generator.generate_detections(i)
            calc.calculate_possession(detections)

        home, away = calc.get_possession_percentage()
        assert home + away == pytest.approx(100.0, abs=0.1)


class TestFullPipelineIntegration:
    """Full end-to-end integration tests"""

    def test_detection_to_analysis_flow(self):
        """Test complete flow from detection to analysis"""
        # Create components
        classifier = TeamClassifier()
        classifier.set_team_colors((255, 0, 0), (0, 0, 255))

        possession_calc = PossessionCalculator()
        pipeline = AnalysisPipeline()

        # Generate and process frames
        generator = SyntheticFrameGenerator()

        for i in range(30):
            detections = generator.generate_detections(i)

            # Classify players
            classifier.classify_players(detections.players)

            # Calculate possession
            possession_calc.calculate_possession(detections)

            # Run analysis
            result = pipeline.process_frame(detections, fps=30.0)
            assert result is not None

        # Verify summary
        summary = pipeline.get_summary()
        assert summary.frames_analyzed == 30

        home_poss, away_poss = possession_calc.get_possession_percentage()
        assert 0 <= home_poss <= 100
        assert 0 <= away_poss <= 100

    def test_cache_improves_with_tracking(self):
        """Test that caching provides benefit with tracked players"""
        classifier = TeamClassifier()
        classifier.set_team_colors((255, 0, 0), (0, 0, 255))

        generator = SyntheticFrameGenerator()

        # Process multiple frames - same tracker IDs should hit cache
        for i in range(20):
            detections = generator.generate_detections(i)
            classifier.classify_players(detections.players)

        stats = classifier.get_cache_stats()
        # After first frame, subsequent frames should hit cache
        assert stats['cache_hits'] > 0
        assert stats['hit_rate_percent'] > 50  # Should be mostly hits


class TestWithSampleVideo:
    """
    Tests that run with actual sample video files.
    These tests are skipped if no sample video is available.
    """

    @pytest.fixture
    def sample_video(self, project_root):
        """Get sample video path if available"""
        video_dir = project_root / "data" / "videos"
        if not video_dir.exists():
            pytest.skip("No video directory found")

        videos = list(video_dir.glob("*.mp4"))
        if not videos:
            pytest.skip("No sample videos available")

        return videos[0]

    def test_load_real_video(self, sample_video):
        """Test loading an actual video file"""
        processor = VideoProcessor()
        video_info = processor.load_video(str(sample_video))

        assert video_info is not None
        assert video_info.width > 0
        assert video_info.height > 0
        assert video_info.fps > 0
        assert video_info.total_frames > 0

        processor.release()

    def test_process_real_video_frames(self, sample_video):
        """Test processing frames from real video"""
        processor = VideoProcessor()
        processor.load_video(str(sample_video))

        # Process just a few frames
        frame = processor.get_frame(0)
        assert frame is not None
        assert len(frame.shape) == 3  # BGR image

        frame_100 = processor.get_frame(100)
        if processor.video_info.total_frames > 100:
            assert frame_100 is not None

        processor.release()
