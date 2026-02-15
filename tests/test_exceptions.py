"""
Tests for custom exception hierarchy.
"""

import pytest
from src.exceptions import (
    SoccerAnalysisError,
    VideoError,
    VideoLoadError,
    VideoFrameError,
    VideoProcessingError,
    DetectionError,
    ModelLoadError,
    DetectionFailedError,
    ColorExtractionError,
    AnalysisError,
    CalibrationError,
    FormationDetectionError,
    DatabaseError,
    SessionNotFoundError,
    PersistenceError,
    ConfigurationError,
    InvalidConfigError,
    ExportError,
    ExportFailedError,
)


class TestExceptionHierarchy:
    """Test the exception class hierarchy."""

    def test_base_exception_is_exception(self):
        assert issubclass(SoccerAnalysisError, Exception)

    def test_video_errors_inherit_base(self):
        assert issubclass(VideoError, SoccerAnalysisError)
        assert issubclass(VideoLoadError, VideoError)
        assert issubclass(VideoFrameError, VideoError)
        assert issubclass(VideoProcessingError, VideoError)

    def test_detection_errors_inherit_base(self):
        assert issubclass(DetectionError, SoccerAnalysisError)
        assert issubclass(ModelLoadError, DetectionError)
        assert issubclass(DetectionFailedError, DetectionError)
        assert issubclass(ColorExtractionError, DetectionError)

    def test_analysis_errors_inherit_base(self):
        assert issubclass(AnalysisError, SoccerAnalysisError)
        assert issubclass(CalibrationError, AnalysisError)
        assert issubclass(FormationDetectionError, AnalysisError)

    def test_database_errors_inherit_base(self):
        assert issubclass(DatabaseError, SoccerAnalysisError)
        assert issubclass(SessionNotFoundError, DatabaseError)
        assert issubclass(PersistenceError, DatabaseError)

    def test_config_errors_inherit_base(self):
        assert issubclass(ConfigurationError, SoccerAnalysisError)
        assert issubclass(InvalidConfigError, ConfigurationError)

    def test_export_errors_inherit_base(self):
        assert issubclass(ExportError, SoccerAnalysisError)
        assert issubclass(ExportFailedError, ExportError)


class TestSoccerAnalysisError:
    """Test the base exception."""

    def test_simple_message(self):
        err = SoccerAnalysisError("something failed")
        assert str(err) == "something failed"
        assert err.message == "something failed"
        assert err.context == {}

    def test_message_with_context(self):
        err = SoccerAnalysisError("failed", {"key": "val"})
        assert "key=val" in str(err)
        assert err.context == {"key": "val"}

    def test_can_be_raised_and_caught(self):
        with pytest.raises(SoccerAnalysisError):
            raise SoccerAnalysisError("test error")


class TestVideoExceptions:
    """Test video-related exceptions."""

    def test_video_load_error_with_reason(self):
        err = VideoLoadError("/path/to/video.mp4", reason="file not found")
        assert "file not found" in str(err)
        assert err.context["path"] == "/path/to/video.mp4"

    def test_video_load_error_without_reason(self):
        err = VideoLoadError("/path/to/video.mp4")
        assert "Failed to load video" in str(err)

    def test_video_frame_error(self):
        err = VideoFrameError(42, total_frames=1000)
        assert err.context["frame"] == 42
        assert err.context["total_frames"] == 1000

    def test_video_frame_error_without_total(self):
        err = VideoFrameError(10)
        assert err.context["frame"] == 10
        assert "total_frames" not in err.context

    def test_video_processing_error(self):
        err = VideoProcessingError("decode failed", frame_number=99, codec="h264")
        assert "decode failed" in str(err)
        assert err.context["frame"] == 99
        assert err.context["codec"] == "h264"


class TestDetectionExceptions:
    """Test detection-related exceptions."""

    def test_model_load_error(self):
        err = ModelLoadError("/models/yolo.pt", reason="corrupted weights")
        assert "corrupted weights" in str(err)
        assert err.context["model"] == "/models/yolo.pt"

    def test_detection_failed_error(self):
        err = DetectionFailedError(frame_number=5, reason="low quality")
        assert "low quality" in str(err)
        assert err.context["frame"] == 5

    def test_color_extraction_error_with_bbox(self):
        err = ColorExtractionError(bbox=(10, 20, 30, 40), reason="too small")
        assert err.context["bbox"] == (10, 20, 30, 40)
        assert "too small" in str(err)

    def test_color_extraction_error_without_bbox(self):
        err = ColorExtractionError(reason="empty region")
        assert "bbox" not in err.context


class TestAnalysisExceptions:
    """Test analysis-related exceptions."""

    def test_calibration_error(self):
        err = CalibrationError("not enough samples")
        assert "not enough samples" in str(err)

    def test_formation_detection_error_with_team(self):
        err = FormationDetectionError(team="home", reason="too few players")
        assert err.context["team"] == "home"
        assert "too few players" in str(err)

    def test_formation_detection_error_without_team(self):
        err = FormationDetectionError(reason="ambiguous")
        assert "team" not in err.context


class TestDatabaseExceptions:
    """Test database-related exceptions."""

    def test_session_not_found(self):
        err = SessionNotFoundError(session_id=42)
        assert err.context["session_id"] == 42

    def test_persistence_error(self):
        err = PersistenceError("player", reason="constraint violation")
        assert "player" in str(err)
        assert "constraint violation" in str(err)


class TestConfigExceptions:
    """Test configuration-related exceptions."""

    def test_invalid_config_error(self):
        err = InvalidConfigError(key="threshold", value=-1, expected="0.0 to 1.0")
        assert err.context["key"] == "threshold"
        assert err.context["value"] == -1
        assert err.context["expected"] == "0.0 to 1.0"


class TestExportExceptions:
    """Test export-related exceptions."""

    def test_export_failed_error(self):
        err = ExportFailedError(format="pdf", path="/tmp/report.pdf", reason="disk full")
        assert err.context["format"] == "pdf"
        assert err.context["path"] == "/tmp/report.pdf"
        assert "disk full" in str(err)

    def test_export_failed_error_minimal(self):
        err = ExportFailedError(format="csv")
        assert err.context["format"] == "csv"
        assert "path" not in err.context
