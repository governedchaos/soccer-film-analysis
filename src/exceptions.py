"""
Custom exceptions for Soccer Film Analysis

Provides structured error handling with context for debugging.
"""

from typing import Optional, Dict, Any


class SoccerAnalysisError(Exception):
    """Base exception for all soccer analysis errors"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message


# Video Processing Errors
class VideoError(SoccerAnalysisError):
    """Base class for video-related errors"""
    pass


class VideoLoadError(VideoError):
    """Failed to load video file"""

    def __init__(self, path: str, reason: str = ""):
        super().__init__(
            f"Failed to load video: {reason}" if reason else "Failed to load video",
            {"path": path}
        )


class VideoFrameError(VideoError):
    """Failed to read a video frame"""

    def __init__(self, frame_number: int, total_frames: int = None):
        ctx = {"frame": frame_number}
        if total_frames:
            ctx["total_frames"] = total_frames
        super().__init__("Failed to read video frame", ctx)


class VideoProcessingError(VideoError):
    """Error during video processing"""

    def __init__(self, message: str, frame_number: int = None, **kwargs):
        ctx = kwargs
        if frame_number is not None:
            ctx["frame"] = frame_number
        super().__init__(message, ctx)


# Detection Errors
class DetectionError(SoccerAnalysisError):
    """Base class for detection-related errors"""
    pass


class ModelLoadError(DetectionError):
    """Failed to load detection model"""

    def __init__(self, model_path: str, reason: str = ""):
        super().__init__(
            f"Failed to load model: {reason}" if reason else "Failed to load model",
            {"model": model_path}
        )


class DetectionFailedError(DetectionError):
    """Detection failed on a frame"""

    def __init__(self, frame_number: int, reason: str = ""):
        super().__init__(
            f"Detection failed: {reason}" if reason else "Detection failed",
            {"frame": frame_number}
        )


class ColorExtractionError(DetectionError):
    """Failed to extract color from detection"""

    def __init__(self, bbox: tuple = None, reason: str = ""):
        ctx = {}
        if bbox:
            ctx["bbox"] = bbox
        super().__init__(
            f"Color extraction failed: {reason}" if reason else "Color extraction failed",
            ctx
        )


# Analysis Errors
class AnalysisError(SoccerAnalysisError):
    """Base class for analysis-related errors"""
    pass


class CalibrationError(AnalysisError):
    """Team color calibration failed"""

    def __init__(self, reason: str = ""):
        super().__init__(
            f"Calibration failed: {reason}" if reason else "Calibration failed"
        )


class FormationDetectionError(AnalysisError):
    """Formation detection failed"""

    def __init__(self, team: str = None, reason: str = ""):
        ctx = {}
        if team:
            ctx["team"] = team
        super().__init__(
            f"Formation detection failed: {reason}" if reason else "Formation detection failed",
            ctx
        )


# Database Errors
class DatabaseError(SoccerAnalysisError):
    """Base class for database-related errors"""
    pass


class SessionNotFoundError(DatabaseError):
    """Analysis session not found in database"""

    def __init__(self, session_id: int):
        super().__init__("Analysis session not found", {"session_id": session_id})


class PersistenceError(DatabaseError):
    """Failed to persist data to database"""

    def __init__(self, entity: str, reason: str = ""):
        super().__init__(
            f"Failed to save {entity}: {reason}" if reason else f"Failed to save {entity}",
            {"entity": entity}
        )


# Configuration Errors
class ConfigurationError(SoccerAnalysisError):
    """Configuration-related errors"""
    pass


class InvalidConfigError(ConfigurationError):
    """Invalid configuration value"""

    def __init__(self, key: str, value: Any, expected: str = ""):
        ctx = {"key": key, "value": value}
        if expected:
            ctx["expected"] = expected
        super().__init__("Invalid configuration", ctx)


# Export Errors
class ExportError(SoccerAnalysisError):
    """Export-related errors"""
    pass


class ExportFailedError(ExportError):
    """Failed to export results"""

    def __init__(self, format: str, path: str = None, reason: str = ""):
        ctx = {"format": format}
        if path:
            ctx["path"] = path
        super().__init__(
            f"Export failed: {reason}" if reason else "Export failed",
            ctx
        )
