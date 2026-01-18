"""
Video Controller - Manages video loading, playback, and navigation
"""

from typing import Optional, Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from loguru import logger

from src.core.video_processor import ThreadedVideoProcessor, VideoInfo
from src.detection.detector import FrameDetections


class VideoController(QObject):
    """
    Controller for video playback and navigation.

    Separates video management logic from the UI, making the MainWindow
    thinner and easier to test.
    """

    # Signals
    video_loaded = pyqtSignal(object)  # VideoInfo
    frame_changed = pyqtSignal(int, object)  # frame_number, frame (numpy array)
    playback_state_changed = pyqtSignal(bool)  # is_playing
    error_occurred = pyqtSignal(str)

    def __init__(self, processor: ThreadedVideoProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.video_info: Optional[VideoInfo] = None

        # Playback state
        self._is_playing = False
        self._current_frame = 0

        # Playback timer
        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._advance_frame)

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def total_frames(self) -> int:
        return self.video_info.total_frames if self.video_info else 0

    @property
    def fps(self) -> float:
        return self.video_info.fps if self.video_info else 30.0

    def load_video(self, file_path: str) -> bool:
        """
        Load a video file.

        Args:
            file_path: Path to the video file

        Returns:
            True if successful
        """
        try:
            self.video_info = self.processor.load_video(file_path)
            self._current_frame = 0

            logger.info(f"Video loaded: {file_path}")
            self.video_loaded.emit(self.video_info)
            return True

        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            self.error_occurred.emit(f"Failed to load video: {e}")
            return False

    def calibrate_teams(self) -> bool:
        """Auto-calibrate team colors from sample frames"""
        try:
            return self.processor.calibrate_teams()
        except Exception as e:
            logger.warning(f"Team calibration warning: {e}")
            return False

    def set_team_colors(self, home_color, away_color):
        """Manually set team colors"""
        self.processor.team_classifier.set_team_colors(home_color, away_color)

    def play(self):
        """Start video playback"""
        if self.video_info is None:
            return

        self._is_playing = True
        self.playback_state_changed.emit(True)

        interval = int(1000 / self.fps)  # milliseconds per frame
        self._playback_timer.start(interval)

    def pause(self):
        """Pause video playback"""
        self._is_playing = False
        self.playback_state_changed.emit(False)
        self._playback_timer.stop()

    def toggle_playback(self):
        """Toggle between play and pause"""
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def seek(self, frame_number: int):
        """
        Seek to a specific frame.

        Args:
            frame_number: Target frame number
        """
        if self.video_info is None:
            return

        # Clamp to valid range
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        self._current_frame = frame_number

        # Get the frame
        frame = self.get_current_frame()
        self.frame_changed.emit(frame_number, frame)

    def seek_relative(self, delta: int):
        """
        Seek relative to current position.

        Args:
            delta: Number of frames to seek (positive or negative)
        """
        self.seek(self._current_frame + delta)

    def get_current_frame(self):
        """Get the current frame as numpy array (with annotations if available)"""
        frame = self.processor.get_annotated_frame(self._current_frame)
        if frame is None:
            frame = self.processor.get_frame(self._current_frame)
        return frame

    def get_current_detections(self) -> Optional[FrameDetections]:
        """Get detections for the current frame (if analysis was run)"""
        return self.processor.frame_detections.get(self._current_frame)

    def get_possession(self):
        """Get current possession percentages"""
        if hasattr(self.processor, 'possession_calculator'):
            return self.processor.possession_calculator.get_possession_percentage()
        return None

    def _advance_frame(self):
        """Advance to next frame during playback (called by timer)"""
        if self.video_info is None:
            return

        next_frame = self._current_frame + 1
        if next_frame >= self.total_frames:
            self.pause()
            return

        self.seek(next_frame)

    def release(self):
        """Release video resources"""
        self.pause()
        self.processor.release()
