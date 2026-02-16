"""
Video Controller - Manages video loading, playback, and navigation

Uses a background thread for frame reading + annotation to keep the
Qt main thread responsive during playback.
"""

import threading
from collections import deque
from typing import Optional, Callable
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from loguru import logger

from src.core.video_processor import ThreadedVideoProcessor, VideoInfo
from src.detection.detector import FrameDetections


class VideoController(QObject):
    """
    Controller for video playback and navigation.

    Separates video management logic from the UI, making the MainWindow
    thinner and easier to test.

    Uses a background prefetch thread during playback to avoid blocking
    the main Qt thread on cap.set() / cap.read() / draw_detections().
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

        # Frame buffer for smooth playback (background prefetch)
        self._frame_buffer: deque = deque(maxlen=30)
        self._buffer_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_stop = threading.Event()

        # Annotated frame cache (avoids re-drawing on seek-back)
        self._annotated_cache: dict = {}
        self._annotated_cache_max = 200  # Cache up to 200 frames

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
            self._stop_prefetch()
            self._frame_buffer.clear()
            self._annotated_cache.clear()

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

        # Start background prefetch thread
        self._start_prefetch(self._current_frame)

        interval = int(1000 / self.fps)  # milliseconds per frame
        self._playback_timer.start(interval)

    def pause(self):
        """Pause video playback"""
        self._is_playing = False
        self.playback_state_changed.emit(False)
        self._playback_timer.stop()
        self._stop_prefetch()

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

        # Get the frame (from cache or fresh)
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
        # Check annotated cache first
        if self._current_frame in self._annotated_cache:
            return self._annotated_cache[self._current_frame]

        frame = self.processor.get_annotated_frame(self._current_frame)
        if frame is None:
            frame = self.processor.get_frame(self._current_frame)

        # Cache the result
        if frame is not None:
            if len(self._annotated_cache) >= self._annotated_cache_max:
                # Remove oldest entries (keep cache bounded)
                keys_to_remove = sorted(self._annotated_cache.keys())[:50]
                for k in keys_to_remove:
                    self._annotated_cache.pop(k, None)
            self._annotated_cache[self._current_frame] = frame

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

        self._current_frame = next_frame

        # Try to pull from prefetch buffer first (non-blocking)
        frame = None
        with self._buffer_lock:
            if self._frame_buffer:
                buf_frame_num, buf_frame = self._frame_buffer[0]
                if buf_frame_num == next_frame:
                    self._frame_buffer.popleft()
                    frame = buf_frame

        # If buffer miss, get directly (blocking but with cache)
        if frame is None:
            frame = self.get_current_frame()

        self.frame_changed.emit(next_frame, frame)

    def _start_prefetch(self, start_frame: int):
        """Start background thread to prefetch frames ahead of playback."""
        self._stop_prefetch()
        self._prefetch_stop.clear()
        self._frame_buffer.clear()

        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(start_frame,),
            daemon=True
        )
        self._prefetch_thread.start()

    def _stop_prefetch(self):
        """Stop the background prefetch thread."""
        self._prefetch_stop.set()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
        self._prefetch_thread = None

    def _prefetch_worker(self, start_frame: int):
        """
        Background worker that reads frames ahead of the playback position.

        Reads frames sequentially and adds annotated frames to the buffer.
        The main thread pulls from this buffer during _advance_frame().
        """
        frame_num = start_frame + 1
        while not self._prefetch_stop.is_set() and frame_num < self.total_frames:
            # Don't get too far ahead of playback
            with self._buffer_lock:
                buffer_size = len(self._frame_buffer)

            if buffer_size >= 20:
                # Buffer is full enough, wait a bit
                self._prefetch_stop.wait(timeout=0.01)
                # Re-check current position (playback may have caught up)
                frame_num = max(frame_num, self._current_frame + 1)
                continue

            try:
                # Check cache first
                if frame_num in self._annotated_cache:
                    frame = self._annotated_cache[frame_num]
                else:
                    frame = self.processor.get_annotated_frame(frame_num)
                    if frame is None:
                        frame = self.processor.get_frame(frame_num)

                    # Cache it
                    if frame is not None:
                        self._annotated_cache[frame_num] = frame

                if frame is not None:
                    with self._buffer_lock:
                        self._frame_buffer.append((frame_num, frame))

            except Exception as e:
                logger.debug(f"Prefetch error on frame {frame_num}: {e}")

            frame_num += 1

    def release(self):
        """Release video resources"""
        self.pause()
        self._stop_prefetch()
        self._frame_buffer.clear()
        self._annotated_cache.clear()
        self.processor.release()
