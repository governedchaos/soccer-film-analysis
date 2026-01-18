"""
Analysis Controller - Manages video analysis workflow
"""

import sys
import faulthandler
import traceback
from typing import Optional, Dict, Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from loguru import logger

from config import AnalysisDepth
from src.core.video_processor import ThreadedVideoProcessor, AnalysisProgress
from src.detection.detector import FrameDetections

# Enable faulthandler to get tracebacks on segfaults
faulthandler.enable()

# Install custom exception hook to catch all unhandled exceptions
def _exception_hook(exc_type, exc_value, exc_tb):
    logger.error("=" * 50)
    logger.error("UNHANDLED EXCEPTION")
    logger.error("=" * 50)
    logger.error(f"Type: {exc_type.__name__}")
    logger.error(f"Value: {exc_value}")
    logger.error("Traceback:")
    for line in traceback.format_tb(exc_tb):
        logger.error(line.strip())
    logger.error("=" * 50)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _exception_hook


class AnalysisController(QObject):
    """
    Controller for video analysis workflow.

    Manages starting/stopping analysis, progress tracking, and result handling.
    Uses signals for thread-safe communication with the UI.
    """

    # Signals for thread-safe UI updates
    analysis_started = pyqtSignal(str)  # depth level
    analysis_progress = pyqtSignal(object)  # AnalysisProgress
    frame_analyzed = pyqtSignal(np.ndarray, object)  # frame, FrameDetections
    analysis_completed = pyqtSignal(dict)  # result dict
    analysis_stopped = pyqtSignal()
    analysis_failed = pyqtSignal(str)  # error message

    def __init__(self, processor: ThreadedVideoProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self._is_analyzing = False

    @property
    def is_analyzing(self) -> bool:
        return self._is_analyzing

    @property
    def has_results(self) -> bool:
        """Check if analysis results are available"""
        return bool(self.processor.frame_detections)

    def start_analysis(self, depth: str = "standard"):
        """
        Start video analysis.

        Args:
            depth: Analysis depth level ("quick", "standard", "deep")
        """
        if self.processor.video_info is None:
            self.analysis_failed.emit("No video loaded")
            return

        if self._is_analyzing:
            self.analysis_failed.emit("Analysis already in progress")
            return

        try:
            depth_enum = AnalysisDepth(depth)
            self._is_analyzing = True
            self.analysis_started.emit(depth)

            logger.info(f"Starting {depth} analysis")

            # Start async analysis with callbacks
            logger.debug("[ANALYSIS] Starting process_video_async...")
            self.processor.process_video_async(
                analysis_depth=depth_enum,
                progress_callback=self._on_progress,
                frame_callback=self._on_frame,
                completion_callback=self._on_complete
            )
            logger.debug("[ANALYSIS] process_video_async started successfully")

        except Exception as e:
            self._is_analyzing = False
            error_msg = f"Failed to start analysis: {e}"
            logger.error(error_msg)
            self.analysis_failed.emit(error_msg)

    def stop_analysis(self):
        """Stop the current analysis"""
        if self._is_analyzing:
            self.processor.stop_processing()
            self._is_analyzing = False
            self.analysis_stopped.emit()
            logger.info("Analysis stopped by user")

    def _on_progress(self, progress: AnalysisProgress):
        """Callback for progress updates (runs in worker thread)"""
        try:
            logger.debug(f"[CALLBACK] Progress: frame {progress.current_frame}/{progress.total_frames}")
            self.analysis_progress.emit(progress)
            logger.debug(f"[CALLBACK] Progress signal emitted successfully")
        except Exception as e:
            logger.error(f"[CALLBACK] Progress callback CRASHED: {e}")
            logger.error(traceback.format_exc())

    def _on_frame(self, frame: np.ndarray, detections: FrameDetections):
        """Callback for analyzed frames (runs in worker thread)"""
        try:
            logger.debug(f"[CALLBACK] Frame {detections.frame_number}: shape={frame.shape if frame is not None else 'None'}")
            # Make a copy of the frame to avoid memory issues
            frame_copy = frame.copy() if frame is not None else None
            logger.debug(f"[CALLBACK] Frame copied, emitting signal...")
            self.frame_analyzed.emit(frame_copy, detections)
            logger.debug(f"[CALLBACK] Frame signal emitted successfully")
        except Exception as e:
            logger.error(f"[CALLBACK] Frame callback CRASHED: {e}")
            logger.error(traceback.format_exc())

    def _on_complete(self, result: Dict[str, Any]):
        """Callback when analysis completes (runs in worker thread)"""
        try:
            logger.debug(f"[CALLBACK] Complete called with status: {result.get('status')}")
            self._is_analyzing = False

            if result.get("status") == "completed":
                logger.info(
                    f"Analysis complete: {result.get('processed_frames', 0)} frames "
                    f"in {result.get('elapsed_seconds', 0)/60:.1f} min"
                )
                self.analysis_completed.emit(result)
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"Analysis failed: {error}")
                self.analysis_failed.emit(error)
            logger.debug(f"[CALLBACK] Complete signal emitted successfully")
        except Exception as e:
            logger.error(f"[CALLBACK] Complete callback CRASHED: {e}")
            logger.error(traceback.format_exc())

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics from the analysis"""
        return self.processor.get_detection_stats()

    def get_analysis_summary(self):
        """Get tactical analysis summary"""
        return self.processor.get_analysis_summary()

    def get_frame_detections(self, frame_number: int) -> Optional[FrameDetections]:
        """Get detections for a specific frame"""
        return self.processor.frame_detections.get(frame_number)

    def get_possession(self):
        """Get possession percentages"""
        if hasattr(self.processor, 'possession_calculator'):
            return self.processor.possession_calculator.get_possession_percentage()
        return (50.0, 50.0)
