"""
Video display widget
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from loguru import logger


class VideoWidget(QLabel):
    """
    Widget for displaying video frames with detection overlays.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("No video loaded\n\nClick 'Load Video' to begin")
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px solid #333;
                border-radius: 8px;
                color: #888;
                font-size: 14px;
            }
        """)

    def display_frame(self, frame: np.ndarray):
        """
        Display an OpenCV frame in the widget.

        Args:
            frame: BGR numpy array
        """
        try:
            logger.debug(f"[VIDEO_WIDGET] display_frame called")
            if frame is None:
                logger.debug(f"[VIDEO_WIDGET] Frame is None, returning")
                return

            logger.debug(f"[VIDEO_WIDGET] Frame shape: {frame.shape}, dtype: {frame.dtype}")

            # Convert BGR to RGB
            logger.debug(f"[VIDEO_WIDGET] Converting BGR to RGB...")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            logger.debug(f"[VIDEO_WIDGET] RGB frame: {w}x{h}, channels: {ch}")

            # Scale to fit widget while maintaining aspect ratio
            widget_w, widget_h = self.width(), self.height()
            scale = min(widget_w / w, widget_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            logger.debug(f"[VIDEO_WIDGET] Scaling: widget={widget_w}x{widget_h}, scale={scale:.3f}, new={new_w}x{new_h}")

            if scale != 1.0:
                logger.debug(f"[VIDEO_WIDGET] Resizing frame...")
                rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                h, w = new_h, new_w
                logger.debug(f"[VIDEO_WIDGET] Resized to {w}x{h}")

            # Convert to QImage - MUST use copy() to avoid crash when numpy array is garbage collected
            bytes_per_line = ch * w
            logger.debug(f"[VIDEO_WIDGET] Creating QImage: {w}x{h}, bytes_per_line={bytes_per_line}")
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            logger.debug(f"[VIDEO_WIDGET] QImage created, size: {qt_image.width()}x{qt_image.height()}")

            # Display
            logger.debug(f"[VIDEO_WIDGET] Setting pixmap...")
            self.setPixmap(QPixmap.fromImage(qt_image))
            logger.debug(f"[VIDEO_WIDGET] display_frame completed successfully")

        except Exception as e:
            logger.error(f"[VIDEO_WIDGET] display_frame CRASHED: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def clear_display(self):
        """Clear the display"""
        self.clear()
        self.setText("No video loaded")
