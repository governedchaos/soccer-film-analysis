"""
Video display widget
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


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
        if frame is None:
            return

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape

        # Scale to fit widget while maintaining aspect ratio
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / w, widget_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        if scale != 1.0:
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w

        # Convert to QImage
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Display
        self.setPixmap(QPixmap.fromImage(qt_image))

    def clear_display(self):
        """Clear the display"""
        self.clear()
        self.setText("No video loaded")
