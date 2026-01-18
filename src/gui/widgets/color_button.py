"""
Color picker button widget
"""

from PyQt6.QtWidgets import QPushButton, QColorDialog
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from typing import Tuple


class ColorButton(QPushButton):
    """Button that shows and allows picking a color"""

    color_changed = pyqtSignal(tuple)  # RGB tuple

    def __init__(self, color: Tuple[int, int, int] = (255, 255, 255), parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(60, 30)
        self.clicked.connect(self._pick_color)
        self._update_style()

    def _update_style(self):
        r, g, b = self._color
        # Determine text color based on brightness
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        text_color = "#000000" if brightness > 128 else "#ffffff"
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                color: {text_color};
                border: 2px solid #555;
                border-radius: 4px;
                font-weight: bold;
            }}
        """)
        self.setText(f"#{r:02x}{g:02x}{b:02x}")

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(*self._color), self, "Select Color")
        if color.isValid():
            self._color = (color.red(), color.green(), color.blue())
            self._update_style()
            self.color_changed.emit(self._color)

    def get_color(self) -> Tuple[int, int, int]:
        return self._color

    def set_color(self, color: Tuple[int, int, int]):
        self._color = color
        self._update_style()
