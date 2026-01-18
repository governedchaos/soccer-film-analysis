"""
Debug log panel widget
"""

from datetime import datetime
from PyQt6.QtWidgets import QTextEdit


class LogPanel(QTextEdit):
    """
    Panel displaying real-time log messages for debugging.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(150)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                font-family: Consolas, monospace;
                font-size: 11px;
                border: 1px solid #333;
            }
        """)
        self.setPlaceholderText("Log messages will appear here...")

    def log(self, message: str, level: str = "INFO"):
        """Add a log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = {
            "DEBUG": "#888888",
            "INFO": "#00ff00",
            "WARNING": "#ffaa00",
            "ERROR": "#ff4444"
        }.get(level, "#ffffff")
        self.append(f'<span style="color: {color}">[{timestamp}] [{level}] {message}</span>')
        # Auto-scroll to bottom
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
