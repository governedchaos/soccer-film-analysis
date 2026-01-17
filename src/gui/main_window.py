"""
Soccer Film Analysis - Main Window
PyQt6-based GUI for the soccer film analysis application
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QLabel, QSlider, QFileDialog, QComboBox,
    QProgressBar, QGroupBox, QGridLayout, QStatusBar, QMessageBox,
    QTabWidget, QTextEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QFrame, QSizePolicy, QStyle, QDialog, QLineEdit, QColorDialog,
    QFormLayout, QDialogButtonBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QMetaObject, Q_ARG
from PyQt6.QtGui import QImage, QPixmap, QAction, QPalette, QColor

import cv2
import numpy as np
from loguru import logger

from config import settings, AnalysisDepth
from src.core.video_processor import ThreadedVideoProcessor, VideoInfo, AnalysisProgress
from src.detection.detector import FrameDetections


class ColorButton(QPushButton):
    """Button that shows and allows picking a color"""

    color_changed = pyqtSignal(tuple)  # RGB tuple

    def __init__(self, color=(255, 255, 255), parent=None):
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

    def get_color(self):
        return self._color

    def set_color(self, color):
        self._color = color
        self._update_style()


class GameConfigDialog(QDialog):
    """
    Dialog for configuring game metadata, team colors, and field settings.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Game Configuration")
        self.setMinimumWidth(500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Create scroll area for all settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # === GAME INFO ===
        game_group = QGroupBox("Game Information")
        game_layout = QFormLayout()

        self.home_team_name = QLineEdit()
        self.home_team_name.setPlaceholderText("e.g., La Follette High School")
        game_layout.addRow("Home Team:", self.home_team_name)

        self.away_team_name = QLineEdit()
        self.away_team_name.setPlaceholderText("e.g., Madison East High School")
        game_layout.addRow("Away Team:", self.away_team_name)

        self.competition = QLineEdit()
        self.competition.setPlaceholderText("e.g., Conference Championship")
        game_layout.addRow("Competition:", self.competition)

        self.venue = QLineEdit()
        self.venue.setPlaceholderText("e.g., Breese Stevens Field")
        game_layout.addRow("Venue:", self.venue)

        game_group.setLayout(game_layout)
        scroll_layout.addWidget(game_group)

        # === TEAM COLORS ===
        colors_group = QGroupBox("Team Colors (for player identification)")
        colors_layout = QGridLayout()

        colors_layout.addWidget(QLabel("Home Team:"), 0, 0)
        self.home_primary_color = ColorButton((255, 215, 0))  # Gold
        colors_layout.addWidget(QLabel("Primary"), 0, 1)
        colors_layout.addWidget(self.home_primary_color, 0, 2)
        self.home_secondary_color = ColorButton((0, 0, 0))  # Black
        colors_layout.addWidget(QLabel("Secondary"), 0, 3)
        colors_layout.addWidget(self.home_secondary_color, 0, 4)

        colors_layout.addWidget(QLabel("Away Team:"), 1, 0)
        self.away_primary_color = ColorButton((255, 0, 0))  # Red
        colors_layout.addWidget(QLabel("Primary"), 1, 1)
        colors_layout.addWidget(self.away_primary_color, 1, 2)
        self.away_secondary_color = ColorButton((255, 255, 255))  # White
        colors_layout.addWidget(QLabel("Secondary"), 1, 3)
        colors_layout.addWidget(self.away_secondary_color, 1, 4)

        colors_layout.addWidget(QLabel("Home GK:"), 2, 0)
        self.home_gk_color = ColorButton((0, 255, 0))  # Green
        colors_layout.addWidget(self.home_gk_color, 2, 2)

        colors_layout.addWidget(QLabel("Away GK:"), 3, 0)
        self.away_gk_color = ColorButton((255, 165, 0))  # Orange
        colors_layout.addWidget(self.away_gk_color, 3, 2)

        colors_layout.addWidget(QLabel("Referees:"), 4, 0)
        self.referee_color = ColorButton((0, 0, 0))  # Black
        colors_layout.addWidget(self.referee_color, 4, 2)

        colors_group.setLayout(colors_layout)
        scroll_layout.addWidget(colors_group)

        # === FIELD SETTINGS ===
        field_group = QGroupBox("Field Settings")
        field_layout = QFormLayout()

        self.field_type = QComboBox()
        self.field_type.addItems(["Soccer-only field (white lines)",
                                   "Multi-sport field (colored lines)",
                                   "Custom"])
        self.field_type.currentIndexChanged.connect(self._on_field_type_changed)
        field_layout.addRow("Field Type:", self.field_type)

        # Line colors
        line_colors_widget = QWidget()
        line_colors_layout = QHBoxLayout(line_colors_widget)
        line_colors_layout.setContentsMargins(0, 0, 0, 0)

        line_colors_layout.addWidget(QLabel("Soccer lines:"))
        self.soccer_line_color = ColorButton((255, 255, 255))  # White
        line_colors_layout.addWidget(self.soccer_line_color)

        line_colors_layout.addWidget(QLabel("Other lines:"))
        self.other_line_color = ColorButton((255, 255, 0))  # Yellow
        line_colors_layout.addWidget(self.other_line_color)

        line_colors_layout.addStretch()
        field_layout.addRow("Line Colors:", line_colors_widget)

        field_group.setLayout(field_layout)
        scroll_layout.addWidget(field_group)

        # === BALL SETTINGS ===
        ball_group = QGroupBox("Ball Identification")
        ball_layout = QFormLayout()

        self.ball_type = QComboBox()
        self.ball_type.addItems(["Standard white/black",
                                  "High-visibility (yellow/orange)",
                                  "Custom color"])
        ball_layout.addRow("Ball Type:", self.ball_type)

        self.ball_color = ColorButton((255, 255, 255))
        ball_layout.addRow("Ball Color:", self.ball_color)

        self.exclude_ballboys = QCheckBox("Exclude ball boys from tracking")
        self.exclude_ballboys.setChecked(True)
        ball_layout.addRow("", self.exclude_ballboys)

        ball_group.setLayout(ball_layout)
        scroll_layout.addWidget(ball_group)

        # === DETECTION SETTINGS ===
        detection_group = QGroupBox("Detection Sensitivity")
        detection_layout = QFormLayout()

        self.player_confidence = QDoubleSpinBox()
        self.player_confidence.setRange(0.1, 1.0)
        self.player_confidence.setSingleStep(0.05)
        self.player_confidence.setValue(0.3)
        detection_layout.addRow("Player Confidence:", self.player_confidence)

        self.ball_confidence = QDoubleSpinBox()
        self.ball_confidence.setRange(0.1, 1.0)
        self.ball_confidence.setSingleStep(0.05)
        self.ball_confidence.setValue(0.25)
        detection_layout.addRow("Ball Confidence:", self.ball_confidence)

        detection_group.setLayout(detection_layout)
        scroll_layout.addWidget(detection_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # === BUTTONS ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply_settings)
        layout.addWidget(button_box)

    def _on_field_type_changed(self, index):
        """Update line colors based on field type selection"""
        if index == 0:  # Soccer-only
            self.soccer_line_color.set_color((255, 255, 255))
        elif index == 1:  # Multi-sport
            self.soccer_line_color.set_color((255, 255, 0))  # Yellow for soccer

    def _apply_settings(self):
        """Apply settings without closing dialog"""
        self.log_message("Settings applied", "INFO")

    def get_config(self):
        """Return all configuration as a dictionary"""
        return {
            "home_team": {
                "name": self.home_team_name.text(),
                "primary_color": self.home_primary_color.get_color(),
                "secondary_color": self.home_secondary_color.get_color(),
                "goalkeeper_color": self.home_gk_color.get_color(),
            },
            "away_team": {
                "name": self.away_team_name.text(),
                "primary_color": self.away_primary_color.get_color(),
                "secondary_color": self.away_secondary_color.get_color(),
                "goalkeeper_color": self.away_gk_color.get_color(),
            },
            "referee_color": self.referee_color.get_color(),
            "competition": self.competition.text(),
            "venue": self.venue.text(),
            "field": {
                "type": self.field_type.currentText(),
                "soccer_line_color": self.soccer_line_color.get_color(),
                "other_line_color": self.other_line_color.get_color(),
            },
            "ball": {
                "type": self.ball_type.currentText(),
                "color": self.ball_color.get_color(),
                "exclude_ballboys": self.exclude_ballboys.isChecked(),
            },
            "detection": {
                "player_confidence": self.player_confidence.value(),
                "ball_confidence": self.ball_confidence.value(),
            }
        }

    def set_config(self, config: dict):
        """Load configuration from dictionary"""
        if "home_team" in config:
            self.home_team_name.setText(config["home_team"].get("name", ""))
            if "primary_color" in config["home_team"]:
                self.home_primary_color.set_color(config["home_team"]["primary_color"])
        if "away_team" in config:
            self.away_team_name.setText(config["away_team"].get("name", ""))
            if "primary_color" in config["away_team"]:
                self.away_primary_color.set_color(config["away_team"]["primary_color"])


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
        from datetime import datetime
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


class VideoWidget(QLabel):
    """
    Widget for displaying video frames with detection overlays.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
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
    
    def clear(self):
        """Clear the display"""
        self.clear()
        self.setText("No video loaded")


class StatsPanel(QGroupBox):
    """
    Panel displaying real-time detection statistics.
    """
    
    def __init__(self, parent=None):
        super().__init__("Detection Statistics", parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QGridLayout()
        layout.setSpacing(10)
        
        # Create stat labels
        self.stat_labels = {}
        stats = [
            ("frame", "Frame:"),
            ("time", "Time:"),
            ("players", "Players:"),
            ("home", "Home Team:"),
            ("away", "Away Team:"),
            ("ball", "Ball:"),
            ("referees", "Referees:"),
            ("possession", "Possession:"),
        ]
        
        for i, (key, label) in enumerate(stats):
            label_widget = QLabel(label)
            label_widget.setStyleSheet("font-weight: bold;")
            value_widget = QLabel("-")
            value_widget.setStyleSheet("color: #4CAF50;")
            
            layout.addWidget(label_widget, i, 0)
            layout.addWidget(value_widget, i, 1)
            self.stat_labels[key] = value_widget
        
        self.setLayout(layout)
    
    def update_stats(self, frame_num: int, fps: float, detections: Optional[FrameDetections]):
        """Update displayed statistics"""
        time_str = f"{frame_num / fps:.1f}s" if fps > 0 else "-"
        
        self.stat_labels["frame"].setText(str(frame_num))
        self.stat_labels["time"].setText(time_str)
        
        if detections:
            total_players = len(detections.players) + len(detections.goalkeepers)
            home_count = sum(1 for p in detections.players if p.team_id == 0)
            away_count = sum(1 for p in detections.players if p.team_id == 1)
            
            self.stat_labels["players"].setText(str(total_players))
            self.stat_labels["home"].setText(str(home_count + sum(1 for g in detections.goalkeepers if g.team_id == 0)))
            self.stat_labels["away"].setText(str(away_count + sum(1 for g in detections.goalkeepers if g.team_id == 1)))
            self.stat_labels["ball"].setText("✓" if detections.ball else "✗")
            self.stat_labels["referees"].setText(str(len(detections.referees)))
        else:
            for key in ["players", "home", "away", "ball", "referees"]:
                self.stat_labels[key].setText("-")


class ControlPanel(QGroupBox):
    """
    Video playback and analysis controls.
    """
    
    # Signals
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    seek_frame = pyqtSignal(int)
    analysis_started = pyqtSignal(str)  # depth level
    analysis_stopped = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        self.setup_ui()
        self._is_playing = False
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._on_play_clicked)
        
        self.prev_frame_btn = QPushButton("◀◀")
        self.prev_frame_btn.clicked.connect(lambda: self.seek_frame.emit(-10))
        
        self.next_frame_btn = QPushButton("▶▶")
        self.next_frame_btn.clicked.connect(lambda: self.seek_frame.emit(10))
        
        playback_layout.addWidget(self.prev_frame_btn)
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.next_frame_btn)
        
        # Timeline slider
        slider_layout = QHBoxLayout()
        self.frame_label = QLabel("0 / 0")
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        
        slider_layout.addWidget(self.timeline_slider, stretch=1)
        slider_layout.addWidget(self.frame_label)
        
        # Analysis controls
        analysis_layout = QHBoxLayout()
        
        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["Quick", "Standard", "Deep"])
        self.depth_combo.setCurrentIndex(1)
        
        self.analyze_btn = QPushButton("🔍 Start Analysis")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666;
            }
        """)
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.analysis_stopped.emit)
        
        analysis_layout.addWidget(QLabel("Depth:"))
        analysis_layout.addWidget(self.depth_combo)
        analysis_layout.addWidget(self.analyze_btn)
        analysis_layout.addWidget(self.stop_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - %v frames")
        
        self.progress_label = QLabel("Ready")
        
        # Add all to main layout
        layout.addLayout(playback_layout)
        layout.addLayout(slider_layout)
        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        layout.addLayout(analysis_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        
        self.setLayout(layout)
    
    def _on_play_clicked(self):
        if self._is_playing:
            self._is_playing = False
            self.play_btn.setText("▶ Play")
            self.pause_clicked.emit()
        else:
            self._is_playing = True
            self.play_btn.setText("⏸ Pause")
            self.play_clicked.emit()
    
    def _on_slider_changed(self, value: int):
        self.seek_frame.emit(value)
    
    def _on_analyze_clicked(self):
        depth = self.depth_combo.currentText().lower()
        self.analysis_started.emit(depth)
    
    def set_frame_range(self, total_frames: int):
        """Set the slider range based on video length"""
        self.timeline_slider.setMaximum(total_frames - 1)
    
    def set_current_frame(self, frame: int):
        """Update the slider position without triggering signal"""
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame)
        self.timeline_slider.blockSignals(False)
    
    def update_frame_label(self, current: int, total: int):
        """Update the frame counter label"""
        self.frame_label.setText(f"{current} / {total}")
    
    def update_progress(self, progress: AnalysisProgress):
        """Update progress bar and label"""
        self.progress_bar.setValue(int(progress.percentage))
        self.progress_label.setText(str(progress))
    
    def set_analyzing(self, analyzing: bool):
        """Update UI state for analysis"""
        self.analyze_btn.setEnabled(not analyzing)
        self.stop_btn.setEnabled(analyzing)
        self.depth_combo.setEnabled(not analyzing)
        # Update button text to show current state
        if analyzing:
            self.analyze_btn.setText("⏳ Analysis Running...")
            self.analyze_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    font-weight: bold;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
            """)
        else:
            self.analyze_btn.setText("🔍 Start Analysis")
            self.analyze_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #666;
                }
            """)

    def set_playing(self, playing: bool):
        """Sync play button state with actual playback"""
        self._is_playing = playing
        if playing:
            self.play_btn.setText("⏸ Pause")
        else:
            self.play_btn.setText("▶ Play")


class MainWindow(QMainWindow):
    """
    Main application window.
    """

    # Signals for thread-safe GUI updates
    frame_ready = pyqtSignal(np.ndarray, object)  # frame, detections
    progress_ready = pyqtSignal(object)  # AnalysisProgress
    analysis_complete = pyqtSignal(dict)  # result dict

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soccer Film Analysis")
        self.setMinimumSize(1200, 800)
        
        # Initialize processor
        self.processor = ThreadedVideoProcessor()
        self.video_info: Optional[VideoInfo] = None
        
        # Playback state
        self._is_playing = False
        self._current_frame = 0
        self._playback_timer = QTimer()
        self._playback_timer.timeout.connect(self._advance_frame)
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.connect_signals()
        
        logger.info("MainWindow initialized")
    
    def setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Main splitter for video and side panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - Video display and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Video widget
        self.video_widget = VideoWidget()
        left_layout.addWidget(self.video_widget, stretch=1)

        # Control panel
        self.control_panel = ControlPanel()
        left_layout.addWidget(self.control_panel)

        # Debug log panel (collapsible)
        self.log_panel = LogPanel()
        self.log_toggle_btn = QPushButton("Show Debug Log")
        self.log_toggle_btn.setCheckable(True)
        self.log_toggle_btn.clicked.connect(self._toggle_log_panel)
        self.log_panel.setVisible(False)
        left_layout.addWidget(self.log_toggle_btn)
        left_layout.addWidget(self.log_panel)
        
        # Right side - Stats and info
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Video info
        self.video_info_label = QLabel("No video loaded")
        self.video_info_label.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #2a2a2a;
                border-radius: 4px;
            }
        """)
        right_layout.addWidget(self.video_info_label)
        
        # Stats panel
        self.stats_panel = StatsPanel()
        right_layout.addWidget(self.stats_panel)
        
        # Team color indicators
        team_colors_group = QGroupBox("Team Colors")
        team_colors_layout = QGridLayout()
        
        self.home_color_label = QLabel("Home: ")
        self.home_color_box = QLabel()
        self.home_color_box.setFixedSize(30, 30)
        self.home_color_box.setStyleSheet("background-color: #FFD700; border: 1px solid #333;")
        
        self.away_color_label = QLabel("Away: ")
        self.away_color_box = QLabel()
        self.away_color_box.setFixedSize(30, 30)
        self.away_color_box.setStyleSheet("background-color: #FF4444; border: 1px solid #333;")
        
        team_colors_layout.addWidget(self.home_color_label, 0, 0)
        team_colors_layout.addWidget(self.home_color_box, 0, 1)
        team_colors_layout.addWidget(self.away_color_label, 1, 0)
        team_colors_layout.addWidget(self.away_color_box, 1, 1)
        team_colors_group.setLayout(team_colors_layout)
        right_layout.addWidget(team_colors_group)
        
        right_layout.addStretch()
        
        # Add buttons at bottom
        button_layout = QVBoxLayout()
        
        self.load_video_btn = QPushButton("📂 Load Video")
        self.load_video_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.load_video_btn.clicked.connect(self.load_video)

        self.config_btn = QPushButton("⚙️ Game Settings")
        self.config_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self.config_btn.clicked.connect(self.open_game_config)

        self.export_btn = QPushButton("📊 Export Report")
        self.export_btn.setEnabled(False)

        button_layout.addWidget(self.load_video_btn)
        button_layout.addWidget(self.config_btn)
        button_layout.addWidget(self.export_btn)
        right_layout.addLayout(button_layout)
        
        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([900, 300])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_video)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")

        start_action = QAction("&Start Analysis", self)
        start_action.setShortcut("Ctrl+R")
        start_action.triggered.connect(lambda: self.start_analysis("standard"))
        analysis_menu.addAction(start_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")

        game_config_action = QAction("&Game Settings...", self)
        game_config_action.setShortcut("Ctrl+G")
        game_config_action.triggered.connect(self.open_game_config)
        settings_menu.addAction(game_config_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def connect_signals(self):
        """Connect all signals"""
        self.control_panel.play_clicked.connect(self.play_video)
        self.control_panel.pause_clicked.connect(self.pause_video)
        self.control_panel.seek_frame.connect(self.seek_frame)
        self.control_panel.analysis_started.connect(self.start_analysis)
        self.control_panel.analysis_stopped.connect(self.stop_analysis)

        # Connect thread-safe signals for analysis updates
        self.frame_ready.connect(self._on_frame_ready)
        self.progress_ready.connect(self._on_progress_ready)
        self.analysis_complete.connect(self._on_analysis_complete)

    def _on_frame_ready(self, frame: np.ndarray, detections):
        """Handle frame ready signal (runs in main thread)"""
        self.video_widget.display_frame(frame)
        fps = self.video_info.fps if self.video_info else 30
        self.stats_panel.update_stats(detections.frame_number, fps, detections)
        # Log detection counts periodically
        if detections.frame_number % 50 == 0:
            player_count = len(detections.players)
            ball_detected = "Yes" if detections.ball else "No"
            self.log_message(
                f"Frame {detections.frame_number}: {player_count} players, ball: {ball_detected}",
                "DEBUG"
            )

    def _on_progress_ready(self, progress):
        """Handle progress update signal (runs in main thread)"""
        self.control_panel.update_progress(progress)
        # Log every 100 frames
        if progress.current_frame % 100 == 0:
            self.log_message(
                f"Frame {progress.current_frame}/{progress.total_frames} "
                f"({progress.percentage:.1f}%) - {progress.frames_per_second:.1f} FPS",
                "DEBUG"
            )

    def _on_analysis_complete(self, result: dict):
        """Handle analysis complete signal (runs in main thread)"""
        self.control_panel.set_analyzing(False)
        if result.get("status") == "completed":
            msg = f"Analysis complete! {result.get('processed_frames', 0)} frames in {result.get('elapsed_seconds', 0)/60:.1f} min"
            self.status_bar.showMessage(msg)
            self.log_message(msg, "INFO")
            self.export_btn.setEnabled(True)
            QMessageBox.information(
                self, "Analysis Complete",
                f"Analysis finished successfully!\n"
                f"Processed {result.get('processed_frames', 0)} frames in "
                f"{result.get('elapsed_seconds', 0)/60:.1f} minutes."
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            self.status_bar.showMessage(f"Analysis failed: {error_msg}")
            self.log_message(f"Analysis failed: {error_msg}", "ERROR")
    
    def load_video(self):
        """Open file dialog and load video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            str(settings.get_video_dir()),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.log_message(f"Loading video: {file_path}", "INFO")
            self.video_info = self.processor.load_video(file_path)

            # Update UI
            self.control_panel.set_frame_range(self.video_info.total_frames)
            self.control_panel.update_frame_label(0, self.video_info.total_frames)

            info_text = (
                f"<b>{self.video_info.path.name}</b><br>"
                f"Resolution: {self.video_info.width}x{self.video_info.height}<br>"
                f"FPS: {self.video_info.fps:.1f}<br>"
                f"Duration: {self.video_info.duration_seconds/60:.1f} min<br>"
                f"Frames: {self.video_info.total_frames:,}"
            )
            self.video_info_label.setText(info_text)

            self.log_message(
                f"Video info: {self.video_info.width}x{self.video_info.height}, "
                f"{self.video_info.fps:.1f} FPS, {self.video_info.total_frames} frames",
                "INFO"
            )

            # Show first frame
            self.show_frame(0)

            self.status_bar.showMessage(f"Loaded: {self.video_info.path.name}")
            logger.info(f"Video loaded: {file_path}")

            # Auto-calibrate team colors
            self.status_bar.showMessage("Calibrating team colors...")
            self.log_message("Calibrating team colors from sample frames...", "INFO")
            try:
                self.processor.calibrate_teams()
                self.log_message("Team color calibration complete", "INFO")
            except Exception as cal_error:
                self.log_message(f"Calibration warning: {cal_error}", "WARNING")
            self.status_bar.showMessage(f"Loaded: {self.video_info.path.name} (ready)")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video:\n{str(e)}")
            self.log_message(f"Failed to load video: {e}", "ERROR")
            logger.error(f"Failed to load video: {e}")
    
    def show_frame(self, frame_number: int):
        """Display a specific frame"""
        self._current_frame = frame_number
        
        # Get frame (with annotations if available)
        frame = self.processor.get_annotated_frame(frame_number)
        if frame is None:
            frame = self.processor.get_frame(frame_number)
        
        if frame is not None:
            self.video_widget.display_frame(frame)
            self.control_panel.set_current_frame(frame_number)
            self.control_panel.update_frame_label(frame_number, self.video_info.total_frames if self.video_info else 0)
            
            # Update stats if we have detections for this frame
            detections = self.processor.frame_detections.get(frame_number)
            fps = self.video_info.fps if self.video_info else 30
            self.stats_panel.update_stats(frame_number, fps, detections)
    
    def play_video(self):
        """Start video playback"""
        if self.video_info is None:
            return

        self._is_playing = True
        self.control_panel.set_playing(True)  # Sync button state
        interval = int(1000 / self.video_info.fps)  # milliseconds per frame
        self._playback_timer.start(interval)

    def pause_video(self):
        """Pause video playback"""
        self._is_playing = False
        self.control_panel.set_playing(False)  # Sync button state
        self._playback_timer.stop()

    def _advance_frame(self):
        """Advance to next frame during playback"""
        if self.video_info is None:
            return

        next_frame = self._current_frame + 1
        if next_frame >= self.video_info.total_frames:
            self.pause_video()  # This will sync button state
            return

        self.show_frame(next_frame)
    
    def seek_frame(self, delta_or_absolute: int):
        """Seek to a frame (relative delta or absolute position)"""
        if self.video_info is None:
            return
        
        # If from slider, it's absolute; if from buttons, it's delta
        if abs(delta_or_absolute) < 100:  # Likely a delta
            new_frame = self._current_frame + delta_or_absolute
        else:
            new_frame = delta_or_absolute
        
        new_frame = max(0, min(new_frame, self.video_info.total_frames - 1))
        self.show_frame(new_frame)
    
    def start_analysis(self, depth: str):
        """Start video analysis"""
        if self.video_info is None:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        depth_enum = AnalysisDepth(depth)

        self.control_panel.set_analyzing(True)
        self.status_bar.showMessage("Analysis in progress...")
        self.log_message(f"Starting {depth} analysis on {self.video_info.path.name}", "INFO")

        # Use signals for thread-safe GUI updates
        def on_progress(progress: AnalysisProgress):
            self.progress_ready.emit(progress)

        def on_frame(frame: np.ndarray, detections: FrameDetections):
            self.frame_ready.emit(frame, detections)

        def on_complete(result: dict):
            self.analysis_complete.emit(result)

        try:
            self.processor.process_video_async(
                analysis_depth=depth_enum,
                progress_callback=on_progress,
                frame_callback=on_frame,
                completion_callback=on_complete
            )
        except Exception as e:
            self.log_message(f"Failed to start analysis: {e}", "ERROR")
            self.control_panel.set_analyzing(False)
            QMessageBox.critical(self, "Error", f"Failed to start analysis:\n{str(e)}")
    
    def stop_analysis(self):
        """Stop current analysis"""
        self.processor.stop_processing()
        self.control_panel.set_analyzing(False)
        self.status_bar.showMessage("Analysis stopped")
    
    def _toggle_log_panel(self, checked: bool):
        """Toggle the debug log panel visibility"""
        self.log_panel.setVisible(checked)
        self.log_toggle_btn.setText("Hide Debug Log" if checked else "Show Debug Log")

    def log_message(self, message: str, level: str = "INFO"):
        """Add message to log panel"""
        self.log_panel.log(message, level)

    def open_game_config(self):
        """Open the game configuration dialog"""
        dialog = GameConfigDialog(self)

        # Load existing config if any
        if hasattr(self, '_game_config'):
            dialog.set_config(self._game_config)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._game_config = dialog.get_config()
            self.log_message("Game configuration updated", "INFO")

            # Apply team colors to classifier
            if self.processor and self.processor.team_classifier:
                home_color = self._game_config["home_team"]["primary_color"]
                away_color = self._game_config["away_team"]["primary_color"]
                self.processor.team_classifier.set_team_colors(home_color, away_color)
                self.log_message(f"Team colors set: Home={home_color}, Away={away_color}", "INFO")

            # Update team color boxes in UI
            home_rgb = self._game_config["home_team"]["primary_color"]
            away_rgb = self._game_config["away_team"]["primary_color"]
            self.home_color_box.setStyleSheet(
                f"background-color: rgb({home_rgb[0]}, {home_rgb[1]}, {home_rgb[2]}); border: 1px solid #333;"
            )
            self.away_color_box.setStyleSheet(
                f"background-color: rgb({away_rgb[0]}, {away_rgb[1]}, {away_rgb[2]}); border: 1px solid #333;"
            )

            # Update video info label with team names
            if self._game_config["home_team"]["name"] or self._game_config["away_team"]["name"]:
                home_name = self._game_config["home_team"]["name"] or "Home"
                away_name = self._game_config["away_team"]["name"] or "Away"
                self.home_color_label.setText(f"{home_name}: ")
                self.away_color_label.setText(f"{away_name}: ")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Soccer Film Analysis",
            "<h2>Soccer Film Analysis</h2>"
            "<p>Version 1.0.0</p>"
            "<p>A comprehensive soccer game film analysis tool for coaches.</p>"
            "<p>Built with:</p>"
            "<ul>"
            "<li>Python + PyQt6</li>"
            "<li>Roboflow Sports</li>"
            "<li>YOLOv8 + Supervision</li>"
            "<li>PostgreSQL</li>"
            "</ul>"
            "<p>© 2025</p>"
        )
    
    def closeEvent(self, event):
        """Handle window close"""
        self.processor.release()
        event.accept()


def apply_dark_theme(app: QApplication):
    """Apply dark theme to application"""
    palette = QPalette()
    
    # Base colors
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(35, 35, 35))
    
    app.setPalette(palette)
    
    # Additional stylesheet with improved contrast
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
            color: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #ffffff;
        }
        QLabel {
            color: #ffffff;
        }
        QPushButton {
            background-color: #555555;
            color: #ffffff;
            border: 1px solid #666;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #666666;
        }
        QPushButton:pressed {
            background-color: #444444;
        }
        QPushButton:disabled {
            background-color: #3a3a3a;
            color: #888888;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 4px;
            text-align: center;
            color: #ffffff;
            background-color: #333;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #555;
            height: 8px;
            background: #333;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #4CAF50;
            border: 1px solid #333;
            width: 16px;
            margin: -4px 0;
            border-radius: 8px;
        }
        QComboBox {
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px;
            min-width: 100px;
            background-color: #444;
            color: #ffffff;
        }
        QComboBox::drop-down {
            border: none;
            background-color: #555;
        }
        QComboBox QAbstractItemView {
            background-color: #444;
            color: #ffffff;
            selection-background-color: #4CAF50;
        }
        QMenuBar {
            background-color: #353535;
            color: #ffffff;
        }
        QMenuBar::item:selected {
            background-color: #4CAF50;
        }
        QMenu {
            background-color: #353535;
            color: #ffffff;
            border: 1px solid #555;
        }
        QMenu::item:selected {
            background-color: #4CAF50;
        }
        QStatusBar {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        QMessageBox {
            background-color: #353535;
            color: #ffffff;
        }
        QMessageBox QLabel {
            color: #ffffff;
        }
        QMessageBox QPushButton {
            min-width: 80px;
        }
    """)


def main():
    """Main entry point"""
    # Setup logging
    logger.add(
        settings.get_logs_dir() / "app_{time}.log",
        rotation="10 MB",
        level=settings.log_level
    )
    
    app = QApplication(sys.argv)
    
    # Apply dark theme
    if settings.theme == "dark":
        apply_dark_theme(app)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
