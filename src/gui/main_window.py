"""
Soccer Film Analysis - Main Window
PyQt6-based GUI for the soccer film analysis application

Refactored to use separate widget and controller modules for better organization.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QLabel, QFileDialog,
    QStatusBar, QMessageBox, QGroupBox, QGridLayout, QDialog,
    QScrollArea, QFormLayout, QLineEdit, QComboBox, QCheckBox,
    QDoubleSpinBox, QDialogButtonBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QPalette, QColor
from loguru import logger

from config import settings, AnalysisDepth
from src.core.video_processor import ThreadedVideoProcessor, VideoInfo, AnalysisProgress
from src.detection.detector import FrameDetections

# Import widgets from the widgets module
from src.gui.widgets import (
    ColorButton, VideoWidget, LogPanel, StatsPanel, ControlPanel,
    EventMarker
)

# Import controllers
from src.gui.controllers import VideoController, AnalysisController


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
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

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


class MainWindow(QMainWindow):
    """
    Main application window.

    Uses VideoController and AnalysisController to separate logic from UI.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Soccer Film Analysis")
        self.setMinimumSize(1200, 800)

        # Initialize processor and controllers
        self.processor = ThreadedVideoProcessor()
        self.video_controller = VideoController(self.processor)
        self.analysis_controller = AnalysisController(self.processor)

        # Game configuration
        self._game_config = {}

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

        self.load_video_btn = QPushButton("Load Video")
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

        self.config_btn = QPushButton("Game Settings")
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

        self.export_btn = QPushButton("Export Report")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)

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

        analysis_menu.addSeparator()

        export_action = QAction("E&xport Results...", self)
        export_action.setShortcut("Ctrl+Shift+E")
        export_action.triggered.connect(self.export_results)
        analysis_menu.addAction(export_action)

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
        # Control panel signals
        self.control_panel.play_clicked.connect(self.video_controller.play)
        self.control_panel.pause_clicked.connect(self.video_controller.pause)
        self.control_panel.seek_frame.connect(self._on_seek_frame)
        self.control_panel.analysis_started.connect(self.start_analysis)
        self.control_panel.analysis_stopped.connect(self.analysis_controller.stop_analysis)

        # Video controller signals
        self.video_controller.video_loaded.connect(self._on_video_loaded)
        self.video_controller.frame_changed.connect(self._on_frame_changed)
        self.video_controller.playback_state_changed.connect(self.control_panel.set_playing)
        self.video_controller.error_occurred.connect(self._on_error)

        # Analysis controller signals
        self.analysis_controller.analysis_started.connect(
            lambda: self.control_panel.set_analyzing(True)
        )
        self.analysis_controller.analysis_progress.connect(self._on_analysis_progress)
        self.analysis_controller.frame_analyzed.connect(self._on_frame_analyzed)
        self.analysis_controller.analysis_completed.connect(self._on_analysis_completed)
        self.analysis_controller.analysis_stopped.connect(
            lambda: self.control_panel.set_analyzing(False)
        )
        self.analysis_controller.analysis_failed.connect(self._on_analysis_failed)

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

        self.log_message(f"Loading video: {file_path}", "INFO")

        if self.video_controller.load_video(file_path):
            # Auto-calibrate team colors
            self.status_bar.showMessage("Calibrating team colors...")
            self.log_message("Calibrating team colors from sample frames...", "INFO")
            self.video_controller.calibrate_teams()
            self.log_message("Team color calibration complete", "INFO")
            self.status_bar.showMessage(f"Loaded: {Path(file_path).name} (ready)")

    def _on_video_loaded(self, video_info: VideoInfo):
        """Handle video loaded signal"""
        self.control_panel.set_frame_range(video_info.total_frames)
        self.control_panel.update_frame_label(0, video_info.total_frames)

        info_text = (
            f"<b>{video_info.path.name}</b><br>"
            f"Resolution: {video_info.width}x{video_info.height}<br>"
            f"FPS: {video_info.fps:.1f}<br>"
            f"Duration: {video_info.duration_seconds/60:.1f} min<br>"
            f"Frames: {video_info.total_frames:,}"
        )
        self.video_info_label.setText(info_text)

        self.log_message(
            f"Video info: {video_info.width}x{video_info.height}, "
            f"{video_info.fps:.1f} FPS, {video_info.total_frames} frames",
            "INFO"
        )

        # Show first frame
        self.video_controller.seek(0)

    def _on_frame_changed(self, frame_number: int, frame):
        """Handle frame changed signal"""
        if frame is not None:
            self.video_widget.display_frame(frame)

        self.control_panel.set_current_frame(frame_number)
        total = self.video_controller.total_frames
        self.control_panel.update_frame_label(frame_number, total)

        # Update stats if we have detections for this frame
        detections = self.video_controller.get_current_detections()
        possession = self.video_controller.get_possession()
        self.stats_panel.update_stats(
            frame_number, self.video_controller.fps, detections, possession
        )

    def _on_seek_frame(self, delta_or_absolute: int):
        """Handle seek request from control panel"""
        if abs(delta_or_absolute) < 100:  # Likely a delta
            self.video_controller.seek_relative(delta_or_absolute)
        else:
            self.video_controller.seek(delta_or_absolute)

    def start_analysis(self, depth: str):
        """Start video analysis"""
        if self.video_controller.video_info is None:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        self.status_bar.showMessage("Analysis in progress...")
        self.log_message(f"Starting {depth} analysis", "INFO")
        self.analysis_controller.start_analysis(depth)

    def _on_analysis_progress(self, progress: AnalysisProgress):
        """Handle analysis progress update"""
        self.control_panel.update_progress(progress)
        if progress.current_frame % 100 == 0:
            self.log_message(
                f"Frame {progress.current_frame}/{progress.total_frames} "
                f"({progress.percentage:.1f}%) - {progress.frames_per_second:.1f} FPS",
                "DEBUG"
            )

    def _on_frame_analyzed(self, frame: np.ndarray, detections: FrameDetections):
        """Handle analyzed frame"""
        self.video_widget.display_frame(frame)
        possession = self.analysis_controller.get_possession()
        self.stats_panel.update_stats(
            detections.frame_number, self.video_controller.fps, detections, possession
        )

        if detections.frame_number % 50 == 0:
            player_count = len(detections.players)
            ball_detected = "Yes" if detections.ball else "No"
            self.log_message(
                f"Frame {detections.frame_number}: {player_count} players, ball: {ball_detected}",
                "DEBUG"
            )

    def _on_analysis_completed(self, result: dict):
        """Handle analysis completion"""
        self.control_panel.set_analyzing(False)
        msg = (
            f"Analysis complete! {result.get('processed_frames', 0)} frames "
            f"in {result.get('elapsed_seconds', 0)/60:.1f} min"
        )
        self.status_bar.showMessage(msg)
        self.log_message(msg, "INFO")
        self.export_btn.setEnabled(True)

        QMessageBox.information(
            self, "Analysis Complete",
            f"Analysis finished successfully!\n"
            f"Processed {result.get('processed_frames', 0)} frames in "
            f"{result.get('elapsed_seconds', 0)/60:.1f} minutes."
        )

    def _on_analysis_failed(self, error: str):
        """Handle analysis failure"""
        self.control_panel.set_analyzing(False)
        self.status_bar.showMessage(f"Analysis failed: {error}")
        self.log_message(f"Analysis failed: {error}", "ERROR")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{error}")

    def _on_error(self, error: str):
        """Handle general errors"""
        self.log_message(error, "ERROR")
        QMessageBox.critical(self, "Error", error)

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

        if self._game_config:
            dialog.set_config(self._game_config)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._game_config = dialog.get_config()
            self.log_message("Game configuration updated", "INFO")
            self._apply_game_config()

    def _apply_game_config(self):
        """Apply game configuration to processor"""
        # Apply team colors
        home_color = self._game_config["home_team"]["primary_color"]
        away_color = self._game_config["away_team"]["primary_color"]
        self.video_controller.set_team_colors(home_color, away_color)
        self.log_message(f"Team colors set: Home={home_color}, Away={away_color}", "INFO")

        # Apply referee and goalkeeper colors
        if self.processor.detector:
            ref_color = self._game_config.get("referee_color")
            if ref_color:
                self.processor.detector.set_referee_colors([ref_color])

            home_gk = self._game_config["home_team"].get("goalkeeper_color")
            away_gk = self._game_config["away_team"].get("goalkeeper_color")
            self.processor.detector.set_goalkeeper_colors(home_gk, away_gk)

        # Update UI color boxes
        self.home_color_box.setStyleSheet(
            f"background-color: rgb({home_color[0]}, {home_color[1]}, {home_color[2]}); "
            "border: 1px solid #333;"
        )
        self.away_color_box.setStyleSheet(
            f"background-color: rgb({away_color[0]}, {away_color[1]}, {away_color[2]}); "
            "border: 1px solid #333;"
        )

        # Update team name labels
        home_name = self._game_config["home_team"]["name"] or "Home"
        away_name = self._game_config["away_team"]["name"] or "Away"
        self.home_color_label.setText(f"{home_name}: ")
        self.away_color_label.setText(f"{away_name}: ")

    def export_results(self):
        """Export analysis results"""
        if not self.analysis_controller.has_results:
            QMessageBox.warning(
                self, "No Analysis",
                "Please run analysis first before exporting."
            )
            return

        from src.analysis.export import AnalysisExporter

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis Results",
            "analysis_report.json",
            "JSON Files (*.json);;CSV Files (*.csv);;Text Reports (*.txt);;All Files (*)"
        )

        if not filepath:
            return

        try:
            exporter = AnalysisExporter()

            events = [
                {
                    "frame": e.frame,
                    "timestamp": f"{int(e.timestamp_seconds//60):02d}:{int(e.timestamp_seconds%60):02d}",
                    "event_type": e.event_type,
                    "description": e.description,
                    "team_id": e.team_id
                }
                for e in self.control_panel.get_events()
            ]

            possession = self.analysis_controller.get_possession()
            stats = self.analysis_controller.get_detection_stats()

            summary = {
                "total_frames_analyzed": stats.get("frames_processed", 0),
                "ball_detected_frames": stats.get("frames_with_ball", 0),
                "ball_detection_rate": f"{stats.get('ball_detection_rate', 0):.1f}%",
                "possession": {"home": possession[0], "away": possession[1]},
                "total_events": len(events)
            }

            if filepath.endswith('.json'):
                exporter.export_json(self._game_config, events, summary, Path(filepath).name)
            elif filepath.endswith('.csv'):
                exporter.export_csv(events, Path(filepath).name)
            else:
                exporter.export_summary_report(
                    self._game_config, possession, [], {}, events, Path(filepath).name
                )

            self.log_message(f"Exported to: {filepath}", "INFO")
            QMessageBox.information(self, "Export Complete", f"Results exported to:\n{filepath}")

        except Exception as e:
            self.log_message(f"Export failed: {e}", "ERROR")
            QMessageBox.critical(self, "Export Failed", f"Failed to export:\n{str(e)}")

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
            "<li>YOLOv8 (Local models - no API key needed)</li>"
            "<li>Supervision + ByteTrack</li>"
            "<li>PostgreSQL</li>"
            "</ul>"
            "<p>2026</p>"
        )

    def closeEvent(self, event):
        """Handle window close"""
        self.video_controller.release()
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

    # Additional stylesheet
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
        QLabel { color: #ffffff; }
        QPushButton {
            background-color: #555555;
            color: #ffffff;
            border: 1px solid #666;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #666666; }
        QPushButton:pressed { background-color: #444444; }
        QPushButton:disabled { background-color: #3a3a3a; color: #888888; }
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
        QComboBox {
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px;
            min-width: 100px;
            background-color: #444;
            color: #ffffff;
        }
        QMenuBar { background-color: #353535; color: #ffffff; }
        QMenuBar::item:selected { background-color: #4CAF50; }
        QMenu { background-color: #353535; color: #ffffff; border: 1px solid #555; }
        QMenu::item:selected { background-color: #4CAF50; }
        QStatusBar { background-color: #2a2a2a; color: #ffffff; }
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
