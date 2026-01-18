"""
Video playback and analysis control panel widget
"""

from typing import List

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QProgressBar, QFrame, QInputDialog
)
from PyQt6.QtCore import pyqtSignal

from src.core.video_processor import AnalysisProgress
from .timeline_widget import TimelineWidget, EventMarker, EventType


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

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._on_play_clicked)

        self.prev_frame_btn = QPushButton("<<")
        self.prev_frame_btn.clicked.connect(lambda: self.seek_frame.emit(-10))

        self.next_frame_btn = QPushButton(">>")
        self.next_frame_btn.clicked.connect(lambda: self.seek_frame.emit(10))

        playback_layout.addWidget(self.prev_frame_btn)
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.next_frame_btn)

        # Timeline with event markers
        timeline_layout = QVBoxLayout()
        self.frame_label = QLabel("0 / 0")

        # Custom timeline widget with event markers
        self.timeline_widget = TimelineWidget()
        self.timeline_widget.seek_frame.connect(self._on_timeline_seek)
        self.timeline_widget.marker_clicked.connect(self._on_marker_clicked)

        # Frame label layout
        frame_info_layout = QHBoxLayout()
        frame_info_layout.addWidget(self.frame_label)
        frame_info_layout.addStretch()

        # Add event button
        self.add_event_btn = QPushButton("+ Add Event")
        self.add_event_btn.setFixedWidth(100)
        self.add_event_btn.clicked.connect(self._on_add_event_clicked)
        frame_info_layout.addWidget(self.add_event_btn)

        timeline_layout.addWidget(self.timeline_widget)
        timeline_layout.addLayout(frame_info_layout)

        # Quick game period marker buttons
        period_layout = QHBoxLayout()
        period_layout.addWidget(QLabel("Mark Period:"))

        self.kickoff_btn = QPushButton("Kickoff")
        self.kickoff_btn.setStyleSheet("background-color: #00AA00; color: white; padding: 4px 8px;")
        self.kickoff_btn.clicked.connect(lambda: self._quick_mark_period(EventType.KICKOFF, "Game Start"))

        self.halftime_btn = QPushButton("Halftime")
        self.halftime_btn.setStyleSheet("background-color: #AA00AA; color: white; padding: 4px 8px;")
        self.halftime_btn.clicked.connect(lambda: self._quick_mark_period(EventType.HALFTIME_START, "Halftime"))

        self.secondhalf_btn = QPushButton("2nd Half")
        self.secondhalf_btn.setStyleSheet("background-color: #00AAAA; color: white; padding: 4px 8px;")
        self.secondhalf_btn.clicked.connect(lambda: self._quick_mark_period(EventType.SECOND_HALF, "Second Half"))

        self.gameend_btn = QPushButton("End")
        self.gameend_btn.setStyleSheet("background-color: #AA0000; color: white; padding: 4px 8px;")
        self.gameend_btn.clicked.connect(lambda: self._quick_mark_period(EventType.GAME_END, "Game End"))

        period_layout.addWidget(self.kickoff_btn)
        period_layout.addWidget(self.halftime_btn)
        period_layout.addWidget(self.secondhalf_btn)
        period_layout.addWidget(self.gameend_btn)
        period_layout.addStretch()

        timeline_layout.addLayout(period_layout)

        # Keep legacy slider reference for compatibility
        self.timeline_slider = self.timeline_widget

        # Analysis controls
        analysis_layout = QHBoxLayout()

        self.depth_combo = QComboBox()
        self.depth_combo.addItems(["Quick", "Standard", "Deep"])
        self.depth_combo.setCurrentIndex(1)

        self.analyze_btn = QPushButton("Start Analysis")
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

        self.stop_btn = QPushButton("Stop")
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
        layout.addLayout(timeline_layout)
        layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        layout.addLayout(analysis_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)

        self.setLayout(layout)

        # Event storage
        self._events: List[EventMarker] = []

    def _on_play_clicked(self):
        if self._is_playing:
            self._is_playing = False
            self.play_btn.setText("Play")
            self.pause_clicked.emit()
        else:
            self._is_playing = True
            self.play_btn.setText("Pause")
            self.play_clicked.emit()

    def _on_timeline_seek(self, frame: int):
        """Handle seek from timeline widget"""
        self.seek_frame.emit(frame)

    def _on_marker_clicked(self, event: EventMarker):
        """Handle click on event marker"""
        self.seek_frame.emit(event.frame)

    def _on_add_event_clicked(self):
        """Show dialog to add manual event at current position"""
        current_frame = self.timeline_widget._current_frame

        # Show event type selection - game periods at top
        event_types = [
            "--- GAME PERIODS ---",
            "Kickoff (Game Start)",
            "Halftime Start",
            "Second Half Start",
            "Game End",
            "--- ACTIONS ---",
            "Goal", "Shot", "Save", "Corner", "Free Kick",
            "Throw-in", "Goal Kick", "Pass", "Steal", "Foul",
            "--- CORRECTIONS ---",
            "Exclude Ball Boy",
            "Custom"
        ]
        event_type, ok = QInputDialog.getItem(
            self, "Add Event", "Select event type:",
            event_types, 1, False  # Default to Kickoff
        )

        if ok and event_type and not event_type.startswith("---"):
            # Map to EventType constants
            type_map = {
                # Game periods
                "Kickoff (Game Start)": EventType.KICKOFF,
                "Halftime Start": EventType.HALFTIME_START,
                "Second Half Start": EventType.SECOND_HALF,
                "Game End": EventType.GAME_END,
                # Actions
                "Goal": EventType.GOAL, "Shot": EventType.SHOT,
                "Save": EventType.SAVE, "Corner": EventType.CORNER,
                "Free Kick": EventType.FREE_KICK, "Throw-in": EventType.THROW_IN,
                "Goal Kick": EventType.GOAL_KICK, "Pass": EventType.PASS,
                "Steal": EventType.STEAL, "Foul": EventType.FOUL,
                # Corrections
                "Exclude Ball Boy": EventType.EXCLUDE_BALLBOY,
                "Custom": EventType.CUSTOM
            }
            event_type_code = type_map.get(event_type, EventType.CUSTOM)

            # Get optional description
            desc, _ = QInputDialog.getText(
                self, "Event Description",
                "Enter description (optional):"
            )

            # Create and add event
            event = EventMarker(
                frame=current_frame,
                event_type=event_type_code,
                description=desc,
                timestamp_seconds=current_frame / 30.0  # Assume 30fps
            )
            self.add_event(event)

    def _quick_mark_period(self, event_type: str, description: str):
        """Quick-add a game period marker at current position"""
        current_frame = self.timeline_widget._current_frame
        event = EventMarker(
            frame=current_frame,
            event_type=event_type,
            description=description,
            timestamp_seconds=current_frame / 30.0  # Will be updated with actual fps
        )
        self.add_event(event)

    def add_event(self, event: EventMarker):
        """Add an event to the timeline"""
        self._events.append(event)
        self.timeline_widget.add_event(event)

    def get_events(self) -> List[EventMarker]:
        """Get all events"""
        return self._events.copy()

    def clear_events(self):
        """Clear all events"""
        self._events.clear()
        self.timeline_widget.clear_events()

    def _on_slider_changed(self, value: int):
        self.seek_frame.emit(value)

    def _on_analyze_clicked(self):
        depth = self.depth_combo.currentText().lower()
        self.analysis_started.emit(depth)

    def set_frame_range(self, total_frames: int):
        """Set the timeline range based on video length"""
        self.timeline_widget.set_total_frames(total_frames)

    def set_current_frame(self, frame: int):
        """Update the timeline position"""
        self.timeline_widget.set_current_frame(frame)

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
            self.analyze_btn.setText("Analysis Running...")
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
            self.analyze_btn.setText("Start Analysis")
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
            self.play_btn.setText("Pause")
        else:
            self.play_btn.setText("Play")
