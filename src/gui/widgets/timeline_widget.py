"""
Timeline and event widgets for video navigation and event marking
"""

from dataclasses import dataclass
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush


class EventType:
    """Soccer event types for timeline markers"""
    # Game period markers
    KICKOFF = "kickoff"
    HALFTIME_START = "halftime_start"
    SECOND_HALF = "second_half"
    GAME_END = "game_end"

    # Action events
    GOAL = "goal"
    SHOT = "shot"
    SAVE = "save"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    THROW_IN = "throw_in"
    GOAL_KICK = "goal_kick"
    PASS = "pass"
    STEAL = "steal"
    FOUL = "foul"
    OFFSIDE = "offside"
    SUBSTITUTION = "substitution"
    HALF_TIME = "half_time"
    CUSTOM = "custom"

    # Correction markers
    EXCLUDE_BALLBOY = "exclude_ballboy"
    CORRECT_TEAM = "correct_team"

    @staticmethod
    def get_color(event_type: str) -> str:
        """Get color for event type (hex format)"""
        colors = {
            # Game periods - bright distinctive colors
            "kickoff": "#00FF00",         # Bright green
            "halftime_start": "#FF00FF",  # Magenta
            "second_half": "#00FFFF",     # Cyan
            "game_end": "#FF0000",        # Red

            # Action events
            "goal": "#FFD700",        # Gold
            "shot": "#FF6B6B",        # Red
            "save": "#4ECDC4",         # Teal
            "corner": "#A8E6CF",       # Light green
            "free_kick": "#FFE66D",    # Yellow
            "throw_in": "#95E1D3",     # Mint
            "goal_kick": "#DDA0DD",    # Plum
            "pass": "#87CEEB",         # Sky blue
            "steal": "#FFA07A",        # Light salmon
            "foul": "#FF4444",         # Bright red
            "offside": "#9370DB",      # Medium purple
            "substitution": "#20B2AA", # Light sea green
            "half_time": "#FFFFFF",    # White
            "custom": "#888888",       # Gray

            # Corrections
            "exclude_ballboy": "#555555",
            "correct_team": "#AAAAAA",
        }
        return colors.get(event_type, "#888888")

    @staticmethod
    def get_label(event_type: str) -> str:
        """Get display label for event type"""
        labels = {
            # Game periods
            "kickoff": "KICKOFF",
            "halftime_start": "HALFTIME",
            "second_half": "2ND HALF",
            "game_end": "GAME END",

            # Actions
            "goal": "Goal",
            "shot": "Shot",
            "save": "Save",
            "corner": "Corner",
            "free_kick": "Free Kick",
            "throw_in": "Throw-in",
            "goal_kick": "Goal Kick",
            "pass": "Pass",
            "steal": "Steal",
            "foul": "Foul",
            "offside": "Offside",
            "substitution": "Sub",
            "half_time": "Half",
            "custom": "Event",
        }
        return labels.get(event_type, "Event")


@dataclass
class EventMarker:
    """Represents an event on the timeline"""
    frame: int
    event_type: str
    team_id: Optional[int] = None  # 0=home, 1=away, None=neutral
    player_id: Optional[int] = None
    description: str = ""
    timestamp_seconds: float = 0.0


class TimelineWidget(QWidget):
    """
    Custom timeline widget with event markers.
    Shows a slider for video navigation with colored markers for events.
    """

    seek_frame = pyqtSignal(int)
    marker_clicked = pyqtSignal(object)  # EventMarker

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)

        self._total_frames = 100
        self._current_frame = 0
        self._events: List[EventMarker] = []
        self._hovered_event: Optional[EventMarker] = None
        self._dragging = False

        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)

        # Colors
        self._bg_color = QColor("#1a1a1a")
        self._track_color = QColor("#333333")
        self._progress_color = QColor("#4CAF50")
        self._handle_color = QColor("#ffffff")

    def set_total_frames(self, total: int):
        """Set the total number of frames"""
        self._total_frames = max(1, total)
        self.update()

    def set_current_frame(self, frame: int):
        """Set current frame position"""
        self._current_frame = max(0, min(frame, self._total_frames - 1))
        self.update()

    def add_event(self, event: EventMarker):
        """Add an event marker"""
        self._events.append(event)
        self.update()

    def clear_events(self):
        """Clear all event markers"""
        self._events.clear()
        self.update()

    def set_events(self, events: List[EventMarker]):
        """Set all event markers"""
        self._events = events
        self.update()

    def get_events(self) -> List[EventMarker]:
        """Get all event markers"""
        return self._events.copy()

    def _frame_to_x(self, frame: int) -> float:
        """Convert frame number to x coordinate"""
        margin = 10
        usable_width = self.width() - 2 * margin
        return margin + (frame / max(1, self._total_frames - 1)) * usable_width

    def _x_to_frame(self, x: float) -> int:
        """Convert x coordinate to frame number"""
        margin = 10
        usable_width = self.width() - 2 * margin
        frame = int(((x - margin) / usable_width) * (self._total_frames - 1))
        return max(0, min(frame, self._total_frames - 1))

    def paintEvent(self, event):
        """Custom paint for timeline with markers"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), self._bg_color)

        margin = 10
        track_y = self.height() // 2
        track_height = 8
        marker_height = 20

        # Draw track background
        track_rect = self.rect().adjusted(margin, track_y - track_height // 2,
                                          -margin, -(self.height() - track_y - track_height // 2))
        painter.setBrush(QBrush(self._track_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(track_rect, 4, 4)

        # Draw progress
        progress_x = self._frame_to_x(self._current_frame)
        progress_rect = track_rect.adjusted(0, 0, int(progress_x - margin - track_rect.width()), 0)
        painter.setBrush(QBrush(self._progress_color))
        painter.drawRoundedRect(progress_rect, 4, 4)

        # Draw event markers
        for ev in self._events:
            x = self._frame_to_x(ev.frame)
            color = QColor(EventType.get_color(ev.event_type))

            # Marker line
            painter.setPen(QPen(color, 2))
            painter.drawLine(int(x), track_y - marker_height // 2,
                           int(x), track_y + marker_height // 2)

            # Marker dot
            painter.setBrush(QBrush(color))
            painter.drawEllipse(int(x) - 4, track_y - 4, 8, 8)

            # Tooltip for hovered event
            if ev == self._hovered_event:
                label = f"{EventType.get_label(ev.event_type)}"
                if ev.description:
                    label += f": {ev.description}"
                painter.setPen(QPen(QColor("#ffffff")))
                painter.drawText(int(x) - 50, track_y - marker_height - 5, label)

        # Draw handle
        handle_x = int(self._frame_to_x(self._current_frame))
        painter.setBrush(QBrush(self._handle_color))
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawEllipse(handle_x - 8, track_y - 8, 16, 16)

        # Draw time labels
        painter.setPen(QPen(QColor("#888888")))
        painter.drawText(margin, self.height() - 5, "0:00")
        if self._total_frames > 1:
            total_time = self._total_frames / 30  # Assume 30fps
            mins = int(total_time // 60)
            secs = int(total_time % 60)
            painter.drawText(self.width() - margin - 40, self.height() - 5, f"{mins}:{secs:02d}")

    def mousePressEvent(self, event):
        """Handle mouse press for seeking"""
        if event.button() == Qt.MouseButton.LeftButton:
            frame = self._x_to_frame(event.position().x())
            self._current_frame = frame
            self._dragging = True
            self.seek_frame.emit(frame)
            self.update()

            # Check if clicking on an event marker
            for ev in self._events:
                ev_x = self._frame_to_x(ev.frame)
                if abs(event.position().x() - ev_x) < 10:
                    self.marker_clicked.emit(ev)
                    break

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging and hover"""
        if self._dragging:
            frame = self._x_to_frame(event.position().x())
            self._current_frame = frame
            self.seek_frame.emit(frame)
            self.update()
        else:
            # Check for hovered event
            self._hovered_event = None
            for ev in self._events:
                ev_x = self._frame_to_x(ev.frame)
                if abs(event.position().x() - ev_x) < 10:
                    self._hovered_event = ev
                    break
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        self._dragging = False

    def leaveEvent(self, event):
        """Handle mouse leave"""
        self._hovered_event = None
        self.update()


class EventListWidget(QWidget):
    """Widget displaying a list of detected events with filtering"""

    event_selected = pyqtSignal(object)  # EventMarker

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self._events: List[EventMarker] = []

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))

        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Events", None)
        self.filter_combo.addItem("Goals", EventType.GOAL)
        self.filter_combo.addItem("Shots", EventType.SHOT)
        self.filter_combo.addItem("Saves", EventType.SAVE)
        self.filter_combo.addItem("Set Pieces", "set_pieces")
        self.filter_combo.addItem("Passes", EventType.PASS)
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()

        layout.addLayout(filter_layout)

        # Event list
        self.event_list = QTextEdit()
        self.event_list.setReadOnly(True)
        self.event_list.setMaximumHeight(150)
        self.event_list.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: Consolas, monospace;
                font-size: 11px;
                border: 1px solid #333;
            }
        """)
        layout.addWidget(self.event_list)

    def set_events(self, events: List[EventMarker]):
        """Set the event list"""
        self._events = events
        self._apply_filter()

    def add_event(self, event: EventMarker):
        """Add a single event"""
        self._events.append(event)
        self._apply_filter()

    def _apply_filter(self):
        """Apply current filter to event list"""
        filter_value = self.filter_combo.currentData()
        self.event_list.clear()

        for ev in self._events:
            if filter_value is None:
                show = True
            elif filter_value == "set_pieces":
                show = ev.event_type in [EventType.CORNER, EventType.FREE_KICK,
                                         EventType.THROW_IN, EventType.GOAL_KICK]
            else:
                show = ev.event_type == filter_value

            if show:
                color = EventType.get_color(ev.event_type)
                mins = int(ev.timestamp_seconds // 60)
                secs = int(ev.timestamp_seconds % 60)
                label = EventType.get_label(ev.event_type)
                desc = f" - {ev.description}" if ev.description else ""
                self.event_list.append(
                    f'<span style="color: {color}">[{mins}:{secs:02d}] {label}{desc}</span>'
                )
