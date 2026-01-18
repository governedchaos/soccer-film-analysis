"""
GUI Widgets for Soccer Film Analysis
"""

from .color_button import ColorButton
from .video_widget import VideoWidget
from .log_panel import LogPanel
from .stats_panel import StatsPanel
from .timeline_widget import TimelineWidget, EventMarker, EventType, EventListWidget
from .control_panel import ControlPanel

__all__ = [
    "ColorButton",
    "VideoWidget",
    "LogPanel",
    "StatsPanel",
    "TimelineWidget",
    "EventMarker",
    "EventType",
    "EventListWidget",
    "ControlPanel",
]
