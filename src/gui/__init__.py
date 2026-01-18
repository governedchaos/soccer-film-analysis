"""
Soccer Film Analysis - GUI Module
PyQt6-based graphical user interface components.
"""

from src.gui.stats_widget import EnhancedStatsPanel

# Main window import is deferred to avoid circular imports
# Use: from src.gui.main_window import SoccerFilmAnalysisWindow

__all__ = [
    "EnhancedStatsPanel",
]
