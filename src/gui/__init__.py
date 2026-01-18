"""
Soccer Film Analysis - GUI Module
PyQt6-based graphical user interface components.
"""

from src.gui.stats_widget import EnhancedStatsPanel

# Keyboard shortcuts
from src.gui.keyboard_shortcuts import (
    KeyboardShortcutManager,
    ShortcutEditorDialog,
    ShortcutAction,
    setup_default_shortcuts
)

# Correction dialog
from src.gui.correction_dialog import (
    CorrectionDialog,
    CorrectionManager,
    DetectionCorrection
)

# Main window import is deferred to avoid circular imports
# Use: from src.gui.main_window import SoccerFilmAnalysisWindow

__all__ = [
    "EnhancedStatsPanel",
    # Keyboard shortcuts
    "KeyboardShortcutManager",
    "ShortcutEditorDialog",
    "ShortcutAction",
    "setup_default_shortcuts",
    # Correction dialog
    "CorrectionDialog",
    "CorrectionManager",
    "DetectionCorrection",
]
