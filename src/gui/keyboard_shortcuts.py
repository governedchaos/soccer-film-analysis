"""
Soccer Film Analysis - Keyboard Shortcuts
Centralized keyboard shortcut management with customization support
"""

from typing import Dict, Callable, Optional, List
from dataclasses import dataclass
from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut, QAction
from PyQt6.QtWidgets import QWidget, QDialog, QVBoxLayout, QHBoxLayout, \
    QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QLabel, \
    QKeySequenceEdit, QMessageBox, QGroupBox
from loguru import logger


@dataclass
class ShortcutAction:
    """Represents a keyboard shortcut action"""
    id: str
    name: str
    description: str
    default_key: str
    current_key: str
    category: str
    callback: Optional[Callable] = None


class KeyboardShortcutManager(QObject):
    """
    Manages all keyboard shortcuts for the application.
    Provides centralized registration, customization, and conflict detection.
    """

    shortcut_triggered = pyqtSignal(str)  # Emits action ID

    # Default shortcuts organized by category
    DEFAULT_SHORTCUTS = {
        # Video Playback
        "playback": [
            ("play_pause", "Play/Pause", "Toggle video playback", "Space"),
            ("step_forward", "Step Forward", "Advance one frame", "Right"),
            ("step_backward", "Step Backward", "Go back one frame", "Left"),
            ("skip_forward", "Skip Forward", "Skip 5 seconds forward", "Shift+Right"),
            ("skip_backward", "Skip Backward", "Skip 5 seconds backward", "Shift+Left"),
            ("goto_start", "Go to Start", "Jump to beginning", "Home"),
            ("goto_end", "Go to End", "Jump to end", "End"),
            ("speed_up", "Speed Up", "Increase playback speed", "]"),
            ("speed_down", "Speed Down", "Decrease playback speed", "["),
            ("normal_speed", "Normal Speed", "Reset to 1x speed", "\\"),
        ],

        # Analysis
        "analysis": [
            ("start_analysis", "Start Analysis", "Begin video analysis", "Ctrl+R"),
            ("stop_analysis", "Stop Analysis", "Stop current analysis", "Escape"),
            ("export_results", "Export Results", "Export analysis results", "Ctrl+Shift+E"),
            ("open_corrections", "Open Corrections", "Open correction dialog", "Ctrl+E"),
        ],

        # Event Markers (Game Periods)
        "events": [
            ("mark_kickoff", "Mark Kickoff", "Mark game kickoff", "K"),
            ("mark_halftime", "Mark Halftime", "Mark halftime start", "H"),
            ("mark_secondhalf", "Mark 2nd Half", "Mark second half start", "Shift+H"),
            ("mark_gameend", "Mark Game End", "Mark end of game", "Shift+K"),
            ("mark_goal", "Mark Goal", "Mark a goal", "G"),
            ("mark_shot", "Mark Shot", "Mark a shot", "S"),
            ("mark_save", "Mark Save", "Mark a save", "Shift+S"),
            ("mark_foul", "Mark Foul", "Mark a foul", "F"),
            ("mark_corner", "Mark Corner", "Mark a corner kick", "C"),
            ("mark_custom", "Custom Event", "Add custom event marker", "M"),
            ("delete_event", "Delete Event", "Delete nearest event", "Delete"),
        ],

        # View
        "view": [
            ("toggle_fullscreen", "Toggle Fullscreen", "Toggle fullscreen mode", "F11"),
            ("toggle_log", "Toggle Log Panel", "Show/hide debug log", "Ctrl+L"),
            ("toggle_stats", "Toggle Stats", "Show/hide stats panel", "Ctrl+I"),
            ("zoom_in", "Zoom In", "Zoom into video", "Ctrl+="),
            ("zoom_out", "Zoom Out", "Zoom out of video", "Ctrl+-"),
            ("zoom_reset", "Reset Zoom", "Reset to fit view", "Ctrl+0"),
        ],

        # File
        "file": [
            ("open_video", "Open Video", "Open video file", "Ctrl+O"),
            ("save_project", "Save Project", "Save current project", "Ctrl+S"),
            ("load_project", "Load Project", "Load saved project", "Ctrl+Shift+O"),
            ("game_settings", "Game Settings", "Open game configuration", "Ctrl+G"),
            ("load_roster", "Load Roster", "Load team roster CSV", "Ctrl+Shift+R"),
        ],

        # Navigation
        "navigation": [
            ("next_event", "Next Event", "Jump to next event", "N"),
            ("prev_event", "Previous Event", "Jump to previous event", "P"),
            ("next_goal", "Next Goal", "Jump to next goal", "Shift+G"),
            ("goto_frame", "Go to Frame", "Jump to specific frame", "Ctrl+J"),
            ("goto_time", "Go to Time", "Jump to specific time", "Ctrl+T"),
        ],
    }

    def __init__(self, parent_widget: QWidget):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.actions: Dict[str, ShortcutAction] = {}
        self.shortcuts: Dict[str, QShortcut] = {}
        self._load_default_shortcuts()

    def _load_default_shortcuts(self):
        """Load default shortcuts from configuration"""
        for category, shortcuts in self.DEFAULT_SHORTCUTS.items():
            for shortcut_tuple in shortcuts:
                action_id, name, desc, default_key = shortcut_tuple
                self.actions[action_id] = ShortcutAction(
                    id=action_id,
                    name=name,
                    description=desc,
                    default_key=default_key,
                    current_key=default_key,
                    category=category
                )

    def register_action(
        self,
        action_id: str,
        callback: Callable,
        override_key: Optional[str] = None
    ) -> bool:
        """
        Register a callback for an action.

        Args:
            action_id: The action identifier
            callback: Function to call when shortcut triggered
            override_key: Optional custom key sequence

        Returns:
            True if registered successfully
        """
        if action_id not in self.actions:
            logger.warning(f"Unknown action: {action_id}")
            return False

        action = self.actions[action_id]
        action.callback = callback

        key = override_key or action.current_key

        # Remove existing shortcut if any
        if action_id in self.shortcuts:
            self.shortcuts[action_id].deleteLater()

        # Create new shortcut
        try:
            shortcut = QShortcut(QKeySequence(key), self.parent_widget)
            shortcut.activated.connect(lambda aid=action_id: self._on_shortcut(aid))
            self.shortcuts[action_id] = shortcut

            logger.debug(f"Registered shortcut: {action_id} -> {key}")
            return True

        except Exception as e:
            logger.error(f"Failed to register shortcut {action_id}: {e}")
            return False

    def _on_shortcut(self, action_id: str):
        """Handle shortcut activation"""
        action = self.actions.get(action_id)
        if action and action.callback:
            try:
                action.callback()
                self.shortcut_triggered.emit(action_id)
            except Exception as e:
                logger.error(f"Error executing shortcut {action_id}: {e}")

    def set_key(self, action_id: str, new_key: str) -> bool:
        """
        Change the key binding for an action.

        Args:
            action_id: Action to modify
            new_key: New key sequence string

        Returns:
            True if changed successfully
        """
        if action_id not in self.actions:
            return False

        # Check for conflicts
        conflict = self._check_conflict(new_key, action_id)
        if conflict:
            logger.warning(f"Key {new_key} conflicts with {conflict}")
            return False

        action = self.actions[action_id]
        action.current_key = new_key

        # Re-register with new key
        if action.callback:
            return self.register_action(action_id, action.callback, new_key)

        return True

    def _check_conflict(self, key: str, exclude_id: str) -> Optional[str]:
        """Check if a key sequence conflicts with existing shortcuts"""
        for action_id, action in self.actions.items():
            if action_id != exclude_id and action.current_key == key:
                return action_id
        return None

    def reset_to_defaults(self):
        """Reset all shortcuts to default values"""
        for action in self.actions.values():
            action.current_key = action.default_key
            if action.callback:
                self.register_action(action.id, action.callback)

    def get_actions_by_category(self, category: str) -> List[ShortcutAction]:
        """Get all actions in a category"""
        return [a for a in self.actions.values() if a.category == category]

    def get_all_categories(self) -> List[str]:
        """Get list of all categories"""
        return list(self.DEFAULT_SHORTCUTS.keys())

    def export_config(self) -> Dict[str, str]:
        """Export current shortcut configuration"""
        return {
            action_id: action.current_key
            for action_id, action in self.actions.items()
        }

    def import_config(self, config: Dict[str, str]):
        """Import shortcut configuration"""
        for action_id, key in config.items():
            if action_id in self.actions:
                self.set_key(action_id, key)


class ShortcutEditorDialog(QDialog):
    """
    Dialog for viewing and customizing keyboard shortcuts.
    """

    def __init__(self, manager: KeyboardShortcutManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(600, 500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        info_label = QLabel(
            "Double-click on a shortcut to change it. "
            "Press Escape to cancel editing."
        )
        info_label.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(info_label)

        # Create table for each category
        for category in self.manager.get_all_categories():
            actions = self.manager.get_actions_by_category(category)
            if not actions:
                continue

            group = QGroupBox(category.replace("_", " ").title())
            group_layout = QVBoxLayout(group)

            table = QTableWidget(len(actions), 3)
            table.setHorizontalHeaderLabels(["Action", "Shortcut", "Description"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
            table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            table.setColumnWidth(1, 120)
            table.verticalHeader().setVisible(False)

            for row, action in enumerate(actions):
                # Action name
                name_item = QTableWidgetItem(action.name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                name_item.setData(Qt.ItemDataRole.UserRole, action.id)
                table.setItem(row, 0, name_item)

                # Shortcut key
                key_item = QTableWidgetItem(action.current_key)
                table.setItem(row, 1, key_item)

                # Description
                desc_item = QTableWidgetItem(action.description)
                desc_item.setFlags(desc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, 2, desc_item)

            table.cellChanged.connect(lambda r, c, t=table: self._on_cell_changed(t, r, c))
            group_layout.addWidget(table)
            layout.addWidget(group)

        # Buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _on_cell_changed(self, table: QTableWidget, row: int, col: int):
        """Handle shortcut cell edit"""
        if col != 1:  # Only handle shortcut column
            return

        action_id = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        new_key = table.item(row, 1).text()

        if not self.manager.set_key(action_id, new_key):
            # Revert to current key
            action = self.manager.actions[action_id]
            table.item(row, 1).setText(action.current_key)
            QMessageBox.warning(
                self, "Invalid Shortcut",
                f"Could not set shortcut to '{new_key}'.\n"
                "It may conflict with another shortcut."
            )

    def _reset_defaults(self):
        """Reset all shortcuts to defaults"""
        reply = QMessageBox.question(
            self, "Reset Shortcuts",
            "Reset all shortcuts to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.manager.reset_to_defaults()
            self.accept()
            # Re-open to refresh UI
            dialog = ShortcutEditorDialog(self.manager, self.parent())
            dialog.exec()


def setup_default_shortcuts(window, shortcut_manager: KeyboardShortcutManager):
    """
    Setup all default shortcuts for the main window.

    Args:
        window: MainWindow instance
        shortcut_manager: KeyboardShortcutManager instance
    """
    # Video Playback
    shortcut_manager.register_action("play_pause", window.toggle_play_pause)
    shortcut_manager.register_action("step_forward", lambda: window.seek_frame(1))
    shortcut_manager.register_action("step_backward", lambda: window.seek_frame(-1))
    shortcut_manager.register_action("skip_forward", lambda: window.seek_seconds(5))
    shortcut_manager.register_action("skip_backward", lambda: window.seek_seconds(-5))
    shortcut_manager.register_action("goto_start", lambda: window.seek_frame_absolute(0))
    shortcut_manager.register_action("goto_end", window.goto_end)

    # Analysis
    shortcut_manager.register_action("start_analysis", lambda: window.start_analysis("standard"))
    shortcut_manager.register_action("stop_analysis", window.stop_analysis)
    shortcut_manager.register_action("export_results", window.export_results)
    shortcut_manager.register_action("open_corrections", window.open_correction_dialog)

    # Event Markers
    shortcut_manager.register_action("mark_kickoff", lambda: window.quick_mark_event("kickoff"))
    shortcut_manager.register_action("mark_halftime", lambda: window.quick_mark_event("halftime_start"))
    shortcut_manager.register_action("mark_secondhalf", lambda: window.quick_mark_event("second_half"))
    shortcut_manager.register_action("mark_gameend", lambda: window.quick_mark_event("game_end"))
    shortcut_manager.register_action("mark_goal", lambda: window.quick_mark_event("goal"))
    shortcut_manager.register_action("mark_shot", lambda: window.quick_mark_event("shot"))
    shortcut_manager.register_action("mark_save", lambda: window.quick_mark_event("save"))
    shortcut_manager.register_action("mark_foul", lambda: window.quick_mark_event("foul"))
    shortcut_manager.register_action("mark_corner", lambda: window.quick_mark_event("corner"))
    shortcut_manager.register_action("mark_custom", window.add_custom_event)

    # Navigation
    shortcut_manager.register_action("next_event", window.goto_next_event)
    shortcut_manager.register_action("prev_event", window.goto_prev_event)

    # View
    shortcut_manager.register_action("toggle_fullscreen", window.toggle_fullscreen)
    shortcut_manager.register_action("toggle_log", window.toggle_log_panel)

    # File
    shortcut_manager.register_action("open_video", window.load_video)
    shortcut_manager.register_action("game_settings", window.open_game_config)
    shortcut_manager.register_action("load_roster", window.load_roster)

    logger.info("Keyboard shortcuts initialized")
