"""
Soccer Film Analysis - Post-Analysis Correction Dialog
Allows users to review and correct detection results
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, field
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QGroupBox, QListWidget, QListWidgetItem, QSplitter,
    QWidget, QComboBox, QCheckBox, QSpinBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor
from loguru import logger


@dataclass
class DetectionCorrection:
    """Represents a correction to a detection"""
    frame_number: int
    tracker_id: int
    correction_type: str  # 'exclude', 'change_team', 'change_role', 'delete'
    original_value: any = None
    new_value: any = None
    reason: str = ""


class CorrectionManager:
    """
    Manages all corrections made to analysis results.
    """

    def __init__(self):
        self.corrections: List[DetectionCorrection] = []
        self.excluded_trackers: Set[int] = set()  # Permanently excluded tracker IDs
        self.team_overrides: Dict[int, int] = {}  # tracker_id -> team_id
        self.role_overrides: Dict[int, str] = {}  # tracker_id -> role
        self.excluded_frames: Dict[int, Set[int]] = {}  # frame -> set of tracker_ids

    def exclude_tracker(self, tracker_id: int, reason: str = ""):
        """Permanently exclude a tracker (e.g., ball boy)"""
        self.excluded_trackers.add(tracker_id)
        self.corrections.append(DetectionCorrection(
            frame_number=-1,  # All frames
            tracker_id=tracker_id,
            correction_type='exclude',
            reason=reason
        ))
        logger.info(f"Excluded tracker {tracker_id}: {reason}")

    def exclude_detection_at_frame(self, frame_number: int, tracker_id: int, reason: str = ""):
        """Exclude a specific detection at a specific frame"""
        if frame_number not in self.excluded_frames:
            self.excluded_frames[frame_number] = set()
        self.excluded_frames[frame_number].add(tracker_id)
        self.corrections.append(DetectionCorrection(
            frame_number=frame_number,
            tracker_id=tracker_id,
            correction_type='exclude_frame',
            reason=reason
        ))

    def change_team(self, tracker_id: int, new_team_id: int):
        """Change the team assignment for a tracker"""
        old_team = self.team_overrides.get(tracker_id)
        self.team_overrides[tracker_id] = new_team_id
        self.corrections.append(DetectionCorrection(
            frame_number=-1,
            tracker_id=tracker_id,
            correction_type='change_team',
            original_value=old_team,
            new_value=new_team_id
        ))
        logger.info(f"Changed tracker {tracker_id} to team {new_team_id}")

    def change_role(self, tracker_id: int, new_role: str):
        """Change the role for a tracker (player/goalkeeper/referee)"""
        old_role = self.role_overrides.get(tracker_id)
        self.role_overrides[tracker_id] = new_role
        self.corrections.append(DetectionCorrection(
            frame_number=-1,
            tracker_id=tracker_id,
            correction_type='change_role',
            original_value=old_role,
            new_value=new_role
        ))
        logger.info(f"Changed tracker {tracker_id} role to {new_role}")

    def is_excluded(self, tracker_id: int, frame_number: Optional[int] = None) -> bool:
        """Check if a tracker is excluded"""
        if tracker_id in self.excluded_trackers:
            return True
        if frame_number and frame_number in self.excluded_frames:
            return tracker_id in self.excluded_frames[frame_number]
        return False

    def get_team_override(self, tracker_id: int) -> Optional[int]:
        """Get team override for a tracker"""
        return self.team_overrides.get(tracker_id)

    def get_role_override(self, tracker_id: int) -> Optional[str]:
        """Get role override for a tracker"""
        return self.role_overrides.get(tracker_id)

    def undo_last(self) -> bool:
        """Undo the last correction"""
        if not self.corrections:
            return False

        correction = self.corrections.pop()

        if correction.correction_type == 'exclude':
            self.excluded_trackers.discard(correction.tracker_id)
        elif correction.correction_type == 'exclude_frame':
            if correction.frame_number in self.excluded_frames:
                self.excluded_frames[correction.frame_number].discard(correction.tracker_id)
        elif correction.correction_type == 'change_team':
            if correction.original_value is not None:
                self.team_overrides[correction.tracker_id] = correction.original_value
            else:
                self.team_overrides.pop(correction.tracker_id, None)
        elif correction.correction_type == 'change_role':
            if correction.original_value is not None:
                self.role_overrides[correction.tracker_id] = correction.original_value
            else:
                self.role_overrides.pop(correction.tracker_id, None)

        logger.info(f"Undid correction: {correction.correction_type}")
        return True

    def get_correction_count(self) -> int:
        """Get total number of corrections"""
        return len(self.corrections)

    def export_corrections(self) -> Dict:
        """Export corrections as a dictionary"""
        return {
            "excluded_trackers": list(self.excluded_trackers),
            "team_overrides": self.team_overrides.copy(),
            "role_overrides": self.role_overrides.copy(),
            "excluded_frames": {k: list(v) for k, v in self.excluded_frames.items()},
            "corrections": [
                {
                    "frame": c.frame_number,
                    "tracker_id": c.tracker_id,
                    "type": c.correction_type,
                    "reason": c.reason
                }
                for c in self.corrections
            ]
        }

    def import_corrections(self, data: Dict):
        """Import corrections from a dictionary"""
        self.excluded_trackers = set(data.get("excluded_trackers", []))
        self.team_overrides = data.get("team_overrides", {})
        self.role_overrides = data.get("role_overrides", {})
        self.excluded_frames = {
            int(k): set(v) for k, v in data.get("excluded_frames", {}).items()
        }


class CorrectionDialog(QDialog):
    """
    Dialog for reviewing and correcting analysis results.
    """

    corrections_applied = pyqtSignal(object)  # CorrectionManager

    def __init__(self, processor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.correction_manager = CorrectionManager()
        self.current_frame = 0
        self.selected_tracker_id = None

        self.setWindowTitle("Review & Correct Detections")
        self.setMinimumSize(1200, 800)
        self.setup_ui()
        self.load_frame(0)

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - Video frame with detections
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Frame display
        self.frame_label = QLabel()
        self.frame_label.setMinimumSize(800, 450)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self.frame_label.mousePressEvent = self._on_frame_click
        left_layout.addWidget(self.frame_label)

        # Frame navigation
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("◀◀ -10")
        self.prev_btn.clicked.connect(lambda: self.navigate_frames(-10))

        self.prev_one_btn = QPushButton("◀ -1")
        self.prev_one_btn.clicked.connect(lambda: self.navigate_frames(-1))

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)

        self.next_one_btn = QPushButton("+1 ▶")
        self.next_one_btn.clicked.connect(lambda: self.navigate_frames(1))

        self.next_btn = QPushButton("+10 ▶▶")
        self.next_btn.clicked.connect(lambda: self.navigate_frames(10))

        self.frame_info_label = QLabel("Frame 0 / 0")

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.prev_one_btn)
        nav_layout.addWidget(self.frame_slider)
        nav_layout.addWidget(self.next_one_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.frame_info_label)

        left_layout.addLayout(nav_layout)

        # Instructions
        instructions = QLabel(
            "Click on a detection in the frame to select it, then use the controls on the right to make corrections."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; font-style: italic;")
        left_layout.addWidget(instructions)

        splitter.addWidget(left_widget)

        # Right side - Detection list and correction controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Detection list
        detection_group = QGroupBox("Detections in Frame")
        detection_layout = QVBoxLayout()

        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(5)
        self.detection_table.setHorizontalHeaderLabels(["ID", "Type", "Team", "Conf", "Status"])
        self.detection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.detection_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.detection_table.itemSelectionChanged.connect(self._on_detection_selected)
        detection_layout.addWidget(self.detection_table)

        detection_group.setLayout(detection_layout)
        right_layout.addWidget(detection_group)

        # Correction controls
        correction_group = QGroupBox("Correction Actions")
        correction_layout = QVBoxLayout()

        # Selected detection info
        self.selected_label = QLabel("No detection selected")
        self.selected_label.setStyleSheet("font-weight: bold;")
        correction_layout.addWidget(self.selected_label)

        correction_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))

        # Action buttons
        self.exclude_btn = QPushButton("🚫 Exclude (Ball Boy/Spectator)")
        self.exclude_btn.clicked.connect(self._exclude_selected)
        self.exclude_btn.setEnabled(False)
        correction_layout.addWidget(self.exclude_btn)

        self.exclude_frame_btn = QPushButton("🚫 Exclude This Frame Only")
        self.exclude_frame_btn.clicked.connect(self._exclude_selected_frame)
        self.exclude_frame_btn.setEnabled(False)
        correction_layout.addWidget(self.exclude_frame_btn)

        # Team change
        team_layout = QHBoxLayout()
        team_layout.addWidget(QLabel("Change Team:"))
        self.team_combo = QComboBox()
        self.team_combo.addItems(["Home (0)", "Away (1)", "Unknown"])
        self.team_combo.setEnabled(False)
        team_layout.addWidget(self.team_combo)
        self.apply_team_btn = QPushButton("Apply")
        self.apply_team_btn.clicked.connect(self._change_team)
        self.apply_team_btn.setEnabled(False)
        team_layout.addWidget(self.apply_team_btn)
        correction_layout.addLayout(team_layout)

        # Role change
        role_layout = QHBoxLayout()
        role_layout.addWidget(QLabel("Change Role:"))
        self.role_combo = QComboBox()
        self.role_combo.addItems(["Player", "Goalkeeper", "Referee"])
        self.role_combo.setEnabled(False)
        role_layout.addWidget(self.role_combo)
        self.apply_role_btn = QPushButton("Apply")
        self.apply_role_btn.clicked.connect(self._change_role)
        self.apply_role_btn.setEnabled(False)
        role_layout.addWidget(self.apply_role_btn)
        correction_layout.addLayout(role_layout)

        correction_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))

        # Undo button
        self.undo_btn = QPushButton("↩️ Undo Last Correction")
        self.undo_btn.clicked.connect(self._undo_last)
        correction_layout.addWidget(self.undo_btn)

        correction_group.setLayout(correction_layout)
        right_layout.addWidget(correction_group)

        # Correction summary
        summary_group = QGroupBox("Correction Summary")
        summary_layout = QVBoxLayout()
        self.summary_label = QLabel("No corrections made")
        summary_layout.addWidget(self.summary_label)

        self.excluded_list = QListWidget()
        self.excluded_list.setMaximumHeight(100)
        summary_layout.addWidget(QLabel("Excluded Trackers:"))
        summary_layout.addWidget(self.excluded_list)

        summary_group.setLayout(summary_layout)
        right_layout.addWidget(summary_group)

        right_layout.addStretch()

        splitter.addWidget(right_widget)
        splitter.setSizes([800, 400])

        layout.addWidget(splitter)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.save_btn = QPushButton("💾 Save Corrections")
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.save_btn.clicked.connect(self._save_corrections)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

        # Initialize slider range
        if self.processor.video_info:
            self.frame_slider.setMaximum(self.processor.video_info.total_frames - 1)

    def navigate_frames(self, delta: int):
        """Navigate frames by delta"""
        new_frame = self.current_frame + delta
        new_frame = max(0, min(new_frame, self.frame_slider.maximum()))
        self.load_frame(new_frame)

    def _on_slider_changed(self, value: int):
        """Handle slider change"""
        self.load_frame(value)

    def load_frame(self, frame_number: int):
        """Load and display a specific frame"""
        self.current_frame = frame_number
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_number)
        self.frame_slider.blockSignals(False)

        # Get frame and detections
        frame = self.processor.get_frame(frame_number)
        if frame is None:
            return

        detections = self.processor.frame_detections.get(frame_number)

        # Draw detections with correction highlights
        annotated = self._draw_annotated_frame(frame, detections)

        # Convert to QPixmap
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to fit
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.frame_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.frame_label.setPixmap(scaled)

        # Update frame info
        total = self.processor.video_info.total_frames if self.processor.video_info else 0
        self.frame_info_label.setText(f"Frame {frame_number} / {total}")

        # Update detection table
        self._update_detection_table(detections)

    def _draw_annotated_frame(self, frame: np.ndarray, detections) -> np.ndarray:
        """Draw frame with detection boxes and correction indicators"""
        annotated = frame.copy()

        if detections is None:
            return annotated

        # Draw all players
        all_detections = (
            [(p, "player") for p in detections.players] +
            [(g, "goalkeeper") for g in detections.goalkeepers] +
            [(r, "referee") for r in detections.referees]
        )

        for detection, role in all_detections:
            tracker_id = detection.tracker_id
            x1, y1, x2, y2 = map(int, detection.bbox)

            # Determine color based on status
            if self.correction_manager.is_excluded(tracker_id, self.current_frame):
                color = (100, 100, 100)  # Gray for excluded
                thickness = 1
            elif tracker_id == self.selected_tracker_id:
                color = (0, 255, 255)  # Yellow for selected
                thickness = 3
            elif role == "referee":
                color = (0, 0, 0)  # Black for referee
                thickness = 2
            elif role == "goalkeeper":
                color = (0, 255, 0)  # Green for goalkeeper
                thickness = 2
            else:
                # Team color
                team_id = self.correction_manager.get_team_override(tracker_id)
                if team_id is None:
                    team_id = detection.team_id

                if team_id == 0:
                    color = (0, 255, 255)  # Yellow for home
                elif team_id == 1:
                    color = (0, 0, 255)  # Red for away
                else:
                    color = (128, 128, 128)  # Gray for unknown
                thickness = 2

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"#{tracker_id}" if tracker_id else "?"
            if self.correction_manager.is_excluded(tracker_id, self.current_frame):
                label += " [EXCL]"

            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw ball
        if detections.ball:
            cx, cy = map(int, detections.ball.center)
            cv2.circle(annotated, (cx, cy), 10, (0, 165, 255), -1)
            cv2.circle(annotated, (cx, cy), 12, (255, 255, 255), 2)

        return annotated

    def _update_detection_table(self, detections):
        """Update the detection table"""
        self.detection_table.setRowCount(0)

        if detections is None:
            return

        all_detections = (
            [(p, "Player") for p in detections.players] +
            [(g, "Goalkeeper") for g in detections.goalkeepers] +
            [(r, "Referee") for r in detections.referees]
        )

        for detection, role in all_detections:
            row = self.detection_table.rowCount()
            self.detection_table.insertRow(row)

            tracker_id = detection.tracker_id or -1

            # ID
            id_item = QTableWidgetItem(str(tracker_id))
            id_item.setData(Qt.ItemDataRole.UserRole, tracker_id)
            self.detection_table.setItem(row, 0, id_item)

            # Type
            actual_role = self.correction_manager.get_role_override(tracker_id) or role
            self.detection_table.setItem(row, 1, QTableWidgetItem(actual_role))

            # Team
            team_id = self.correction_manager.get_team_override(tracker_id)
            if team_id is None:
                team_id = detection.team_id
            team_str = "Home" if team_id == 0 else "Away" if team_id == 1 else "?"
            self.detection_table.setItem(row, 2, QTableWidgetItem(team_str))

            # Confidence
            conf_str = f"{detection.confidence:.0%}"
            self.detection_table.setItem(row, 3, QTableWidgetItem(conf_str))

            # Status
            if self.correction_manager.is_excluded(tracker_id, self.current_frame):
                status = "EXCLUDED"
                color = QColor(255, 100, 100)
            else:
                status = "OK"
                color = QColor(100, 255, 100)

            status_item = QTableWidgetItem(status)
            status_item.setBackground(color)
            self.detection_table.setItem(row, 4, status_item)

    def _on_detection_selected(self):
        """Handle detection selection in table"""
        rows = self.detection_table.selectedItems()
        if not rows:
            self.selected_tracker_id = None
            self._update_controls(False)
            return

        row = rows[0].row()
        tracker_id = self.detection_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.selected_tracker_id = tracker_id

        self._update_controls(True)
        self.selected_label.setText(f"Selected: Tracker #{tracker_id}")

        # Refresh frame to show selection
        self.load_frame(self.current_frame)

    def _on_frame_click(self, event):
        """Handle click on frame to select detection"""
        # This would require coordinate mapping - simplified for now
        pass

    def _update_controls(self, enabled: bool):
        """Enable/disable correction controls"""
        self.exclude_btn.setEnabled(enabled)
        self.exclude_frame_btn.setEnabled(enabled)
        self.team_combo.setEnabled(enabled)
        self.apply_team_btn.setEnabled(enabled)
        self.role_combo.setEnabled(enabled)
        self.apply_role_btn.setEnabled(enabled)

    def _exclude_selected(self):
        """Exclude selected tracker from all frames"""
        if self.selected_tracker_id is None:
            return

        self.correction_manager.exclude_tracker(
            self.selected_tracker_id,
            "Manually excluded (ball boy/spectator)"
        )
        self._update_summary()
        self.load_frame(self.current_frame)

    def _exclude_selected_frame(self):
        """Exclude selected tracker from current frame only"""
        if self.selected_tracker_id is None:
            return

        self.correction_manager.exclude_detection_at_frame(
            self.current_frame,
            self.selected_tracker_id,
            "Excluded from single frame"
        )
        self._update_summary()
        self.load_frame(self.current_frame)

    def _change_team(self):
        """Change team assignment for selected tracker"""
        if self.selected_tracker_id is None:
            return

        team_map = {"Home (0)": 0, "Away (1)": 1, "Unknown": -1}
        new_team = team_map.get(self.team_combo.currentText(), -1)

        self.correction_manager.change_team(self.selected_tracker_id, new_team)
        self._update_summary()
        self.load_frame(self.current_frame)

    def _change_role(self):
        """Change role for selected tracker"""
        if self.selected_tracker_id is None:
            return

        role_map = {"Player": "player", "Goalkeeper": "goalkeeper", "Referee": "referee"}
        new_role = role_map.get(self.role_combo.currentText(), "player")

        self.correction_manager.change_role(self.selected_tracker_id, new_role)
        self._update_summary()
        self.load_frame(self.current_frame)

    def _undo_last(self):
        """Undo the last correction"""
        if self.correction_manager.undo_last():
            self._update_summary()
            self.load_frame(self.current_frame)

    def _update_summary(self):
        """Update the correction summary display"""
        count = self.correction_manager.get_correction_count()
        self.summary_label.setText(f"Total corrections: {count}")

        self.excluded_list.clear()
        for tracker_id in self.correction_manager.excluded_trackers:
            self.excluded_list.addItem(f"Tracker #{tracker_id}")

    def _save_corrections(self):
        """Save and apply corrections"""
        self.corrections_applied.emit(self.correction_manager)
        QMessageBox.information(
            self, "Corrections Saved",
            f"Applied {self.correction_manager.get_correction_count()} corrections.\n"
            "The analysis will be updated with these corrections."
        )
        self.accept()
