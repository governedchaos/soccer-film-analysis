"""
Enhanced Real-Time Statistics Widget
Displays comprehensive match statistics including possession, shots, passes, etc.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QGroupBox, QProgressBar, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field
from collections import defaultdict

from src.detection.detector import FrameDetections


@dataclass
class MatchStats:
    """Accumulated match statistics."""
    # Possession
    home_possession_frames: int = 0
    away_possession_frames: int = 0

    # Detection counts
    total_frames_processed: int = 0
    ball_detected_frames: int = 0

    # Shot tracking (simplified - based on ball position near goal)
    home_shots: int = 0
    away_shots: int = 0
    home_shots_on_target: int = 0
    away_shots_on_target: int = 0

    # Pass tracking (simplified - based on ball movement between players)
    home_passes: int = 0
    away_passes: int = 0
    home_passes_completed: int = 0
    away_passes_completed: int = 0

    # Player counts per frame (for averaging)
    home_player_counts: List[int] = field(default_factory=list)
    away_player_counts: List[int] = field(default_factory=list)

    # Ball position history for analysis
    ball_positions: List[Tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)

    # Possession by zone (attacking third, middle, defensive)
    home_attacking_third: int = 0
    home_middle_third: int = 0
    home_defensive_third: int = 0
    away_attacking_third: int = 0
    away_middle_third: int = 0
    away_defensive_third: int = 0

    def get_possession_pct(self) -> Tuple[float, float]:
        """Get possession percentage for each team."""
        total = self.home_possession_frames + self.away_possession_frames
        if total == 0:
            return 50.0, 50.0
        return (
            (self.home_possession_frames / total) * 100,
            (self.away_possession_frames / total) * 100
        )

    def get_ball_detection_rate(self) -> float:
        """Get percentage of frames with ball detected."""
        if self.total_frames_processed == 0:
            return 0.0
        return (self.ball_detected_frames / self.total_frames_processed) * 100

    def get_pass_accuracy(self, team: int) -> float:
        """Get pass completion percentage for a team."""
        if team == 0:
            total = self.home_passes
            completed = self.home_passes_completed
        else:
            total = self.away_passes
            completed = self.away_passes_completed

        if total == 0:
            return 0.0
        return (completed / total) * 100


class StatLabel(QWidget):
    """A stat display with label and value."""

    def __init__(self, label: str, value: str = "-", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)

        self.label = QLabel(label)
        self.label.setStyleSheet("font-weight: bold; color: #888;")

        self.value = QLabel(value)
        self.value.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.value)

        self.setLayout(layout)

    def set_value(self, value: str, color: str = "#4CAF50"):
        """Update the displayed value."""
        self.value.setText(value)
        self.value.setStyleSheet(f"color: {color}; font-weight: bold;")


class PossessionBar(QWidget):
    """Visual possession indicator bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Labels row
        labels_layout = QHBoxLayout()
        self.home_label = QLabel("Home 50%")
        self.home_label.setStyleSheet("color: #FFC107; font-weight: bold;")
        self.away_label = QLabel("50% Away")
        self.away_label.setStyleSheet("color: #F44336; font-weight: bold;")
        self.away_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        labels_layout.addWidget(self.home_label)
        labels_layout.addStretch()
        labels_layout.addWidget(self.away_label)

        layout.addLayout(labels_layout)

        # Progress bar
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(50)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(12)
        self.bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                border-radius: 5px;
                background-color: #F44336;
            }
            QProgressBar::chunk {
                background-color: #FFC107;
                border-radius: 4px;
            }
        """)

        layout.addWidget(self.bar)
        self.setLayout(layout)

    def set_possession(self, home_pct: float, away_pct: float):
        """Update possession display."""
        self.bar.setValue(int(home_pct))
        self.home_label.setText(f"Home {home_pct:.0f}%")
        self.away_label.setText(f"{away_pct:.0f}% Away")


class TeamStatsBox(QGroupBox):
    """Stats box for one team."""

    def __init__(self, team_name: str, color: str, parent=None):
        super().__init__(team_name, parent)
        self.color = color
        self.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {color};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {color};
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(4)

        self.players_stat = StatLabel("Players:", "0")
        self.shots_stat = StatLabel("Shots:", "0")
        self.shots_on_target = StatLabel("On Target:", "0")
        self.passes_stat = StatLabel("Passes:", "0")
        self.pass_accuracy = StatLabel("Pass %:", "0%")

        layout.addWidget(self.players_stat)
        layout.addWidget(self.shots_stat)
        layout.addWidget(self.shots_on_target)
        layout.addWidget(self.passes_stat)
        layout.addWidget(self.pass_accuracy)

        self.setLayout(layout)

    def update_stats(
        self,
        players: int,
        shots: int,
        on_target: int,
        passes: int,
        pass_pct: float
    ):
        """Update all team stats."""
        self.players_stat.set_value(str(players), self.color)
        self.shots_stat.set_value(str(shots), self.color)
        self.shots_on_target.set_value(str(on_target), self.color)
        self.passes_stat.set_value(str(passes), self.color)
        self.pass_accuracy.set_value(f"{pass_pct:.0f}%", self.color)


class EnhancedStatsPanel(QWidget):
    """
    Enhanced statistics panel showing comprehensive real-time match stats.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stats = MatchStats()
        self.last_ball_pos: Optional[Tuple[float, float]] = None
        self.last_possession_team: Optional[int] = None

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Frame/Time info
        info_group = QGroupBox("Match Info")
        info_layout = QGridLayout()

        self.frame_label = StatLabel("Frame:", "0")
        self.time_label = StatLabel("Time:", "0:00")
        self.fps_label = StatLabel("FPS:", "-")
        self.ball_label = StatLabel("Ball:", "Not detected")

        info_layout.addWidget(self.frame_label, 0, 0)
        info_layout.addWidget(self.time_label, 0, 1)
        info_layout.addWidget(self.fps_label, 1, 0)
        info_layout.addWidget(self.ball_label, 1, 1)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Possession bar
        possession_group = QGroupBox("Possession")
        poss_layout = QVBoxLayout()
        self.possession_bar = PossessionBar()
        poss_layout.addWidget(self.possession_bar)
        possession_group.setLayout(poss_layout)
        layout.addWidget(possession_group)

        # Team stats side by side
        teams_layout = QHBoxLayout()

        self.home_box = TeamStatsBox("Home", "#FFC107")
        self.away_box = TeamStatsBox("Away", "#F44336")

        teams_layout.addWidget(self.home_box)
        teams_layout.addWidget(self.away_box)

        layout.addLayout(teams_layout)

        # Detection quality
        quality_group = QGroupBox("Detection Quality")
        quality_layout = QVBoxLayout()

        self.ball_detection_rate = StatLabel("Ball Detection:", "0%")
        self.referee_count = StatLabel("Referees:", "0")
        self.total_detections = StatLabel("Total Tracked:", "0")

        quality_layout.addWidget(self.ball_detection_rate)
        quality_layout.addWidget(self.referee_count)
        quality_layout.addWidget(self.total_detections)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        # Add stretch to push everything up
        layout.addStretch()

        self.setLayout(layout)

    def update_from_detections(
        self,
        frame_num: int,
        fps: float,
        detections: Optional[FrameDetections],
        possession: Optional[Tuple[float, float]] = None
    ):
        """Update stats from frame detections."""
        # Update frame info
        time_secs = frame_num / fps if fps > 0 else 0
        mins = int(time_secs // 60)
        secs = int(time_secs % 60)

        self.frame_label.set_value(str(frame_num))
        self.time_label.set_value(f"{mins}:{secs:02d}")
        self.fps_label.set_value(f"{fps:.1f}")

        if detections is None:
            return

        # Update accumulated stats
        self.stats.total_frames_processed += 1

        # Ball detection
        if detections.ball is not None:
            self.stats.ball_detected_frames += 1
            cx, cy = detections.ball.center
            self.stats.ball_positions.append((frame_num, cx, cy))

            conf = detections.ball.confidence
            self.ball_label.set_value(f"Detected ({conf:.0%})", "#00FF00")
        else:
            self.ball_label.set_value("Not detected", "#FF4444")

        # Count players per team
        home_count = sum(1 for p in detections.players if p.team_id == 0)
        away_count = sum(1 for p in detections.players if p.team_id == 1)

        # Add goalkeepers
        home_count += sum(1 for g in detections.goalkeepers if g.team_id == 0)
        away_count += sum(1 for g in detections.goalkeepers if g.team_id == 1)

        self.stats.home_player_counts.append(home_count)
        self.stats.away_player_counts.append(away_count)

        # Update possession
        if possession:
            home_pct, away_pct = possession
            self.possession_bar.set_possession(home_pct, away_pct)

            # Accumulate possession frames
            if home_pct > away_pct:
                self.stats.home_possession_frames += 1
            else:
                self.stats.away_possession_frames += 1

        # Update team boxes
        self.home_box.update_stats(
            players=home_count,
            shots=self.stats.home_shots,
            on_target=self.stats.home_shots_on_target,
            passes=self.stats.home_passes,
            pass_pct=self.stats.get_pass_accuracy(0)
        )

        self.away_box.update_stats(
            players=away_count,
            shots=self.stats.away_shots,
            on_target=self.stats.away_shots_on_target,
            passes=self.stats.away_passes,
            pass_pct=self.stats.get_pass_accuracy(1)
        )

        # Update detection quality
        ball_rate = self.stats.get_ball_detection_rate()
        color = "#00FF00" if ball_rate > 80 else "#FFC107" if ball_rate > 50 else "#FF4444"
        self.ball_detection_rate.set_value(f"{ball_rate:.0f}%", color)

        self.referee_count.set_value(str(len(detections.referees)))

        total_tracked = len(detections.players) + len(detections.goalkeepers) + len(detections.referees)
        self.total_detections.set_value(str(total_tracked))

    def record_shot(self, team_id: int, on_target: bool = False):
        """Record a shot for a team."""
        if team_id == 0:
            self.stats.home_shots += 1
            if on_target:
                self.stats.home_shots_on_target += 1
        else:
            self.stats.away_shots += 1
            if on_target:
                self.stats.away_shots_on_target += 1

    def record_pass(self, team_id: int, completed: bool = True):
        """Record a pass for a team."""
        if team_id == 0:
            self.stats.home_passes += 1
            if completed:
                self.stats.home_passes_completed += 1
        else:
            self.stats.away_passes += 1
            if completed:
                self.stats.away_passes_completed += 1

    def reset(self):
        """Reset all statistics."""
        self.stats = MatchStats()
        self.last_ball_pos = None
        self.last_possession_team = None

        self.frame_label.set_value("0")
        self.time_label.set_value("0:00")
        self.possession_bar.set_possession(50, 50)
        self.home_box.update_stats(0, 0, 0, 0, 0)
        self.away_box.update_stats(0, 0, 0, 0, 0)
        self.ball_detection_rate.set_value("0%")
        self.referee_count.set_value("0")
        self.total_detections.set_value("0")

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get a summary of all stats for export."""
        home_pct, away_pct = self.stats.get_possession_pct()

        return {
            'possession': {
                'home': home_pct,
                'away': away_pct
            },
            'shots': {
                'home': self.stats.home_shots,
                'away': self.stats.away_shots,
                'home_on_target': self.stats.home_shots_on_target,
                'away_on_target': self.stats.away_shots_on_target
            },
            'passes': {
                'home': self.stats.home_passes,
                'away': self.stats.away_passes,
                'home_completed': self.stats.home_passes_completed,
                'away_completed': self.stats.away_passes_completed,
                'home_accuracy': self.stats.get_pass_accuracy(0),
                'away_accuracy': self.stats.get_pass_accuracy(1)
            },
            'detection': {
                'total_frames': self.stats.total_frames_processed,
                'ball_detected_frames': self.stats.ball_detected_frames,
                'ball_detection_rate': self.stats.get_ball_detection_rate()
            }
        }
