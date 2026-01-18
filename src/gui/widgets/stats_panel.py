"""
Detection statistics panel widget
"""

from typing import Optional, Tuple
from PyQt6.QtWidgets import QGroupBox, QGridLayout, QLabel

from src.detection.detector import FrameDetections


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

    def update_stats(
        self,
        frame_num: int,
        fps: float,
        detections: Optional[FrameDetections],
        possession: Optional[Tuple[float, float]] = None
    ):
        """Update displayed statistics"""
        time_str = f"{frame_num / fps:.1f}s" if fps > 0 else "-"

        self.stat_labels["frame"].setText(str(frame_num))
        self.stat_labels["time"].setText(time_str)

        if detections:
            total_players = len(detections.players) + len(detections.goalkeepers)
            home_count = sum(1 for p in detections.players if p.team_id == 0)
            away_count = sum(1 for p in detections.players if p.team_id == 1)

            self.stat_labels["players"].setText(str(total_players))
            self.stat_labels["home"].setText(
                str(home_count + sum(1 for g in detections.goalkeepers if g.team_id == 0))
            )
            self.stat_labels["away"].setText(
                str(away_count + sum(1 for g in detections.goalkeepers if g.team_id == 1))
            )

            # Ball detection status with confidence
            if detections.ball:
                ball_conf = f"[checkmark] ({detections.ball.confidence:.0%})"
                self.stat_labels["ball"].setText(ball_conf)
                self.stat_labels["ball"].setStyleSheet("color: #00FF00;")
            else:
                self.stat_labels["ball"].setText("[X] Not detected")
                self.stat_labels["ball"].setStyleSheet("color: #FF4444;")

            self.stat_labels["referees"].setText(str(len(detections.referees)))
        else:
            for key in ["players", "home", "away", "ball", "referees"]:
                self.stat_labels[key].setText("-")

        # Update possession
        if possession:
            home_pct, away_pct = possession
            self.stat_labels["possession"].setText(f"{home_pct:.0f}% / {away_pct:.0f}%")
        else:
            self.stat_labels["possession"].setText("-")
