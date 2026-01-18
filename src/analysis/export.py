"""
Soccer Film Analysis - Export Module
Generates reports and exports analysis data in various formats
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from loguru import logger


class AnalysisExporter:
    """
    Exports analysis results to various formats.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Args:
            output_dir: Directory for output files (default: data/outputs)
        """
        from config import settings
        self.output_dir = output_dir or settings.get_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        game_data: Dict,
        events: List[Dict],
        detections_summary: Dict,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export analysis to JSON format.

        Args:
            game_data: Game metadata (teams, date, etc.)
            events: List of detected/marked events
            detections_summary: Summary statistics

        Returns:
            Path to the output file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"analysis_{timestamp}.json"
        output_path = self.output_dir / filename

        export_data = {
            "export_info": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
                "format": "soccer_film_analysis_v1"
            },
            "game": game_data,
            "events": events,
            "summary": detections_summary
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported JSON to: {output_path}")
        return output_path

    def export_csv(
        self,
        events: List[Dict],
        filename: Optional[str] = None
    ) -> Path:
        """
        Export events to CSV format.

        Args:
            events: List of events to export

        Returns:
            Path to the output file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"events_{timestamp}.csv"
        output_path = self.output_dir / filename

        if not events:
            # Create empty file with headers
            headers = ["timestamp", "event_type", "team", "player", "description"]
        else:
            headers = list(events[0].keys())

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(events)

        logger.info(f"Exported CSV to: {output_path}")
        return output_path

    def export_summary_report(
        self,
        game_data: Dict,
        possession: tuple,
        player_stats: List[Dict],
        team_stats: Dict,
        events: List[Dict],
        filename: Optional[str] = None
    ) -> Path:
        """
        Export a human-readable summary report.

        Returns:
            Path to the output file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"report_{timestamp}.txt"
        output_path = self.output_dir / filename

        home_team = game_data.get("home_team", {}).get("name", "Home")
        away_team = game_data.get("away_team", {}).get("name", "Away")
        home_pct, away_pct = possession

        lines = [
            "=" * 60,
            "SOCCER FILM ANALYSIS - GAME REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 60,
            "GAME INFORMATION",
            "-" * 60,
            f"Home Team: {home_team}",
            f"Away Team: {away_team}",
            f"Competition: {game_data.get('competition', 'N/A')}",
            f"Venue: {game_data.get('venue', 'N/A')}",
            "",
            "-" * 60,
            "POSSESSION",
            "-" * 60,
            f"{home_team}: {home_pct:.1f}%",
            f"{away_team}: {away_pct:.1f}%",
            "",
            "-" * 60,
            "TEAM STATISTICS",
            "-" * 60,
        ]

        for team_name, stats in team_stats.items():
            lines.append(f"\n{team_name}:")
            for stat_name, value in stats.items():
                lines.append(f"  {stat_name}: {value}")

        lines.extend([
            "",
            "-" * 60,
            "EVENTS TIMELINE",
            "-" * 60,
        ])

        for event in events:
            time_str = event.get("timestamp", "00:00")
            event_type = event.get("event_type", "unknown")
            desc = event.get("description", "")
            team = event.get("team", "")

            line = f"[{time_str}] {event_type.upper()}"
            if team:
                line += f" - {team}"
            if desc:
                line += f": {desc}"
            lines.append(line)

        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Exported report to: {output_path}")
        return output_path

    def generate_debug_log(
        self,
        detections_by_frame: Dict[int, Any],
        events: List[Dict],
        errors: List[str],
        filename: Optional[str] = None
    ) -> Path:
        """
        Generate a detailed debug log for troubleshooting.

        Returns:
            Path to the output file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"debug_log_{timestamp}.txt"
        output_path = self.output_dir / filename

        lines = [
            "=" * 60,
            "SOCCER FILM ANALYSIS - DEBUG LOG",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 60,
            "FRAME DETECTION SUMMARY",
            "-" * 60,
            f"Total frames analyzed: {len(detections_by_frame)}",
        ]

        # Sample frame statistics
        ball_detected_count = 0
        player_counts = []
        referee_counts = []

        for frame_num, det in detections_by_frame.items():
            if hasattr(det, 'ball') and det.ball:
                ball_detected_count += 1
            if hasattr(det, 'players'):
                player_counts.append(len(det.players))
            if hasattr(det, 'referees'):
                referee_counts.append(len(det.referees))

        if player_counts:
            avg_players = sum(player_counts) / len(player_counts)
            lines.append(f"Average players per frame: {avg_players:.1f}")
            lines.append(f"Ball detected in {ball_detected_count}/{len(detections_by_frame)} frames ({100*ball_detected_count/max(1,len(detections_by_frame)):.1f}%)")

        if referee_counts:
            avg_refs = sum(referee_counts) / len(referee_counts)
            lines.append(f"Average referees per frame: {avg_refs:.1f}")

        lines.extend([
            "",
            "-" * 60,
            "EVENTS DETECTED",
            "-" * 60,
            f"Total events: {len(events)}",
        ])

        # Count by type
        event_type_counts = {}
        for event in events:
            etype = event.get("event_type", "unknown")
            event_type_counts[etype] = event_type_counts.get(etype, 0) + 1

        for etype, count in sorted(event_type_counts.items()):
            lines.append(f"  {etype}: {count}")

        if errors:
            lines.extend([
                "",
                "-" * 60,
                "ERRORS",
                "-" * 60,
            ])
            for error in errors:
                lines.append(f"  - {error}")

        lines.extend([
            "",
            "-" * 60,
            "SAMPLE FRAME DETAILS (First 10 frames)",
            "-" * 60,
        ])

        sample_frames = list(detections_by_frame.items())[:10]
        for frame_num, det in sample_frames:
            lines.append(f"\nFrame {frame_num}:")
            if hasattr(det, 'players'):
                lines.append(f"  Players: {len(det.players)}")
                for i, p in enumerate(det.players[:3]):
                    team = f"Team {p.team_id}" if p.team_id is not None else "Unknown"
                    lines.append(f"    Player {i+1}: {team}, conf={p.confidence:.2f}")
            if hasattr(det, 'referees'):
                lines.append(f"  Referees: {len(det.referees)}")
            if hasattr(det, 'ball') and det.ball:
                lines.append(f"  Ball: detected at {det.ball.center}, conf={det.ball.confidence:.2f}")
            else:
                lines.append(f"  Ball: NOT DETECTED")

        lines.extend([
            "",
            "=" * 60,
            "END OF DEBUG LOG",
            "=" * 60,
        ])

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Exported debug log to: {output_path}")
        return output_path


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# Example of what the exports look like:
EXAMPLE_JSON_OUTPUT = """
{
  "export_info": {
    "generated_at": "2026-01-17T13:30:00",
    "version": "1.0",
    "format": "soccer_film_analysis_v1"
  },
  "game": {
    "home_team": {"name": "La Follette", "primary_color": [255, 215, 0]},
    "away_team": {"name": "Madison East", "primary_color": [255, 0, 0]},
    "competition": "Conference Championship",
    "venue": "Breese Stevens Field"
  },
  "events": [
    {"timestamp": "00:00", "event_type": "kickoff", "description": "Game Start"},
    {"timestamp": "12:34", "event_type": "shot", "team": "home", "player": "#10"},
    {"timestamp": "45:00", "event_type": "halftime_start", "description": "Halftime"}
  ],
  "summary": {
    "possession": {"home": 52.3, "away": 47.7},
    "total_shots": 8,
    "total_passes": 342,
    "total_corners": 4
  }
}
"""

EXAMPLE_REPORT_OUTPUT = """
============================================================
SOCCER FILM ANALYSIS - GAME REPORT
============================================================
Generated: 2026-01-17 13:30:00

------------------------------------------------------------
GAME INFORMATION
------------------------------------------------------------
Home Team: La Follette
Away Team: Madison East
Competition: Conference Championship
Venue: Breese Stevens Field

------------------------------------------------------------
POSSESSION
------------------------------------------------------------
La Follette: 52.3%
Madison East: 47.7%

------------------------------------------------------------
EVENTS TIMELINE
------------------------------------------------------------
[00:00] KICKOFF: Game Start
[12:34] SHOT - La Follette: Shot on goal by #10
[23:45] CORNER - Madison East
[45:00] HALFTIME: Halftime begins

============================================================
END OF REPORT
============================================================
"""

EXAMPLE_DEBUG_LOG = """
============================================================
SOCCER FILM ANALYSIS - DEBUG LOG
============================================================
Generated: 2026-01-17 13:30:00

------------------------------------------------------------
FRAME DETECTION SUMMARY
------------------------------------------------------------
Total frames analyzed: 54000
Average players per frame: 18.3
Ball detected in 42150/54000 frames (78.1%)
Average referees per frame: 2.1

------------------------------------------------------------
EVENTS DETECTED
------------------------------------------------------------
Total events: 127
  corner: 4
  free_kick: 8
  goal: 2
  halftime_start: 1
  kickoff: 1
  shot: 12
  throw_in: 23

------------------------------------------------------------
SAMPLE FRAME DETAILS (First 10 frames)
------------------------------------------------------------

Frame 0:
  Players: 22
    Player 1: Team 0, conf=0.87
    Player 2: Team 1, conf=0.92
    Player 3: Team 0, conf=0.84
  Referees: 3
  Ball: detected at (640.5, 320.2), conf=0.65

Frame 30:
  Players: 21
  Referees: 2
  Ball: NOT DETECTED

============================================================
END OF DEBUG LOG
============================================================
"""
