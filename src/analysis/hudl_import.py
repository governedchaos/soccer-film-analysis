"""
Soccer Film Analysis - Hudl CSV Import
Imports event data from Hudl CSV exports for comparison with detected events
"""

import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class HudlEvent:
    """Represents an event from Hudl CSV export"""
    timestamp: str  # Original timestamp string (MM:SS or HH:MM:SS)
    timestamp_seconds: float  # Converted to seconds
    event_type: str
    team: Optional[str] = None
    player: Optional[str] = None
    player_number: Optional[int] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    outcome: Optional[str] = None  # e.g., "Goal", "Saved", "Missed"
    half: int = 1  # 1 or 2

    # Position data if available
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None


@dataclass
class HudlImportResult:
    """Result of importing a Hudl CSV file"""
    success: bool
    events: List[HudlEvent] = field(default_factory=list)
    error_message: str = ""

    # Metadata
    home_team: str = ""
    away_team: str = ""
    game_date: Optional[datetime] = None
    total_events: int = 0

    # Event type counts
    event_counts: Dict[str, int] = field(default_factory=dict)


class HudlImporter:
    """
    Imports and parses Hudl CSV exports.

    Hudl exports can vary in format, so this class handles multiple
    common column naming conventions.
    """

    # Common column name variations
    TIMESTAMP_COLUMNS = ["Time", "Timestamp", "Time Code", "Timecode", "time", "timestamp"]
    EVENT_TYPE_COLUMNS = ["Event", "Event Type", "Action", "Type", "event", "action"]
    TEAM_COLUMNS = ["Team", "team", "Side", "Possession"]
    PLAYER_COLUMNS = ["Player", "Name", "player", "Athlete"]
    PLAYER_NUMBER_COLUMNS = ["Number", "#", "Jersey", "No.", "number"]
    DESCRIPTION_COLUMNS = ["Description", "Notes", "Comment", "Details", "description"]
    OUTCOME_COLUMNS = ["Outcome", "Result", "outcome", "result"]
    TAGS_COLUMNS = ["Tags", "Labels", "tags", "labels"]
    HALF_COLUMNS = ["Half", "Period", "half", "period"]

    # Event type mapping from Hudl to internal types
    EVENT_TYPE_MAP = {
        # Hudl terminology -> internal type
        "goal": "goal",
        "shot": "shot",
        "shot on target": "shot",
        "shot off target": "shot",
        "save": "save",
        "corner kick": "corner",
        "corner": "corner",
        "free kick": "free_kick",
        "free-kick": "free_kick",
        "throw in": "throw_in",
        "throw-in": "throw_in",
        "goal kick": "goal_kick",
        "pass": "pass",
        "pass completed": "pass",
        "pass incomplete": "pass",
        "tackle": "steal",
        "interception": "steal",
        "steal": "steal",
        "foul": "foul",
        "yellow card": "foul",
        "red card": "foul",
        "offside": "offside",
        "substitution": "substitution",
        "sub": "substitution",
        "kickoff": "half_time",
        "kick off": "half_time",
        "half time": "half_time",
        "halftime": "half_time",
    }

    def __init__(self):
        self.events: List[HudlEvent] = []
        self._column_map: Dict[str, str] = {}

    def import_file(self, filepath: str | Path) -> HudlImportResult:
        """
        Import a Hudl CSV file.

        Args:
            filepath: Path to the CSV file

        Returns:
            HudlImportResult with parsed events
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return HudlImportResult(
                success=False,
                error_message=f"File not found: {filepath}"
            )

        if not filepath.suffix.lower() == ".csv":
            return HudlImportResult(
                success=False,
                error_message=f"Expected CSV file, got: {filepath.suffix}"
            )

        try:
            events = []
            event_counts: Dict[str, int] = {}

            with open(filepath, 'r', encoding='utf-8-sig') as f:
                # Try to detect delimiter
                sample = f.read(4096)
                f.seek(0)

                if '\t' in sample:
                    delimiter = '\t'
                else:
                    delimiter = ','

                reader = csv.DictReader(f, delimiter=delimiter)

                # Map columns to standard names
                self._map_columns(reader.fieldnames or [])

                for row in reader:
                    event = self._parse_row(row)
                    if event:
                        events.append(event)

                        # Count event types
                        event_type = event.event_type
                        event_counts[event_type] = event_counts.get(event_type, 0) + 1

            # Sort events by timestamp
            events.sort(key=lambda e: e.timestamp_seconds)

            logger.info(f"Imported {len(events)} events from {filepath.name}")

            return HudlImportResult(
                success=True,
                events=events,
                total_events=len(events),
                event_counts=event_counts
            )

        except Exception as e:
            logger.error(f"Failed to import Hudl CSV: {e}")
            return HudlImportResult(
                success=False,
                error_message=str(e)
            )

    def _map_columns(self, fieldnames: List[str]):
        """Map CSV columns to standard names"""
        self._column_map = {}

        for field in fieldnames:
            field_lower = field.lower().strip()

            # Check each category
            if field in self.TIMESTAMP_COLUMNS or field_lower in [c.lower() for c in self.TIMESTAMP_COLUMNS]:
                self._column_map["timestamp"] = field
            elif field in self.EVENT_TYPE_COLUMNS or field_lower in [c.lower() for c in self.EVENT_TYPE_COLUMNS]:
                self._column_map["event_type"] = field
            elif field in self.TEAM_COLUMNS or field_lower in [c.lower() for c in self.TEAM_COLUMNS]:
                self._column_map["team"] = field
            elif field in self.PLAYER_COLUMNS or field_lower in [c.lower() for c in self.PLAYER_COLUMNS]:
                self._column_map["player"] = field
            elif field in self.PLAYER_NUMBER_COLUMNS or field_lower in [c.lower() for c in self.PLAYER_NUMBER_COLUMNS]:
                self._column_map["player_number"] = field
            elif field in self.DESCRIPTION_COLUMNS or field_lower in [c.lower() for c in self.DESCRIPTION_COLUMNS]:
                self._column_map["description"] = field
            elif field in self.OUTCOME_COLUMNS or field_lower in [c.lower() for c in self.OUTCOME_COLUMNS]:
                self._column_map["outcome"] = field
            elif field in self.TAGS_COLUMNS or field_lower in [c.lower() for c in self.TAGS_COLUMNS]:
                self._column_map["tags"] = field
            elif field in self.HALF_COLUMNS or field_lower in [c.lower() for c in self.HALF_COLUMNS]:
                self._column_map["half"] = field

        logger.debug(f"Column mapping: {self._column_map}")

    def _parse_row(self, row: Dict[str, str]) -> Optional[HudlEvent]:
        """Parse a single CSV row into a HudlEvent"""
        try:
            # Get timestamp
            timestamp_col = self._column_map.get("timestamp", "")
            timestamp_str = row.get(timestamp_col, "").strip()

            if not timestamp_str:
                return None

            timestamp_seconds = self._parse_timestamp(timestamp_str)

            # Get event type
            event_type_col = self._column_map.get("event_type", "")
            event_type_raw = row.get(event_type_col, "").strip().lower()
            event_type = self.EVENT_TYPE_MAP.get(event_type_raw, "custom")

            # Get other fields
            team = row.get(self._column_map.get("team", ""), "").strip() or None
            player = row.get(self._column_map.get("player", ""), "").strip() or None
            description = row.get(self._column_map.get("description", ""), "").strip()
            outcome = row.get(self._column_map.get("outcome", ""), "").strip() or None

            # Player number
            player_number = None
            number_str = row.get(self._column_map.get("player_number", ""), "").strip()
            if number_str.isdigit():
                player_number = int(number_str)

            # Half
            half = 1
            half_str = row.get(self._column_map.get("half", ""), "").strip()
            if half_str == "2" or "second" in half_str.lower():
                half = 2

            # Tags
            tags = []
            tags_str = row.get(self._column_map.get("tags", ""), "").strip()
            if tags_str:
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]

            return HudlEvent(
                timestamp=timestamp_str,
                timestamp_seconds=timestamp_seconds,
                event_type=event_type,
                team=team,
                player=player,
                player_number=player_number,
                description=description,
                tags=tags,
                outcome=outcome,
                half=half
            )

        except Exception as e:
            logger.debug(f"Failed to parse row: {e}")
            return None

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """
        Parse timestamp string to seconds.

        Handles formats:
        - MM:SS
        - HH:MM:SS
        - M:SS
        - SS
        """
        try:
            parts = timestamp_str.replace(".", ":").split(":")

            if len(parts) == 1:
                # Just seconds
                return float(parts[0])
            elif len(parts) == 2:
                # MM:SS
                minutes, seconds = parts
                return float(minutes) * 60 + float(seconds)
            elif len(parts) >= 3:
                # HH:MM:SS
                hours, minutes, seconds = parts[:3]
                return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            else:
                return 0.0

        except (ValueError, IndexError):
            return 0.0


def compare_events(
    detected_events: List[Dict],
    hudl_events: List[HudlEvent],
    time_tolerance_seconds: float = 5.0
) -> Dict:
    """
    Compare detected events with Hudl events.

    Args:
        detected_events: List of detected events (with timestamp_seconds)
        hudl_events: List of HudlEvent from Hudl import
        time_tolerance_seconds: Time window for matching events

    Returns:
        Dict with comparison statistics
    """
    matched_events = []
    unmatched_detected = []
    unmatched_hudl = []

    hudl_matched = set()

    for det_event in detected_events:
        det_time = det_event.get("timestamp_seconds", 0)
        det_type = det_event.get("event_type", "")

        # Find matching Hudl event
        best_match = None
        best_time_diff = float("inf")

        for i, hudl_event in enumerate(hudl_events):
            if i in hudl_matched:
                continue

            # Check if event types match (or are compatible)
            if hudl_event.event_type != det_type:
                continue

            time_diff = abs(hudl_event.timestamp_seconds - det_time)
            if time_diff <= time_tolerance_seconds and time_diff < best_time_diff:
                best_match = (i, hudl_event, time_diff)
                best_time_diff = time_diff

        if best_match:
            idx, hudl_event, time_diff = best_match
            hudl_matched.add(idx)
            matched_events.append({
                "detected": det_event,
                "hudl": hudl_event,
                "time_difference": time_diff
            })
        else:
            unmatched_detected.append(det_event)

    # Find unmatched Hudl events
    for i, hudl_event in enumerate(hudl_events):
        if i not in hudl_matched:
            unmatched_hudl.append(hudl_event)

    # Calculate statistics
    total_detected = len(detected_events)
    total_hudl = len(hudl_events)
    total_matched = len(matched_events)

    precision = total_matched / total_detected if total_detected > 0 else 0
    recall = total_matched / total_hudl if total_hudl > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "matched_events": matched_events,
        "unmatched_detected": unmatched_detected,
        "unmatched_hudl": unmatched_hudl,
        "statistics": {
            "total_detected": total_detected,
            "total_hudl": total_hudl,
            "total_matched": total_matched,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "avg_time_difference": sum(m["time_difference"] for m in matched_events) / len(matched_events) if matched_events else 0
        }
    }
