"""
Formation Auto-Detection
ML-based system to identify and track team formations throughout a match
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
from collections import defaultdict
from loguru import logger


class FormationType(Enum):
    """Standard soccer formations."""
    F_4_4_2 = "4-4-2"
    F_4_3_3 = "4-3-3"
    F_4_2_3_1 = "4-2-3-1"
    F_4_1_4_1 = "4-1-4-1"
    F_3_5_2 = "3-5-2"
    F_3_4_3 = "3-4-3"
    F_5_3_2 = "5-3-2"
    F_5_4_1 = "5-4-1"
    F_4_5_1 = "4-5-1"
    F_4_4_1_1 = "4-4-1-1"
    F_4_1_2_1_2 = "4-1-2-1-2"  # Diamond
    F_3_4_1_2 = "3-4-1-2"
    UNKNOWN = "Unknown"


@dataclass
class FormationTemplate:
    """Template defining expected player positions for a formation."""
    formation_type: FormationType
    # Positions as (x, y) where x=0 is goal line, x=1 is opponent goal
    # y=0 is left touchline, y=1 is right touchline
    positions: List[Tuple[float, float]]
    lines: List[int]  # Number of players in each line (defense to attack)

    def __post_init__(self):
        # Exclude goalkeeper from template (always at ~0.05, 0.5)
        assert len(self.positions) == 10, "Template must have 10 outfield positions"


# Pre-defined formation templates (normalized coordinates, excluding GK)
FORMATION_TEMPLATES = {
    FormationType.F_4_4_2: FormationTemplate(
        formation_type=FormationType.F_4_4_2,
        positions=[
            # Defense line
            (0.2, 0.15), (0.2, 0.38), (0.2, 0.62), (0.2, 0.85),
            # Midfield line
            (0.45, 0.15), (0.45, 0.38), (0.45, 0.62), (0.45, 0.85),
            # Attack line
            (0.7, 0.35), (0.7, 0.65)
        ],
        lines=[4, 4, 2]
    ),
    FormationType.F_4_3_3: FormationTemplate(
        formation_type=FormationType.F_4_3_3,
        positions=[
            # Defense
            (0.2, 0.15), (0.2, 0.38), (0.2, 0.62), (0.2, 0.85),
            # Midfield
            (0.45, 0.25), (0.45, 0.5), (0.45, 0.75),
            # Attack
            (0.75, 0.15), (0.75, 0.5), (0.75, 0.85)
        ],
        lines=[4, 3, 3]
    ),
    FormationType.F_4_2_3_1: FormationTemplate(
        formation_type=FormationType.F_4_2_3_1,
        positions=[
            # Defense
            (0.2, 0.15), (0.2, 0.38), (0.2, 0.62), (0.2, 0.85),
            # Defensive mid
            (0.35, 0.35), (0.35, 0.65),
            # Attacking mid
            (0.55, 0.15), (0.55, 0.5), (0.55, 0.85),
            # Striker
            (0.75, 0.5)
        ],
        lines=[4, 2, 3, 1]
    ),
    FormationType.F_3_5_2: FormationTemplate(
        formation_type=FormationType.F_3_5_2,
        positions=[
            # Defense (3)
            (0.2, 0.25), (0.2, 0.5), (0.2, 0.75),
            # Midfield (5) - wingbacks + 3 central
            (0.4, 0.1), (0.35, 0.35), (0.35, 0.5), (0.35, 0.65), (0.4, 0.9),
            # Attack (2)
            (0.7, 0.35), (0.7, 0.65)
        ],
        lines=[3, 5, 2]
    ),
    FormationType.F_3_4_3: FormationTemplate(
        formation_type=FormationType.F_3_4_3,
        positions=[
            # Defense (3)
            (0.2, 0.25), (0.2, 0.5), (0.2, 0.75),
            # Midfield (4)
            (0.4, 0.15), (0.4, 0.4), (0.4, 0.6), (0.4, 0.85),
            # Attack (3)
            (0.7, 0.2), (0.7, 0.5), (0.7, 0.8)
        ],
        lines=[3, 4, 3]
    ),
    FormationType.F_5_3_2: FormationTemplate(
        formation_type=FormationType.F_5_3_2,
        positions=[
            # Defense (5)
            (0.18, 0.1), (0.2, 0.3), (0.2, 0.5), (0.2, 0.7), (0.18, 0.9),
            # Midfield (3)
            (0.4, 0.3), (0.4, 0.5), (0.4, 0.7),
            # Attack (2)
            (0.65, 0.35), (0.65, 0.65)
        ],
        lines=[5, 3, 2]
    ),
    FormationType.F_4_5_1: FormationTemplate(
        formation_type=FormationType.F_4_5_1,
        positions=[
            # Defense (4)
            (0.2, 0.15), (0.2, 0.38), (0.2, 0.62), (0.2, 0.85),
            # Midfield (5)
            (0.4, 0.1), (0.4, 0.3), (0.4, 0.5), (0.4, 0.7), (0.4, 0.9),
            # Attack (1)
            (0.7, 0.5)
        ],
        lines=[4, 5, 1]
    ),
    FormationType.F_4_1_4_1: FormationTemplate(
        formation_type=FormationType.F_4_1_4_1,
        positions=[
            # Defense (4)
            (0.2, 0.15), (0.2, 0.38), (0.2, 0.62), (0.2, 0.85),
            # Defensive mid (1)
            (0.32, 0.5),
            # Midfield (4)
            (0.5, 0.15), (0.5, 0.38), (0.5, 0.62), (0.5, 0.85),
            # Attack (1)
            (0.72, 0.5)
        ],
        lines=[4, 1, 4, 1]
    ),
    FormationType.F_4_4_1_1: FormationTemplate(
        formation_type=FormationType.F_4_4_1_1,
        positions=[
            # Defense (4)
            (0.2, 0.15), (0.2, 0.38), (0.2, 0.62), (0.2, 0.85),
            # Midfield (4)
            (0.4, 0.15), (0.4, 0.38), (0.4, 0.62), (0.4, 0.85),
            # Attacking mid (1)
            (0.58, 0.5),
            # Striker (1)
            (0.75, 0.5)
        ],
        lines=[4, 4, 1, 1]
    ),
    FormationType.F_5_4_1: FormationTemplate(
        formation_type=FormationType.F_5_4_1,
        positions=[
            # Defense (5)
            (0.18, 0.1), (0.2, 0.28), (0.2, 0.5), (0.2, 0.72), (0.18, 0.9),
            # Midfield (4)
            (0.42, 0.2), (0.42, 0.4), (0.42, 0.6), (0.42, 0.8),
            # Attack (1)
            (0.68, 0.5)
        ],
        lines=[5, 4, 1]
    ),
}


@dataclass
class FormationMatch:
    """Result of formation matching."""
    formation_type: FormationType
    confidence: float
    match_score: float
    player_assignments: Dict[int, int]  # player_id -> template_position_index
    avg_distance: float  # Average distance from template positions


@dataclass
class FormationSnapshot:
    """Formation detected at a specific moment."""
    frame: int
    timestamp: float
    team_id: int
    formation: FormationType
    confidence: float
    player_positions: Dict[int, Tuple[float, float]]  # player_id -> (x, y) normalized


@dataclass
class FormationChange:
    """Detected formation change event."""
    frame: int
    timestamp: float
    team_id: int
    from_formation: FormationType
    to_formation: FormationType
    confidence: float


@dataclass
class FormationAnalysisResult:
    """Complete formation analysis for a match."""
    team_id: int
    primary_formation: FormationType
    primary_confidence: float
    formation_distribution: Dict[FormationType, float]  # formation -> percentage of time
    formation_changes: List[FormationChange]
    snapshots: List[FormationSnapshot]


class FormationDetector:
    """
    Detects and tracks team formations using player position data.

    Uses template matching with Hungarian algorithm for optimal
    player-to-position assignment.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        min_players_required: int = 8,
        smoothing_window: int = 30,  # frames
        change_threshold: float = 0.3,  # confidence diff to trigger change
        min_change_duration: int = 60  # frames before confirming change
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.min_players_required = min_players_required
        self.smoothing_window = smoothing_window
        self.change_threshold = change_threshold
        self.min_change_duration = min_change_duration

        # Formation history for smoothing
        self.formation_history: Dict[int, List[Tuple[int, FormationType, float]]] = defaultdict(list)

        # Current detected formation per team
        self.current_formations: Dict[int, FormationType] = {}

    def _normalize_positions(
        self,
        positions: Dict[int, Tuple[float, float]],
        attacking_direction: int = 1  # 1 = left to right, -1 = right to left
    ) -> Dict[int, Tuple[float, float]]:
        """Normalize positions to 0-1 range, always attacking left to right."""
        normalized = {}

        for player_id, (x, y) in positions.items():
            # Normalize to 0-1
            norm_x = x / self.pitch_length
            norm_y = y / self.pitch_width

            # Flip if attacking right to left
            if attacking_direction == -1:
                norm_x = 1 - norm_x
                norm_y = 1 - norm_y

            normalized[player_id] = (norm_x, norm_y)

        return normalized

    def _identify_goalkeeper(
        self,
        positions: Dict[int, Tuple[float, float]]
    ) -> Optional[int]:
        """Identify goalkeeper as the player closest to own goal."""
        if not positions:
            return None

        # Goalkeeper is typically closest to x=0 (own goal)
        min_x = float('inf')
        gk_id = None

        for player_id, (x, y) in positions.items():
            if x < min_x:
                min_x = x
                gk_id = player_id

        # Verify goalkeeper is actually near goal (x < 0.15)
        if gk_id and positions[gk_id][0] < 0.15:
            return gk_id

        return None

    def _calculate_template_distance(
        self,
        player_positions: List[Tuple[float, float]],
        template: FormationTemplate
    ) -> Tuple[float, Dict[int, int]]:
        """
        Calculate minimum total distance between players and template positions.
        Uses Hungarian algorithm for optimal assignment.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            # Fallback to greedy assignment
            return self._greedy_assignment(player_positions, template)

        n_players = len(player_positions)
        n_template = len(template.positions)

        # Build cost matrix
        cost_matrix = np.zeros((n_players, n_template))

        for i, (px, py) in enumerate(player_positions):
            for j, (tx, ty) in enumerate(template.positions):
                # Euclidean distance
                cost_matrix[i, j] = np.sqrt((px - tx) ** 2 + (py - ty) ** 2)

        # Handle case where we have fewer players than template positions
        if n_players < n_template:
            # Pad cost matrix with high costs for missing players
            padded = np.full((n_template, n_template), 1000.0)
            padded[:n_players, :] = cost_matrix
            cost_matrix = padded

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Calculate total distance and build assignment
        total_distance = 0
        assignments = {}

        for i, j in zip(row_ind, col_ind):
            if i < n_players:  # Skip padded rows
                total_distance += cost_matrix[i, j]
                assignments[i] = j

        avg_distance = total_distance / max(n_players, 1)
        return avg_distance, assignments

    def _greedy_assignment(
        self,
        player_positions: List[Tuple[float, float]],
        template: FormationTemplate
    ) -> Tuple[float, Dict[int, int]]:
        """Fallback greedy assignment when scipy not available."""
        assignments = {}
        used_template_positions = set()
        total_distance = 0

        for i, (px, py) in enumerate(player_positions):
            min_dist = float('inf')
            best_j = -1

            for j, (tx, ty) in enumerate(template.positions):
                if j in used_template_positions:
                    continue

                dist = np.sqrt((px - tx) ** 2 + (py - ty) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_j = j

            if best_j >= 0:
                assignments[i] = best_j
                used_template_positions.add(best_j)
                total_distance += min_dist

        avg_distance = total_distance / max(len(player_positions), 1)
        return avg_distance, assignments

    def detect_formation(
        self,
        frame: int,
        team_positions: Dict[int, Tuple[float, float]],
        team_id: int,
        attacking_direction: int = 1
    ) -> Optional[FormationSnapshot]:
        """
        Detect formation from player positions.

        Args:
            frame: Current frame number
            team_positions: Dict of player_id -> (x, y) in meters
            team_id: Team identifier
            attacking_direction: 1 for left-to-right, -1 for right-to-left

        Returns:
            FormationSnapshot or None if detection failed
        """
        if len(team_positions) < self.min_players_required:
            return None

        # Normalize positions
        normalized = self._normalize_positions(team_positions, attacking_direction)

        # Identify and remove goalkeeper
        gk_id = self._identify_goalkeeper(normalized)
        outfield_positions = {
            pid: pos for pid, pos in normalized.items()
            if pid != gk_id
        }

        if len(outfield_positions) < self.min_players_required - 1:
            return None

        # Convert to list for matching
        player_ids = list(outfield_positions.keys())
        positions_list = [outfield_positions[pid] for pid in player_ids]

        # Match against all templates
        best_match: Optional[FormationMatch] = None

        for formation_type, template in FORMATION_TEMPLATES.items():
            avg_distance, assignments = self._calculate_template_distance(
                positions_list, template
            )

            # Convert distance to confidence score (inverse relationship)
            # Lower distance = higher confidence
            confidence = max(0, 1 - (avg_distance * 2))

            if best_match is None or confidence > best_match.confidence:
                # Remap assignments to player IDs
                player_assignments = {
                    player_ids[i]: j for i, j in assignments.items()
                }

                best_match = FormationMatch(
                    formation_type=formation_type,
                    confidence=confidence,
                    match_score=1 - avg_distance,
                    player_assignments=player_assignments,
                    avg_distance=avg_distance
                )

        if best_match is None or best_match.confidence < 0.3:
            formation_type = FormationType.UNKNOWN
            confidence = 0.0
            logger.debug(f"Frame {frame}: Formation unknown (confidence too low)")
        else:
            formation_type = best_match.formation_type
            confidence = best_match.confidence
            logger.debug(
                f"Frame {frame}: Detected {formation_type.value} "
                f"(confidence={confidence:.2f}, distance={best_match.avg_distance:.3f})"
            )

        # Store in history for smoothing
        timestamp = frame / 30.0  # Assume 30fps
        self.formation_history[team_id].append((frame, formation_type, confidence))

        # Keep only recent history
        cutoff_frame = frame - self.smoothing_window * 2
        self.formation_history[team_id] = [
            (f, ft, c) for f, ft, c in self.formation_history[team_id]
            if f > cutoff_frame
        ]

        return FormationSnapshot(
            frame=frame,
            timestamp=timestamp,
            team_id=team_id,
            formation=formation_type,
            confidence=confidence,
            player_positions=normalized
        )

    def get_smoothed_formation(
        self,
        team_id: int,
        current_frame: int
    ) -> Tuple[FormationType, float]:
        """
        Get smoothed formation using recent history.
        Helps avoid noisy frame-by-frame changes.
        """
        history = self.formation_history.get(team_id, [])

        if not history:
            return FormationType.UNKNOWN, 0.0

        # Get recent entries within smoothing window
        recent = [
            (ft, c) for f, ft, c in history
            if current_frame - self.smoothing_window <= f <= current_frame
        ]

        if not recent:
            return FormationType.UNKNOWN, 0.0

        # Count weighted votes for each formation
        formation_scores: Dict[FormationType, float] = defaultdict(float)

        for formation_type, confidence in recent:
            formation_scores[formation_type] += confidence

        # Get highest scoring formation
        best_formation = max(formation_scores.items(), key=lambda x: x[1])
        avg_confidence = best_formation[1] / len(recent)

        return best_formation[0], avg_confidence

    def detect_formation_change(
        self,
        team_id: int,
        new_formation: FormationType,
        new_confidence: float,
        frame: int,
        timestamp: float
    ) -> Optional[FormationChange]:
        """
        Detect if a formation change has occurred.

        Returns FormationChange if change is confirmed, None otherwise.
        """
        current = self.current_formations.get(team_id)

        if current is None:
            self.current_formations[team_id] = new_formation
            return None

        if new_formation == current:
            return None

        if new_formation == FormationType.UNKNOWN:
            return None

        # Check if change is sustained
        history = self.formation_history.get(team_id, [])
        recent_frames = [
            (f, ft) for f, ft, c in history
            if frame - self.min_change_duration <= f <= frame
            and c > 0.4
        ]

        if len(recent_frames) < self.min_change_duration // 2:
            return None

        # Count formations in recent window
        formation_counts: Dict[FormationType, int] = defaultdict(int)
        for _, ft in recent_frames:
            formation_counts[ft] += 1

        # Check if new formation is dominant
        total = len(recent_frames)
        if formation_counts[new_formation] / total > 0.6:
            old_formation = current
            self.current_formations[team_id] = new_formation

            return FormationChange(
                frame=frame,
                timestamp=timestamp,
                team_id=team_id,
                from_formation=old_formation,
                to_formation=new_formation,
                confidence=new_confidence
            )

        return None


class FormationAnalyzer:
    """
    High-level formation analysis for complete matches.
    """

    def __init__(self, detector: Optional[FormationDetector] = None):
        self.detector = detector or FormationDetector()
        self.snapshots: Dict[int, List[FormationSnapshot]] = defaultdict(list)
        self.changes: Dict[int, List[FormationChange]] = defaultdict(list)

    def process_frame(
        self,
        frame: int,
        team_positions: Dict[int, Dict[int, Tuple[float, float]]],
        attacking_directions: Dict[int, int]
    ) -> Dict[int, FormationSnapshot]:
        """
        Process a single frame for all teams.

        Args:
            frame: Frame number
            team_positions: {team_id: {player_id: (x, y)}}
            attacking_directions: {team_id: direction}

        Returns:
            Dict of team_id -> FormationSnapshot
        """
        results = {}

        for team_id, positions in team_positions.items():
            direction = attacking_directions.get(team_id, 1)

            snapshot = self.detector.detect_formation(
                frame=frame,
                team_positions=positions,
                team_id=team_id,
                attacking_direction=direction
            )

            if snapshot:
                self.snapshots[team_id].append(snapshot)

                # Check for formation change
                change = self.detector.detect_formation_change(
                    team_id=team_id,
                    new_formation=snapshot.formation,
                    new_confidence=snapshot.confidence,
                    frame=frame,
                    timestamp=snapshot.timestamp
                )

                if change:
                    self.changes[team_id].append(change)
                    logger.info(
                        f"Formation change detected: Team {team_id} "
                        f"{change.from_formation.value} -> {change.to_formation.value}"
                    )

                results[team_id] = snapshot

        return results

    def get_analysis(self, team_id: int) -> FormationAnalysisResult:
        """Get complete formation analysis for a team."""
        snapshots = self.snapshots.get(team_id, [])
        changes = self.changes.get(team_id, [])

        if not snapshots:
            return FormationAnalysisResult(
                team_id=team_id,
                primary_formation=FormationType.UNKNOWN,
                primary_confidence=0.0,
                formation_distribution={},
                formation_changes=[],
                snapshots=[]
            )

        # Calculate formation distribution
        formation_counts: Dict[FormationType, int] = defaultdict(int)
        total_confidence: Dict[FormationType, float] = defaultdict(float)

        for snapshot in snapshots:
            formation_counts[snapshot.formation] += 1
            total_confidence[snapshot.formation] += snapshot.confidence

        total_snapshots = len(snapshots)
        distribution = {
            ft: count / total_snapshots
            for ft, count in formation_counts.items()
        }

        # Find primary formation (highest weighted score)
        weighted_scores = {
            ft: (formation_counts[ft] / total_snapshots) * (total_confidence[ft] / formation_counts[ft])
            for ft in formation_counts
        }

        primary = max(weighted_scores.items(), key=lambda x: x[1])
        primary_formation = primary[0]
        primary_confidence = total_confidence[primary_formation] / formation_counts[primary_formation]

        return FormationAnalysisResult(
            team_id=team_id,
            primary_formation=primary_formation,
            primary_confidence=primary_confidence,
            formation_distribution=distribution,
            formation_changes=changes,
            snapshots=snapshots
        )

    def get_formation_at_time(
        self,
        team_id: int,
        timestamp: float
    ) -> Optional[FormationSnapshot]:
        """Get formation snapshot closest to given timestamp."""
        snapshots = self.snapshots.get(team_id, [])

        if not snapshots:
            return None

        # Find closest snapshot
        closest = min(snapshots, key=lambda s: abs(s.timestamp - timestamp))
        return closest

    def get_formation_timeline(
        self,
        team_id: int,
        interval_seconds: float = 60.0
    ) -> List[Tuple[float, FormationType, float]]:
        """
        Get formation timeline at regular intervals.

        Returns list of (timestamp, formation, confidence)
        """
        snapshots = self.snapshots.get(team_id, [])

        if not snapshots:
            return []

        # Group by interval
        timeline = []
        current_interval = 0

        while True:
            interval_start = current_interval * interval_seconds
            interval_end = (current_interval + 1) * interval_seconds

            # Get snapshots in this interval
            interval_snapshots = [
                s for s in snapshots
                if interval_start <= s.timestamp < interval_end
            ]

            if not interval_snapshots:
                if interval_start > max(s.timestamp for s in snapshots):
                    break
                current_interval += 1
                continue

            # Get most common formation in interval
            formation_counts: Dict[FormationType, int] = defaultdict(int)
            confidence_sum: Dict[FormationType, float] = defaultdict(float)

            for s in interval_snapshots:
                formation_counts[s.formation] += 1
                confidence_sum[s.formation] += s.confidence

            best = max(formation_counts.items(), key=lambda x: x[1])
            avg_confidence = confidence_sum[best[0]] / best[1]

            timeline.append((interval_start, best[0], avg_confidence))
            current_interval += 1

        return timeline

    def reset(self):
        """Reset all stored data."""
        self.snapshots.clear()
        self.changes.clear()
        self.detector.formation_history.clear()
        self.detector.current_formations.clear()
