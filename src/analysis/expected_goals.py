"""
Expected Goals (xG) Model
Shot quality model incorporating distance, angle, body part, pressure, and keeper position
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import math
from loguru import logger


class ShotType(Enum):
    """Type of shot taken."""
    RIGHT_FOOT = "right_foot"
    LEFT_FOOT = "left_foot"
    HEADER = "header"
    OTHER = "other"  # Chest, shoulder, etc.


class ShotSituation(Enum):
    """Situation leading to the shot."""
    OPEN_PLAY = "open_play"
    COUNTER_ATTACK = "counter_attack"
    SET_PIECE_DIRECT = "set_piece_direct"  # Direct free kick
    SET_PIECE_INDIRECT = "set_piece_indirect"  # From corner/free kick
    PENALTY = "penalty"
    REBOUND = "rebound"
    ONE_ON_ONE = "one_on_one"


class ShotOutcome(Enum):
    """Outcome of the shot."""
    GOAL = "goal"
    SAVED = "saved"
    BLOCKED = "blocked"
    OFF_TARGET = "off_target"
    POST = "post"


@dataclass
class ShotContext:
    """Context information for a shot."""
    # Position (in meters from goal line center)
    x: float  # Distance from goal line
    y: float  # Lateral position (0 = center)

    # Shot characteristics
    shot_type: ShotType = ShotType.RIGHT_FOOT
    situation: ShotSituation = ShotSituation.OPEN_PLAY

    # Pressure metrics
    defender_count: int = 0  # Defenders between shooter and goal
    nearest_defender_distance: float = 10.0  # meters
    is_under_pressure: bool = False

    # Goalkeeper position
    goalkeeper_x: Optional[float] = None
    goalkeeper_y: Optional[float] = None
    goalkeeper_off_line: bool = False

    # Game state
    is_home: bool = True
    score_differential: int = 0  # positive = winning
    minute: int = 45

    # Additional context
    previous_action: Optional[str] = None  # "cross", "through_ball", "dribble"
    touch_count: int = 1  # Number of touches before shot
    is_first_time: bool = False  # First-time shot


@dataclass
class ShotRecord:
    """Complete record of a shot."""
    frame: int
    timestamp: float
    player_id: int
    team_id: int
    context: ShotContext
    xg_value: float
    outcome: Optional[ShotOutcome] = None


@dataclass
class xGAnalysisResult:
    """xG analysis for a match or period."""
    team_id: int
    total_xg: float
    shot_count: int
    goals: int
    shots: List[ShotRecord]
    xg_per_shot: float
    conversion_rate: float
    over_performance: float  # Goals - xG


class ExpectedGoalsModel:
    """
    Expected Goals model based on shot location and context.

    Uses a logistic regression-style approach with pre-calibrated
    coefficients based on historical data patterns.
    """

    # Goal dimensions (standard)
    GOAL_WIDTH = 7.32  # meters
    GOAL_HEIGHT = 2.44  # meters

    # Base xG coefficients (calibrated from historical patterns)
    # Adjusted to produce realistic xG values:
    # - 6m central: ~0.35-0.40
    # - 11m (penalty spot): ~0.15-0.20
    # - 20m central: ~0.05-0.08
    COEFFICIENTS = {
        'intercept': -0.8,
        'distance': -0.09,
        'angle': 1.2,
        'angle_squared': -0.4,
        'header': -0.5,
        'weak_foot': -0.2,
        'under_pressure': -0.3,
        'defender_blocking': -0.15,
        'counter_attack': 0.2,
        'one_on_one': 0.8,
        'rebound': 0.3,
        'first_time': -0.1,
        'gk_off_line': 0.4,
        'big_chance': 0.6,
    }

    # Shot type modifiers
    SHOT_TYPE_MODIFIERS = {
        ShotType.RIGHT_FOOT: 1.0,
        ShotType.LEFT_FOOT: 0.95,
        ShotType.HEADER: 0.7,
        ShotType.OTHER: 0.5,
    }

    # Situation modifiers
    SITUATION_MODIFIERS = {
        ShotSituation.OPEN_PLAY: 1.0,
        ShotSituation.COUNTER_ATTACK: 1.15,
        ShotSituation.SET_PIECE_DIRECT: 0.08,  # Free kicks are hard
        ShotSituation.SET_PIECE_INDIRECT: 0.9,
        ShotSituation.PENALTY: 0.76,  # Historical conversion rate
        ShotSituation.REBOUND: 1.2,
        ShotSituation.ONE_ON_ONE: 1.4,
    }

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.goal_center_y = pitch_width / 2
        logger.debug(f"xG model initialized: pitch={pitch_length}x{pitch_width}m")

    def calculate_angle(self, x: float, y: float) -> float:
        """
        Calculate the angle to goal in radians.

        The angle is the visual angle subtended by the goal from the shot position.
        """
        # Goal posts positions
        left_post_y = self.goal_center_y - self.GOAL_WIDTH / 2
        right_post_y = self.goal_center_y + self.GOAL_WIDTH / 2

        # Angles to each post
        angle_left = math.atan2(left_post_y - y, x)
        angle_right = math.atan2(right_post_y - y, x)

        # Visible angle
        angle = abs(angle_left - angle_right)
        return angle

    def calculate_distance(self, x: float, y: float) -> float:
        """Calculate distance to goal center."""
        dy = y - self.goal_center_y
        return math.sqrt(x ** 2 + dy ** 2)

    def calculate_base_xg(self, x: float, y: float) -> float:
        """
        Calculate base xG from position only.

        Uses a simplified model based on distance and angle.
        """
        distance = self.calculate_distance(x, y)
        angle = self.calculate_angle(x, y)

        # Logit calculation
        logit = (
            self.COEFFICIENTS['intercept'] +
            self.COEFFICIENTS['distance'] * distance +
            self.COEFFICIENTS['angle'] * angle +
            self.COEFFICIENTS['angle_squared'] * (angle ** 2)
        )

        # Convert to probability
        xg = 1 / (1 + math.exp(-logit))
        return xg

    def calculate_xg(self, context: ShotContext) -> float:
        """
        Calculate full xG incorporating all context factors.

        Args:
            context: Shot context with all relevant information

        Returns:
            Expected goals value (0.0 to 1.0)
        """
        # Handle penalty
        if context.situation == ShotSituation.PENALTY:
            return self.SITUATION_MODIFIERS[ShotSituation.PENALTY]

        # Calculate base xG
        base_xg = self.calculate_base_xg(context.x, context.y)

        # Apply shot type modifier
        shot_modifier = self.SHOT_TYPE_MODIFIERS.get(context.shot_type, 1.0)

        # Apply situation modifier
        situation_modifier = self.SITUATION_MODIFIERS.get(context.situation, 1.0)

        # Calculate pressure adjustment
        pressure_adjustment = 1.0
        if context.is_under_pressure:
            pressure_adjustment *= 0.85

        if context.defender_count > 0:
            # Each defender between shooter and goal reduces xG
            blocking_factor = 0.9 ** context.defender_count
            pressure_adjustment *= blocking_factor

        if context.nearest_defender_distance < 1.5:
            pressure_adjustment *= 0.8

        # Goalkeeper positioning
        gk_adjustment = 1.0
        if context.goalkeeper_x is not None and context.goalkeeper_y is not None:
            # Check if GK is off their line
            if context.goalkeeper_off_line:
                gk_adjustment = 1.2  # Easier to score
            else:
                # Check if GK is well-positioned
                gk_distance_from_center = abs(context.goalkeeper_y - self.goal_center_y)
                if gk_distance_from_center < 1.0:  # Well centered
                    gk_adjustment = 0.95

        # Big chance detection
        is_big_chance = (
            context.x < 11 and  # Inside box
            abs(context.y - self.goal_center_y) < 8 and  # Central
            context.defender_count == 0 and
            not context.is_under_pressure
        )

        big_chance_bonus = 1.3 if is_big_chance else 1.0

        # First-time shot adjustment
        first_time_adjustment = 0.92 if context.is_first_time else 1.0

        # Combine all factors
        final_xg = (
            base_xg *
            shot_modifier *
            situation_modifier *
            pressure_adjustment *
            gk_adjustment *
            big_chance_bonus *
            first_time_adjustment
        )

        # Clamp to valid range
        final_xg = min(max(final_xg, 0.01), 0.99)

        logger.debug(
            f"xG calculated: pos=({context.x:.1f}, {context.y:.1f}), "
            f"base={base_xg:.3f}, final={final_xg:.3f}, "
            f"type={context.shot_type.value}, situation={context.situation.value}"
        )

        return final_xg

    def get_xg_breakdown(self, context: ShotContext) -> Dict[str, float]:
        """Get detailed breakdown of xG calculation."""
        base_xg = self.calculate_base_xg(context.x, context.y)

        breakdown = {
            'base_xg': base_xg,
            'distance': self.calculate_distance(context.x, context.y),
            'angle': math.degrees(self.calculate_angle(context.x, context.y)),
            'shot_type_modifier': self.SHOT_TYPE_MODIFIERS.get(context.shot_type, 1.0),
            'situation_modifier': self.SITUATION_MODIFIERS.get(context.situation, 1.0),
            'pressure_penalty': 0.85 if context.is_under_pressure else 1.0,
            'defender_blocking': 0.9 ** context.defender_count,
            'final_xg': self.calculate_xg(context)
        }

        return breakdown


class xGTracker:
    """
    Tracks and analyzes expected goals throughout a match.
    """

    def __init__(self, model: Optional[ExpectedGoalsModel] = None):
        self.model = model or ExpectedGoalsModel()
        self.shots: Dict[int, List[ShotRecord]] = {}  # team_id -> shots

    def record_shot(
        self,
        frame: int,
        timestamp: float,
        player_id: int,
        team_id: int,
        context: ShotContext,
        outcome: Optional[ShotOutcome] = None
    ) -> ShotRecord:
        """
        Record a shot and calculate its xG.

        Returns:
            ShotRecord with calculated xG
        """
        xg_value = self.model.calculate_xg(context)

        record = ShotRecord(
            frame=frame,
            timestamp=timestamp,
            player_id=player_id,
            team_id=team_id,
            context=context,
            xg_value=xg_value,
            outcome=outcome
        )

        if team_id not in self.shots:
            self.shots[team_id] = []

        self.shots[team_id].append(record)

        logger.info(
            f"Shot recorded: Player {player_id}, xG={xg_value:.3f}, "
            f"outcome={outcome.value if outcome else 'unknown'}"
        )

        return record

    def update_outcome(
        self,
        team_id: int,
        shot_index: int,
        outcome: ShotOutcome
    ):
        """Update the outcome of a previously recorded shot."""
        if team_id in self.shots and shot_index < len(self.shots[team_id]):
            self.shots[team_id][shot_index].outcome = outcome

    def get_team_xg(self, team_id: int) -> float:
        """Get total xG for a team."""
        if team_id not in self.shots:
            return 0.0
        return sum(shot.xg_value for shot in self.shots[team_id])

    def get_team_goals(self, team_id: int) -> int:
        """Get actual goals scored by a team."""
        if team_id not in self.shots:
            return 0
        return sum(
            1 for shot in self.shots[team_id]
            if shot.outcome == ShotOutcome.GOAL
        )

    def get_player_xg(self, player_id: int) -> float:
        """Get total xG for a player across all teams."""
        total = 0.0
        for shots in self.shots.values():
            total += sum(
                shot.xg_value for shot in shots
                if shot.player_id == player_id
            )
        return total

    def get_xg_timeline(
        self,
        team_id: int,
        interval_seconds: float = 60.0
    ) -> List[Tuple[float, float]]:
        """
        Get cumulative xG over time.

        Returns list of (timestamp, cumulative_xg)
        """
        if team_id not in self.shots:
            return []

        sorted_shots = sorted(self.shots[team_id], key=lambda s: s.timestamp)

        timeline = []
        cumulative = 0.0

        for shot in sorted_shots:
            cumulative += shot.xg_value
            timeline.append((shot.timestamp, cumulative))

        return timeline

    def get_analysis(self, team_id: int) -> xGAnalysisResult:
        """Get comprehensive xG analysis for a team."""
        shots = self.shots.get(team_id, [])

        total_xg = sum(s.xg_value for s in shots)
        goals = sum(1 for s in shots if s.outcome == ShotOutcome.GOAL)
        shot_count = len(shots)

        return xGAnalysisResult(
            team_id=team_id,
            total_xg=total_xg,
            shot_count=shot_count,
            goals=goals,
            shots=shots,
            xg_per_shot=total_xg / shot_count if shot_count > 0 else 0.0,
            conversion_rate=goals / shot_count if shot_count > 0 else 0.0,
            over_performance=goals - total_xg
        )

    def get_shot_map_data(
        self,
        team_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get shot data formatted for visualization.

        Returns list of dicts with position, xG, outcome for each shot.
        """
        shots = self.shots.get(team_id, [])

        return [
            {
                'x': shot.context.x,
                'y': shot.context.y,
                'xg': shot.xg_value,
                'outcome': shot.outcome.value if shot.outcome else None,
                'player_id': shot.player_id,
                'timestamp': shot.timestamp,
                'shot_type': shot.context.shot_type.value,
                'situation': shot.context.situation.value
            }
            for shot in shots
        ]

    def compare_teams(self) -> Dict[str, Any]:
        """Compare xG metrics between all tracked teams."""
        comparison = {}

        for team_id in self.shots:
            analysis = self.get_analysis(team_id)
            comparison[f'team_{team_id}'] = {
                'total_xg': analysis.total_xg,
                'shots': analysis.shot_count,
                'goals': analysis.goals,
                'xg_per_shot': analysis.xg_per_shot,
                'conversion_rate': analysis.conversion_rate,
                'over_performance': analysis.over_performance
            }

        return comparison

    def reset(self):
        """Reset all tracked data."""
        self.shots.clear()


class xGZoneAnalyzer:
    """
    Analyze xG by pitch zones for identifying dangerous areas.
    """

    def __init__(
        self,
        model: Optional[ExpectedGoalsModel] = None,
        zone_size: float = 5.0  # meters per zone
    ):
        self.model = model or ExpectedGoalsModel()
        self.zone_size = zone_size
        self.zone_shots: Dict[Tuple[int, int], List[float]] = {}

    def add_shot(self, x: float, y: float, xg: float):
        """Add a shot to zone tracking."""
        zone_x = int(x / self.zone_size)
        zone_y = int(y / self.zone_size)
        zone = (zone_x, zone_y)

        if zone not in self.zone_shots:
            self.zone_shots[zone] = []

        self.zone_shots[zone].append(xg)

    def get_zone_stats(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Get statistics for each zone."""
        stats = {}

        for zone, xgs in self.zone_shots.items():
            stats[zone] = {
                'shot_count': len(xgs),
                'total_xg': sum(xgs),
                'avg_xg': sum(xgs) / len(xgs) if xgs else 0,
                'max_xg': max(xgs) if xgs else 0
            }

        return stats

    def get_danger_heatmap(
        self,
        grid_x: int = 20,
        grid_y: int = 14
    ) -> np.ndarray:
        """
        Generate xG-weighted danger heatmap.

        Returns 2D array of xG density values.
        """
        heatmap = np.zeros((grid_y, grid_x))

        for zone, xgs in self.zone_shots.items():
            if 0 <= zone[0] < grid_x and 0 <= zone[1] < grid_y:
                heatmap[zone[1], zone[0]] = sum(xgs)

        return heatmap

    def get_theoretical_xg_map(
        self,
        grid_x: int = 40,
        grid_y: int = 28,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0
    ) -> np.ndarray:
        """
        Generate theoretical xG values across the pitch.

        Useful for visualization of shooting zones.
        """
        xg_map = np.zeros((grid_y, grid_x))

        for i in range(grid_x):
            for j in range(grid_y):
                # Convert grid to pitch coordinates
                x = (i + 0.5) * (pitch_length / grid_x)
                y = (j + 0.5) * (pitch_width / grid_y)

                # Only calculate for attacking half
                if x < pitch_length / 2:
                    xg_map[j, i] = 0
                else:
                    # Calculate xG from goal line (x=0 for shooting)
                    shot_x = pitch_length - x
                    context = ShotContext(x=shot_x, y=y)
                    xg_map[j, i] = self.model.calculate_base_xg(shot_x, y)

        return xg_map
