"""
Space Creation Analysis & Third-Man Run Detection
Tracks player runs that create space for teammates
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set, Any
from collections import defaultdict
from enum import Enum
import math
from loguru import logger


class RunType(Enum):
    """Type of off-ball movement."""
    DECOY_RUN = "decoy_run"  # Run to draw defenders
    THIRD_MAN_RUN = "third_man_run"  # Classic third-man pattern
    OVERLAP_RUN = "overlap_run"  # Overlapping run wide
    UNDERLAP_RUN = "underlap_run"  # Inside run behind defense
    DIAGONAL_RUN = "diagonal_run"  # Diagonal run into space
    CHANNEL_RUN = "channel_run"  # Run between defenders
    DUMMY_RUN = "dummy_run"  # Run without receiving
    PULL_BACK_RUN = "pull_back_run"  # Movement to receive to feet
    BLINDSIDE_RUN = "blindside_run"  # Run from defender's blindside


class SpaceType(Enum):
    """Type of space created or exploited."""
    DEFENSIVE_THIRD = "defensive_third"
    MIDDLE_THIRD = "middle_third"
    ATTACKING_THIRD = "attacking_third"
    HALF_SPACE = "half_space"
    WIDE_CHANNEL = "wide_channel"
    CENTRAL_CHANNEL = "central_channel"
    BEHIND_DEFENSE = "behind_defense"
    BOX = "box"


@dataclass
class PlayerMovement:
    """Tracking data for a player's movement."""
    player_id: int
    team_id: int
    start_frame: int
    end_frame: int
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    velocity: float  # meters per second
    direction: float  # radians, 0 = towards opponent goal
    acceleration: float
    distance_covered: float


@dataclass
class SpaceCreated:
    """Space created by a player's movement."""
    frame: int
    timestamp: float
    creator_id: int
    team_id: int
    run_type: RunType
    space_type: SpaceType
    space_area: float  # square meters
    space_location: Tuple[float, float]
    beneficiary_id: Optional[int]  # Player who could use the space
    defenders_drawn: List[int]  # Defender IDs pulled by the run
    was_exploited: bool = False


@dataclass
class ThirdManRun:
    """Detected third-man run pattern."""
    frame: int
    timestamp: float
    team_id: int

    # The three players involved
    player_a_id: int  # Initial passer
    player_b_id: int  # Receiver who lays off
    player_c_id: int  # Third man making the run

    # Positions
    player_a_position: Tuple[float, float]
    player_b_position: Tuple[float, float]
    player_c_start: Tuple[float, float]
    player_c_end: Tuple[float, float]

    # Quality metrics
    run_distance: float
    run_speed: float
    space_gained: float  # meters advanced
    defenders_beaten: int
    success: bool = False  # Did pass reach player C?
    confidence: float = 0.0


@dataclass
class SpaceAnalysisResult:
    """Complete space analysis for a period."""
    team_id: int
    total_space_created: float  # square meters
    run_count: int
    runs_by_type: Dict[RunType, int]
    third_man_runs: List[ThirdManRun]
    exploitation_rate: float  # % of created space that was used
    top_creators: List[Tuple[int, float]]  # player_id, space_created


class SpaceCalculator:
    """
    Calculates available space on the pitch.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        space_threshold: float = 4.0  # meters - distance for "space"
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.space_threshold = space_threshold

    def calculate_space_at_point(
        self,
        point: Tuple[float, float],
        defender_positions: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate available space around a point.

        Returns the area of the largest circle that fits
        without touching defenders.
        """
        if not defender_positions:
            return math.pi * self.space_threshold ** 2

        # Find nearest defender
        min_dist = float('inf')
        for def_pos in defender_positions:
            dist = math.sqrt(
                (point[0] - def_pos[0]) ** 2 +
                (point[1] - def_pos[1]) ** 2
            )
            min_dist = min(min_dist, dist)

        # Space is proportional to distance squared
        effective_radius = min(min_dist / 2, self.space_threshold)
        return math.pi * effective_radius ** 2

    def calculate_space_grid(
        self,
        defender_positions: List[Tuple[float, float]],
        grid_size: int = 20
    ) -> np.ndarray:
        """
        Calculate space availability across the pitch.

        Returns a grid of space values.
        """
        space_grid = np.zeros((grid_size, grid_size))

        cell_width = self.pitch_length / grid_size
        cell_height = self.pitch_width / grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                point = (
                    (i + 0.5) * cell_width,
                    (j + 0.5) * cell_height
                )
                space_grid[j, i] = self.calculate_space_at_point(
                    point, defender_positions
                )

        return space_grid

    def find_open_spaces(
        self,
        defender_positions: List[Tuple[float, float]],
        min_space: float = 30.0  # minimum square meters
    ) -> List[Tuple[Tuple[float, float], float]]:
        """
        Find open spaces on the pitch.

        Returns list of (position, space_area) tuples.
        """
        open_spaces = []
        grid_size = 30

        cell_width = self.pitch_length / grid_size
        cell_height = self.pitch_width / grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                point = (
                    (i + 0.5) * cell_width,
                    (j + 0.5) * cell_height
                )
                space = self.calculate_space_at_point(point, defender_positions)

                if space >= min_space:
                    open_spaces.append((point, space))

        return sorted(open_spaces, key=lambda x: x[1], reverse=True)


class MovementTracker:
    """
    Tracks player movements frame by frame.
    """

    def __init__(
        self,
        fps: float = 30.0,
        min_run_distance: float = 5.0,  # meters
        min_run_speed: float = 4.0,  # m/s (jogging pace)
        sprint_threshold: float = 7.0  # m/s
    ):
        self.fps = fps
        self.min_run_distance = min_run_distance
        self.min_run_speed = min_run_speed
        self.sprint_threshold = sprint_threshold

        # Position history per player
        self.position_history: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        # frame, x, y

    def update(
        self,
        frame: int,
        player_positions: Dict[int, Tuple[float, float]]
    ):
        """Update position history for all players."""
        for player_id, (x, y) in player_positions.items():
            self.position_history[player_id].append((frame, x, y))

            # Keep only last 90 frames (3 seconds)
            if len(self.position_history[player_id]) > 90:
                self.position_history[player_id] = self.position_history[player_id][-90:]

    def get_velocity(
        self,
        player_id: int,
        window: int = 5
    ) -> Tuple[float, float, float]:
        """
        Get current velocity for a player.

        Returns (speed, direction, acceleration)
        """
        history = self.position_history.get(player_id, [])

        if len(history) < 2:
            return 0.0, 0.0, 0.0

        recent = history[-window:]
        if len(recent) < 2:
            return 0.0, 0.0, 0.0

        # Calculate displacement
        dx = recent[-1][1] - recent[0][1]
        dy = recent[-1][2] - recent[0][2]
        dt = (recent[-1][0] - recent[0][0]) / self.fps

        if dt <= 0:
            return 0.0, 0.0, 0.0

        # Velocity
        vx = dx / dt
        vy = dy / dt
        speed = math.sqrt(vx ** 2 + vy ** 2)
        direction = math.atan2(dy, dx)

        # Acceleration (if enough history)
        acceleration = 0.0
        if len(history) >= window * 2:
            old_recent = history[-(window * 2):-window]
            if len(old_recent) >= 2:
                old_dx = old_recent[-1][1] - old_recent[0][1]
                old_dy = old_recent[-1][2] - old_recent[0][2]
                old_dt = (old_recent[-1][0] - old_recent[0][0]) / self.fps
                if old_dt > 0:
                    old_speed = math.sqrt(old_dx ** 2 + old_dy ** 2) / old_dt
                    acceleration = (speed - old_speed) / dt

        return speed, direction, acceleration

    def detect_run(
        self,
        player_id: int,
        team_id: int,
        frames_back: int = 30
    ) -> Optional[PlayerMovement]:
        """
        Detect if player is making a significant run.
        """
        history = self.position_history.get(player_id, [])

        if len(history) < frames_back:
            return None

        start = history[-frames_back]
        end = history[-1]

        dx = end[1] - start[1]
        dy = end[2] - start[2]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance < self.min_run_distance:
            return None

        dt = (end[0] - start[0]) / self.fps
        if dt <= 0:
            return None

        speed = distance / dt

        if speed < self.min_run_speed:
            return None

        speed_current, direction, acceleration = self.get_velocity(player_id)

        movement = PlayerMovement(
            player_id=player_id,
            team_id=team_id,
            start_frame=start[0],
            end_frame=end[0],
            start_position=(start[1], start[2]),
            end_position=(end[1], end[2]),
            velocity=speed,
            direction=direction,
            acceleration=acceleration,
            distance_covered=distance
        )

        logger.debug(
            f"Run detected: player={player_id}, distance={distance:.1f}m, "
            f"speed={speed:.1f}m/s, direction={direction:.0f}°"
        )

        return movement


class SpaceCreationAnalyzer:
    """
    Analyzes space creation from player movements.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        self.space_calculator = SpaceCalculator(pitch_length, pitch_width)
        self.movement_tracker = MovementTracker()

        self.space_events: List[SpaceCreated] = []
        self.player_space_created: Dict[int, float] = defaultdict(float)

        logger.debug(f"SpaceCreationAnalyzer initialized: pitch={pitch_length}x{pitch_width}m")

    def classify_run_type(
        self,
        movement: PlayerMovement,
        ball_position: Tuple[float, float],
        teammate_positions: Dict[int, Tuple[float, float]],
        defender_positions: List[Tuple[float, float]]
    ) -> RunType:
        """Classify the type of run being made."""
        start = movement.start_position
        end = movement.end_position

        # Direction towards goal
        towards_goal = end[0] > start[0]
        moving_wide = abs(end[1] - self.pitch_width / 2) > abs(start[1] - self.pitch_width / 2)
        moving_central = abs(end[1] - self.pitch_width / 2) < abs(start[1] - self.pitch_width / 2)

        # Check if run is behind defensive line
        if defender_positions:
            max_defender_x = max(pos[0] for pos in defender_positions)
            behind_defense = end[0] > max_defender_x

            if behind_defense:
                if moving_wide:
                    return RunType.OVERLAP_RUN
                elif moving_central:
                    return RunType.UNDERLAP_RUN
                else:
                    return RunType.CHANNEL_RUN

        # Check diagonal movement
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = abs(math.atan2(dy, dx))

        if 0.3 < angle < 1.2:  # Roughly 15-70 degrees
            return RunType.DIAGONAL_RUN

        # Check if decoy (moving away from play)
        dist_to_ball_start = math.sqrt(
            (start[0] - ball_position[0]) ** 2 +
            (start[1] - ball_position[1]) ** 2
        )
        dist_to_ball_end = math.sqrt(
            (end[0] - ball_position[0]) ** 2 +
            (end[1] - ball_position[1]) ** 2
        )

        if dist_to_ball_end > dist_to_ball_start + 5:
            return RunType.DECOY_RUN

        # Pull back if moving towards own goal to receive
        if not towards_goal and dist_to_ball_end < dist_to_ball_start:
            return RunType.PULL_BACK_RUN

        # Default to dummy run
        return RunType.DUMMY_RUN

    def classify_space_type(
        self,
        position: Tuple[float, float]
    ) -> SpaceType:
        """Classify the zone where space was created."""
        x, y = position

        # Thirds
        third_length = self.pitch_length / 3

        if x < third_length:
            return SpaceType.DEFENSIVE_THIRD
        elif x < 2 * third_length:
            # Check half-spaces
            if self.pitch_width * 0.2 < y < self.pitch_width * 0.4:
                return SpaceType.HALF_SPACE
            elif self.pitch_width * 0.6 < y < self.pitch_width * 0.8:
                return SpaceType.HALF_SPACE
            elif y < self.pitch_width * 0.2 or y > self.pitch_width * 0.8:
                return SpaceType.WIDE_CHANNEL
            else:
                return SpaceType.MIDDLE_THIRD
        else:
            # Attacking third
            if x > self.pitch_length - 16.5:  # Box
                return SpaceType.BOX
            return SpaceType.ATTACKING_THIRD

    def analyze_frame(
        self,
        frame: int,
        timestamp: float,
        team_positions: Dict[int, Dict[int, Tuple[float, float]]],
        ball_position: Tuple[float, float],
        team_in_possession: int
    ) -> List[SpaceCreated]:
        """
        Analyze space creation in a single frame.

        Args:
            frame: Frame number
            timestamp: Time in seconds
            team_positions: {team_id: {player_id: (x, y)}}
            ball_position: Ball (x, y)
            team_in_possession: Team with ball

        Returns:
            List of space created events
        """
        events = []

        for team_id, positions in team_positions.items():
            # Update movement tracking
            for player_id, pos in positions.items():
                self.movement_tracker.update(frame, {player_id: pos})

        # Only analyze team in possession for space creation
        if team_in_possession not in team_positions:
            return events

        our_positions = team_positions[team_in_possession]

        # Get opponent positions
        opponent_team = [t for t in team_positions if t != team_in_possession]
        if not opponent_team:
            return events

        defender_positions = list(team_positions[opponent_team[0]].values())

        # Detect runs and analyze space creation
        for player_id, position in our_positions.items():
            movement = self.movement_tracker.detect_run(
                player_id, team_in_possession, frames_back=15
            )

            if movement is None:
                continue

            # Calculate space created
            space_before = self.space_calculator.calculate_space_at_point(
                movement.start_position, defender_positions
            )
            space_after = self.space_calculator.calculate_space_at_point(
                movement.end_position, defender_positions
            )

            # Check space vacated
            space_vacated = self.space_calculator.calculate_space_at_point(
                movement.start_position, defender_positions
            )

            # Identify defenders drawn by the run
            defenders_drawn = []
            for i, def_pos in enumerate(defender_positions):
                dist_to_runner = math.sqrt(
                    (movement.end_position[0] - def_pos[0]) ** 2 +
                    (movement.end_position[1] - def_pos[1]) ** 2
                )
                if dist_to_runner < 5.0:  # Within 5 meters
                    defenders_drawn.append(i)

            if len(defenders_drawn) > 0 or space_vacated > 20:
                run_type = self.classify_run_type(
                    movement, ball_position, our_positions, defender_positions
                )
                space_type = self.classify_space_type(movement.start_position)

                # Find potential beneficiary
                beneficiary = None
                max_benefit = 0

                for other_id, other_pos in our_positions.items():
                    if other_id == player_id:
                        continue

                    dist_to_vacated = math.sqrt(
                        (other_pos[0] - movement.start_position[0]) ** 2 +
                        (other_pos[1] - movement.start_position[1]) ** 2
                    )

                    if dist_to_vacated < 15 and space_vacated > max_benefit:
                        beneficiary = other_id
                        max_benefit = space_vacated

                event = SpaceCreated(
                    frame=frame,
                    timestamp=timestamp,
                    creator_id=player_id,
                    team_id=team_in_possession,
                    run_type=run_type,
                    space_type=space_type,
                    space_area=space_vacated,
                    space_location=movement.start_position,
                    beneficiary_id=beneficiary,
                    defenders_drawn=defenders_drawn,
                    was_exploited=False
                )

                events.append(event)
                self.space_events.append(event)
                self.player_space_created[player_id] += space_vacated

        return events

    def get_analysis(self, team_id: int) -> SpaceAnalysisResult:
        """Get complete space analysis for a team."""
        team_events = [e for e in self.space_events if e.team_id == team_id]

        runs_by_type: Dict[RunType, int] = defaultdict(int)
        for event in team_events:
            runs_by_type[event.run_type] += 1

        total_space = sum(e.space_area for e in team_events)
        exploited = sum(1 for e in team_events if e.was_exploited)
        exploitation_rate = exploited / len(team_events) if team_events else 0

        # Get top creators
        player_space: Dict[int, float] = defaultdict(float)
        for event in team_events:
            player_space[event.creator_id] += event.space_area

        top_creators = sorted(
            player_space.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return SpaceAnalysisResult(
            team_id=team_id,
            total_space_created=total_space,
            run_count=len(team_events),
            runs_by_type=dict(runs_by_type),
            third_man_runs=[],  # Filled by ThirdManRunDetector
            exploitation_rate=exploitation_rate,
            top_creators=top_creators
        )


class ThirdManRunDetector:
    """
    Detects third-man run patterns.

    A third-man run occurs when:
    1. Player A passes to Player B
    2. Player B receives and lays off (first touch or quick pass)
    3. Player C makes a run to receive the layoff
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        layoff_time_limit: float = 2.0,  # seconds
        run_trigger_distance: float = 15.0  # meters
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.layoff_time_limit = layoff_time_limit
        self.run_trigger_distance = run_trigger_distance

        self.movement_tracker = MovementTracker()
        self.detected_runs: List[ThirdManRun] = []

        # Pass tracking
        self.recent_passes: List[Dict] = []

    def register_pass(
        self,
        frame: int,
        timestamp: float,
        passer_id: int,
        receiver_id: int,
        team_id: int,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float]
    ):
        """Register a pass for third-man detection."""
        self.recent_passes.append({
            'frame': frame,
            'timestamp': timestamp,
            'passer_id': passer_id,
            'receiver_id': receiver_id,
            'team_id': team_id,
            'start_pos': start_pos,
            'end_pos': end_pos
        })

        # Keep only recent passes
        cutoff = timestamp - self.layoff_time_limit * 2
        self.recent_passes = [
            p for p in self.recent_passes
            if p['timestamp'] > cutoff
        ]

    def update_positions(
        self,
        frame: int,
        player_positions: Dict[int, Tuple[float, float]]
    ):
        """Update position tracking for all players."""
        for player_id, pos in player_positions.items():
            self.movement_tracker.update(frame, {player_id: pos})

    def detect(
        self,
        frame: int,
        timestamp: float,
        team_positions: Dict[int, Dict[int, Tuple[float, float]]],
        latest_pass: Optional[Dict] = None
    ) -> Optional[ThirdManRun]:
        """
        Detect third-man run patterns.

        Call this after each pass is registered.

        Returns ThirdManRun if pattern detected.
        """
        if latest_pass is None and not self.recent_passes:
            return None

        pass_to_check = latest_pass or self.recent_passes[-1]

        # Check if this is a potential layoff (quick pass after receiving)
        team_id = pass_to_check['team_id']
        player_b = pass_to_check['passer_id']  # Current passer received ball

        # Find the previous pass to player B
        previous_pass = None
        for p in reversed(self.recent_passes[:-1]):
            if p['receiver_id'] == player_b and p['team_id'] == team_id:
                # Check time difference
                time_diff = pass_to_check['timestamp'] - p['timestamp']
                if time_diff <= self.layoff_time_limit:
                    previous_pass = p
                    break

        if previous_pass is None:
            return None

        player_a = previous_pass['passer_id']
        player_c = pass_to_check['receiver_id']

        # Verify it's a third-man pattern (all different players)
        if len({player_a, player_b, player_c}) != 3:
            return None

        # Get positions
        our_positions = team_positions.get(team_id, {})

        if player_c not in our_positions:
            return None

        # Check if player C made a run
        movement = self.movement_tracker.detect_run(
            player_c, team_id, frames_back=30
        )

        if movement is None:
            return None

        # Check if run was forward (progressive)
        start_x = movement.start_position[0]
        end_x = movement.end_position[0]

        if end_x <= start_x:
            return None

        # Calculate metrics
        run_distance = math.sqrt(
            (end_x - start_x) ** 2 +
            (movement.end_position[1] - movement.start_position[1]) ** 2
        )

        if run_distance < 5:  # Minimum run distance
            return None

        # Check if run beats defenders
        opponent_team = [t for t in team_positions if t != team_id]
        defenders_beaten = 0

        if opponent_team:
            defender_positions = team_positions[opponent_team[0]]
            for def_pos in defender_positions.values():
                if start_x < def_pos[0] < end_x:
                    defenders_beaten += 1

        third_man_run = ThirdManRun(
            frame=frame,
            timestamp=timestamp,
            team_id=team_id,
            player_a_id=player_a,
            player_b_id=player_b,
            player_c_id=player_c,
            player_a_position=previous_pass['start_pos'],
            player_b_position=previous_pass['end_pos'],
            player_c_start=movement.start_position,
            player_c_end=movement.end_position,
            run_distance=run_distance,
            run_speed=movement.velocity,
            space_gained=end_x - start_x,
            defenders_beaten=defenders_beaten,
            success=True,  # Pass was made
            confidence=min(0.9, 0.5 + run_distance / 20 + defenders_beaten * 0.1)
        )

        self.detected_runs.append(third_man_run)

        logger.info(
            f"Third-man run detected: A({player_a}) -> B({player_b}) -> C({player_c}), "
            f"distance={run_distance:.1f}m, defenders_beaten={defenders_beaten}"
        )

        return third_man_run

    def get_third_man_runs(
        self,
        team_id: Optional[int] = None
    ) -> List[ThirdManRun]:
        """Get all detected third-man runs."""
        if team_id is None:
            return self.detected_runs.copy()

        return [r for r in self.detected_runs if r.team_id == team_id]

    def get_player_involvement(
        self,
        player_id: int
    ) -> Dict[str, int]:
        """Get player's involvement in third-man patterns."""
        as_a = sum(1 for r in self.detected_runs if r.player_a_id == player_id)
        as_b = sum(1 for r in self.detected_runs if r.player_b_id == player_id)
        as_c = sum(1 for r in self.detected_runs if r.player_c_id == player_id)

        return {
            'as_initiator': as_a,
            'as_pivot': as_b,
            'as_runner': as_c,
            'total': as_a + as_b + as_c
        }

    def reset(self):
        """Reset all tracking data."""
        self.recent_passes.clear()
        self.detected_runs.clear()
        self.movement_tracker.position_history.clear()


class OffBallMovementAnalyzer:
    """
    Comprehensive off-ball movement analysis combining
    space creation and third-man run detection.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0
    ):
        self.space_analyzer = SpaceCreationAnalyzer(pitch_length, pitch_width)
        self.third_man_detector = ThirdManRunDetector(pitch_length, pitch_width)

    def process_frame(
        self,
        frame: int,
        timestamp: float,
        team_positions: Dict[int, Dict[int, Tuple[float, float]]],
        ball_position: Tuple[float, float],
        team_in_possession: int
    ) -> Dict[str, Any]:
        """
        Process a frame for all off-ball movement analysis.

        Returns dict with detected events.
        """
        # Update third-man detector positions
        for team_id, positions in team_positions.items():
            self.third_man_detector.update_positions(frame, positions)

        # Analyze space creation
        space_events = self.space_analyzer.analyze_frame(
            frame, timestamp, team_positions, ball_position, team_in_possession
        )

        return {
            'space_created': space_events,
            'frame': frame,
            'timestamp': timestamp
        }

    def register_pass(
        self,
        frame: int,
        timestamp: float,
        passer_id: int,
        receiver_id: int,
        team_id: int,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        team_positions: Dict[int, Dict[int, Tuple[float, float]]]
    ) -> Optional[ThirdManRun]:
        """
        Register a pass and check for third-man pattern.
        """
        self.third_man_detector.register_pass(
            frame, timestamp, passer_id, receiver_id,
            team_id, start_pos, end_pos
        )

        return self.third_man_detector.detect(
            frame, timestamp, team_positions
        )

    def get_comprehensive_analysis(
        self,
        team_id: int
    ) -> Dict[str, Any]:
        """Get full analysis combining all metrics."""
        space_analysis = self.space_analyzer.get_analysis(team_id)
        third_man_runs = self.third_man_detector.get_third_man_runs(team_id)

        # Add third-man runs to space analysis
        space_analysis.third_man_runs = third_man_runs

        return {
            'space_analysis': space_analysis,
            'third_man_runs': third_man_runs,
            'third_man_count': len(third_man_runs),
            'successful_third_man': sum(1 for r in third_man_runs if r.success),
            'avg_third_man_distance': (
                sum(r.run_distance for r in third_man_runs) / len(third_man_runs)
                if third_man_runs else 0
            ),
            'total_defenders_beaten': sum(r.defenders_beaten for r in third_man_runs)
        }

    def reset(self):
        """Reset all analysis data."""
        self.space_analyzer.space_events.clear()
        self.space_analyzer.player_space_created.clear()
        self.third_man_detector.reset()
