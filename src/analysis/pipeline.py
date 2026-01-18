"""
Analysis Pipeline
Orchestrates all tactical analysis modules during video processing.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
from loguru import logger

from src.detection.detector import FrameDetections, PlayerDetection

# Import all analyzers
from .formation_detection import FormationDetector, FormationType, FormationSnapshot
from .tactical_analytics import (
    PressingAnalyzer,
    CounterAttackDetector,
    PitchZoneAnalyzer,
)
from .advanced_tactical import TeamShapeAnalyzer, FatigueDetector
from .expected_goals import xGTracker, ShotContext, ShotType, ShotSituation
from .space_analysis import SpaceCreationAnalyzer, ThirdManRunDetector


class AnalysisDepthLevel(Enum):
    """How much analysis to run per frame."""
    MINIMAL = "minimal"      # Just possession
    STANDARD = "standard"    # + formations, team shape
    COMPREHENSIVE = "comprehensive"  # + all analyzers


@dataclass
class AnalysisPipelineConfig:
    """Configuration for which analyzers to enable."""
    # Core analysis (always on)
    track_possession: bool = True

    # Formation analysis
    detect_formations: bool = True
    formation_update_interval: int = 30  # frames between formation checks

    # Team shape analysis
    analyze_team_shape: bool = True
    team_shape_interval: int = 15  # frames between shape calculations

    # Pressing analysis
    analyze_pressing: bool = True

    # Space analysis
    analyze_space_creation: bool = True

    # Counter attack detection
    detect_counter_attacks: bool = True

    # xG tracking (requires shot detection)
    track_xg: bool = True

    # Fatigue detection
    detect_fatigue: bool = True
    fatigue_update_interval: int = 900  # frames (~30 sec at 30fps)

    # Pitch dimensions (meters)
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    @classmethod
    def from_depth(cls, depth: AnalysisDepthLevel) -> "AnalysisPipelineConfig":
        """Create config based on analysis depth."""
        if depth == AnalysisDepthLevel.MINIMAL:
            return cls(
                detect_formations=False,
                analyze_team_shape=False,
                analyze_pressing=False,
                analyze_space_creation=False,
                detect_counter_attacks=False,
                track_xg=False,
                detect_fatigue=False
            )
        elif depth == AnalysisDepthLevel.STANDARD:
            return cls(
                analyze_space_creation=False,
                detect_fatigue=False
            )
        else:  # COMPREHENSIVE
            return cls()


@dataclass
class FrameAnalysisResult:
    """Analysis results for a single frame."""
    frame_number: int
    timestamp: float

    # Possession
    possession_team: Optional[int] = None

    # Formation
    home_formation: Optional[FormationSnapshot] = None
    away_formation: Optional[FormationSnapshot] = None

    # Team shape
    home_team_shape: Optional[Dict[str, float]] = None
    away_team_shape: Optional[Dict[str, float]] = None

    # Pressing
    pressing_intensity: Optional[float] = None
    ppda: Optional[float] = None

    # Space creation events
    space_events: List[Any] = field(default_factory=list)

    # Counter attack
    counter_attack_detected: bool = False
    counter_attack_team: Optional[int] = None

    # Shots detected this frame
    shots: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MatchAnalysisSummary:
    """Summary statistics for the entire match."""
    frames_analyzed: int = 0

    # Possession
    possession_home: float = 0.0
    possession_away: float = 0.0

    # Formations (most common)
    home_primary_formation: Optional[FormationType] = None
    away_primary_formation: Optional[FormationType] = None
    formation_changes_home: int = 0
    formation_changes_away: int = 0

    # Team shape averages
    home_avg_compactness: float = 0.0
    away_avg_compactness: float = 0.0
    home_avg_width: float = 0.0
    away_avg_width: float = 0.0

    # Pressing
    home_ppda: float = 0.0
    away_ppda: float = 0.0

    # xG
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_shots: int = 0
    away_shots: int = 0

    # Counter attacks
    home_counter_attacks: int = 0
    away_counter_attacks: int = 0

    # Space creation
    total_runs_detected: int = 0


class AnalysisPipeline:
    """
    Orchestrates tactical analysis during video processing.

    Usage:
        pipeline = AnalysisPipeline(config)
        for frame_detections in video_frames:
            result = pipeline.process_frame(frame_detections)
        summary = pipeline.get_summary()
    """

    def __init__(self, config: Optional[AnalysisPipelineConfig] = None):
        self.config = config or AnalysisPipelineConfig()
        self._frame_count = 0
        self._last_possession_team: Optional[int] = None

        # Initialize analyzers based on config
        self._init_analyzers()

        # Tracking data
        self._possession_frames = {0: 0, 1: 0}  # team_id -> frame count
        self._formation_history = {0: [], 1: []}  # team_id -> list of formations
        self._team_shape_history = {0: [], 1: []}
        self._counter_attacks = {0: 0, 1: 0}
        self._runs_detected = 0

        logger.info(f"AnalysisPipeline initialized with config: {self._get_enabled_features()}")

    def _init_analyzers(self):
        """Initialize all enabled analyzers."""
        pitch_length = self.config.pitch_length
        pitch_width = self.config.pitch_width

        # Formation detector
        self.formation_detector: Optional[FormationDetector] = None
        if self.config.detect_formations:
            self.formation_detector = FormationDetector(
                pitch_length=pitch_length,
                pitch_width=pitch_width
            )

        # Team shape analyzer
        self.team_shape_analyzer: Optional[TeamShapeAnalyzer] = None
        if self.config.analyze_team_shape:
            self.team_shape_analyzer = TeamShapeAnalyzer(
                pitch_length=pitch_length,
                pitch_width=pitch_width
            )

        # Pressing analyzer
        self.pressing_analyzer: Optional[PressingAnalyzer] = None
        if self.config.analyze_pressing:
            self.pressing_analyzer = PressingAnalyzer()

        # Space creation analyzer
        self.space_analyzer: Optional[SpaceCreationAnalyzer] = None
        if self.config.analyze_space_creation:
            self.space_analyzer = SpaceCreationAnalyzer(
                pitch_length=pitch_length,
                pitch_width=pitch_width
            )

        # Counter attack detector
        self.counter_detector: Optional[CounterAttackDetector] = None
        if self.config.detect_counter_attacks:
            self.counter_detector = CounterAttackDetector()

        # xG tracker
        self.xg_tracker: Optional[xGTracker] = None
        if self.config.track_xg:
            self.xg_tracker = xGTracker()

        # Fatigue detector
        self.fatigue_detector: Optional[FatigueDetector] = None
        if self.config.detect_fatigue:
            self.fatigue_detector = FatigueDetector()

    def _get_enabled_features(self) -> str:
        """Get string of enabled features for logging."""
        features = []
        if self.config.detect_formations:
            features.append("formations")
        if self.config.analyze_team_shape:
            features.append("team_shape")
        if self.config.analyze_pressing:
            features.append("pressing")
        if self.config.analyze_space_creation:
            features.append("space")
        if self.config.detect_counter_attacks:
            features.append("counters")
        if self.config.track_xg:
            features.append("xG")
        if self.config.detect_fatigue:
            features.append("fatigue")
        return ", ".join(features) if features else "minimal"

    def process_frame(
        self,
        detections: FrameDetections,
        fps: float = 30.0
    ) -> FrameAnalysisResult:
        """
        Process a single frame through all enabled analyzers.

        Args:
            detections: Detection results for this frame
            fps: Video frame rate

        Returns:
            FrameAnalysisResult with all analysis for this frame
        """
        self._frame_count += 1
        frame_num = detections.frame_number
        timestamp = detections.timestamp_seconds

        result = FrameAnalysisResult(
            frame_number=frame_num,
            timestamp=timestamp
        )

        # Extract player positions by team
        team_positions = self._extract_team_positions(detections)
        ball_position = self._get_ball_position(detections)

        # 1. Track possession
        result.possession_team = detections.possession_team
        possession_changed = False
        if result.possession_team is not None:
            if self._last_possession_team != result.possession_team:
                possession_changed = True
            self._possession_frames[result.possession_team] = \
                self._possession_frames.get(result.possession_team, 0) + 1
            self._last_possession_team = result.possession_team

        # 2. Formation detection (periodic)
        if self.formation_detector and self._should_update(frame_num, self.config.formation_update_interval):
            result.home_formation = self._detect_formation(team_positions, 0, frame_num)
            result.away_formation = self._detect_formation(team_positions, 1, frame_num)

        # 3. Team shape analysis (periodic)
        if self.team_shape_analyzer and self._should_update(frame_num, self.config.team_shape_interval):
            result.home_team_shape, result.away_team_shape = self._analyze_team_shape(
                team_positions, frame_num
            )

        # 4. Pressing analysis
        if self.pressing_analyzer:
            pressing_result = self._analyze_pressing(
                team_positions, ball_position, result.possession_team
            )
            result.pressing_intensity = pressing_result.get("intensity")
            result.ppda = pressing_result.get("ppda")

        # 5. Space creation analysis
        if self.space_analyzer and ball_position:
            space_events = self._analyze_space_creation(
                frame_num, team_positions, ball_position, timestamp, result.possession_team
            )
            result.space_events = space_events
            self._runs_detected += len(space_events)

        # 6. Counter attack detection
        if self.counter_detector:
            counter_result = self._detect_counter_attack(
                frame_num, timestamp, ball_position,
                result.possession_team, possession_changed
            )
            result.counter_attack_detected = counter_result.get("detected", False)
            result.counter_attack_team = counter_result.get("team")
            if result.counter_attack_detected and result.counter_attack_team is not None:
                self._counter_attacks[result.counter_attack_team] += 1

        # 7. Update fatigue tracking (periodic)
        if self.fatigue_detector and self._should_update(frame_num, self.config.fatigue_update_interval):
            self._update_fatigue_tracking(detections, timestamp)

        # Log periodic summary
        if self._frame_count % 300 == 0:  # Every 10 seconds at 30fps
            self._log_progress_summary()

        return result

    def _should_update(self, frame_num: int, interval: int) -> bool:
        """Check if we should run a periodic update."""
        return frame_num % interval == 0

    def _extract_team_positions(
        self,
        detections: FrameDetections
    ) -> Dict[int, Dict[int, Tuple[float, float]]]:
        """Extract player positions organized by team."""
        team_positions: Dict[int, Dict[int, Tuple[float, float]]] = {0: {}, 1: {}}

        all_players = detections.players + detections.goalkeepers

        for player in all_players:
            if player.team_id is not None and player.team_id in [0, 1]:
                player_id = player.tracker_id or id(player)
                center = player.center
                team_positions[player.team_id][player_id] = center

        return team_positions

    def _get_ball_position(self, detections: FrameDetections) -> Optional[Tuple[float, float]]:
        """Get ball position from detections."""
        if detections.ball:
            return detections.ball.center
        return None

    def _detect_formation(
        self,
        team_positions: Dict[int, Dict[int, Tuple[float, float]]],
        team_id: int,
        frame_num: int
    ) -> Optional[FormationSnapshot]:
        """Detect formation for a team."""
        positions = team_positions.get(team_id, {})

        if len(positions) < 7:  # Need minimum players
            return None

        try:
            # Determine attacking direction (0=left-to-right, 1=right-to-left)
            attacking_direction = 1 if team_id == 0 else -1

            formation = self.formation_detector.detect_formation(
                frame=frame_num,
                team_positions=positions,
                team_id=team_id,
                attacking_direction=attacking_direction
            )

            if formation:
                self._formation_history[team_id].append(formation.formation_type)
                # Keep only last 100 formations
                if len(self._formation_history[team_id]) > 100:
                    self._formation_history[team_id] = self._formation_history[team_id][-100:]

            return formation
        except Exception as e:
            logger.debug(f"Formation detection failed for team {team_id}: {e}")
            return None

    def _analyze_team_shape(
        self,
        team_positions: Dict[int, Dict[int, Tuple[float, float]]],
        frame_num: int
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """Analyze team shape metrics for both teams."""
        home_positions = list(team_positions.get(0, {}).values())
        away_positions = list(team_positions.get(1, {}).values())

        if len(home_positions) < 4 and len(away_positions) < 4:
            return None, None

        try:
            # Normalize positions to 0-1 range for the analyzer
            # Assuming positions are in pixels, normalize by frame dimensions
            # The analyzer expects normalized positions
            home_metrics, away_metrics = self.team_shape_analyzer.analyze_frame(
                frame_num=frame_num,
                home_positions=home_positions,
                away_positions=away_positions
            )

            # Convert metrics to dict for storage
            home_shape = None
            away_shape = None

            if home_metrics:
                home_shape = {
                    "compactness": home_metrics.compactness,
                    "width": home_metrics.width,
                    "length": home_metrics.length,
                    "defensive_line": home_metrics.defensive_line,
                }
                self._team_shape_history[0].append(home_shape)

            if away_metrics:
                away_shape = {
                    "compactness": away_metrics.compactness,
                    "width": away_metrics.width,
                    "length": away_metrics.length,
                    "defensive_line": away_metrics.defensive_line,
                }
                self._team_shape_history[1].append(away_shape)

            # Trim history
            for team_id in [0, 1]:
                if len(self._team_shape_history[team_id]) > 100:
                    self._team_shape_history[team_id] = self._team_shape_history[team_id][-100:]

            return home_shape, away_shape
        except Exception as e:
            logger.debug(f"Team shape analysis failed: {e}")
            return None, None

    def _analyze_pressing(
        self,
        team_positions: Dict[int, Dict[int, Tuple[float, float]]],
        ball_position: Optional[Tuple[float, float]],
        possession_team: Optional[int]
    ) -> Dict[str, Any]:
        """
        Analyze pressing intensity.

        Note: The PressingAnalyzer uses event-based tracking (record_pass,
        record_defensive_action). Here we estimate pressing intensity based
        on player positions relative to the ball.
        """
        try:
            if ball_position is None or possession_team is None:
                return {}

            # Calculate pressing intensity based on distance to ball
            defending_team = 1 - possession_team
            defending_positions = list(team_positions.get(defending_team, {}).values())

            if not defending_positions:
                return {}

            # Calculate average distance of defending players to ball
            distances = []
            pressing_players = 0  # Players within pressing distance
            pressing_threshold = 150  # pixels - adjust based on your video resolution

            for pos in defending_positions:
                dist = ((pos[0] - ball_position[0])**2 + (pos[1] - ball_position[1])**2)**0.5
                distances.append(dist)
                if dist < pressing_threshold:
                    pressing_players += 1

            avg_distance = sum(distances) / len(distances) if distances else 0

            # Calculate intensity (inverse of distance, normalized)
            max_distance = 500  # Max expected distance in pixels
            intensity = max(0, 1 - (avg_distance / max_distance))

            # PPDA is tracked separately via events, but we can estimate
            # Lower distance = higher pressing = lower PPDA
            estimated_ppda = 5 + (avg_distance / max_distance) * 10  # Range ~5-15

            return {
                "intensity": intensity,
                "pressing_players": pressing_players,
                "avg_distance_to_ball": avg_distance,
                "ppda": estimated_ppda
            }
        except Exception as e:
            logger.debug(f"Pressing analysis failed: {e}")
            return {}

    def _analyze_space_creation(
        self,
        frame_num: int,
        team_positions: Dict[int, Dict[int, Tuple[float, float]]],
        ball_position: Tuple[float, float],
        timestamp: float,
        possession_team: Optional[int]
    ) -> List[Any]:
        """Analyze space creation and runs."""
        try:
            if possession_team is None:
                return []

            # Analyze for the team in possession
            events = self.space_analyzer.analyze_frame(
                frame=frame_num,
                timestamp=timestamp,
                team_positions=team_positions,
                ball_position=ball_position,
                team_in_possession=possession_team
            )
            return events or []
        except Exception as e:
            logger.debug(f"Space analysis failed: {e}")
            return []

    def _detect_counter_attack(
        self,
        frame_num: int,
        timestamp: float,
        ball_position: Optional[Tuple[float, float]],
        possession_team: Optional[int],
        possession_changed: bool
    ) -> Dict[str, Any]:
        """Detect counter attack situations."""
        try:
            if possession_team is None or ball_position is None:
                return {"detected": False}

            # Normalize ball position to 0-1 range (assuming typical video dimensions)
            # You may need to adjust this based on actual pitch detection
            ball_x = ball_position[0] / 1920  # Normalize assuming 1920 width
            ball_y = ball_position[1] / 1080

            # Estimate ball velocity (simplified - would be better with tracking)
            ball_velocity = 0.02  # Default moderate speed

            self.counter_detector.process_frame(
                frame=frame_num,
                timestamp=timestamp,
                ball_x=ball_x,
                ball_y=ball_y,
                team_with_possession=possession_team,
                possession_just_won=possession_changed,
                ball_velocity=ball_velocity
            )

            # Check if a counter attack was detected
            if self.counter_detector.counter_attacks:
                latest = self.counter_detector.counter_attacks[-1]
                if latest.end_frame == frame_num:
                    return {
                        "detected": True,
                        "team": latest.team_id,
                        "duration": latest.end_time - latest.start_time
                    }

            return {"detected": False}
        except Exception as e:
            logger.debug(f"Counter attack detection failed: {e}")
            return {"detected": False}

    def _update_fatigue_tracking(
        self,
        detections: FrameDetections,
        timestamp: float
    ):
        """Update fatigue tracking for all players."""
        try:
            for player in detections.players:
                if player.tracker_id:
                    self.fatigue_detector.update_player(
                        player_id=player.tracker_id,
                        timestamp=timestamp,
                        position=player.center
                    )
        except Exception as e:
            logger.debug(f"Fatigue tracking update failed: {e}")

    def record_shot(
        self,
        frame: int,
        timestamp: float,
        player_id: int,
        team_id: int,
        position: Tuple[float, float],
        shot_type: ShotType = ShotType.RIGHT_FOOT,
        situation: ShotSituation = ShotSituation.OPEN_PLAY,
        defender_count: int = 0
    ) -> Optional[float]:
        """
        Record a detected shot and calculate xG.

        Call this when shot detection identifies a shot.
        """
        if not self.xg_tracker:
            return None

        try:
            context = ShotContext(
                x=position[0],
                y=position[1],
                shot_type=shot_type,
                situation=situation,
                defender_count=defender_count
            )

            record = self.xg_tracker.record_shot(
                frame=frame,
                timestamp=timestamp,
                player_id=player_id,
                team_id=team_id,
                context=context
            )

            logger.info(
                f"Shot recorded: team={team_id}, player={player_id}, "
                f"xG={record.xg_value:.3f}, pos=({position[0]:.1f}, {position[1]:.1f})"
            )

            return record.xg_value
        except Exception as e:
            logger.warning(f"Failed to record shot: {e}")
            return None

    def _log_progress_summary(self):
        """Log periodic progress summary."""
        total_possession = sum(self._possession_frames.values())
        if total_possession > 0:
            home_poss = self._possession_frames[0] / total_possession * 100
            away_poss = self._possession_frames[1] / total_possession * 100
        else:
            home_poss = away_poss = 50.0

        logger.debug(
            f"Analysis progress: {self._frame_count} frames | "
            f"Possession: {home_poss:.0f}%-{away_poss:.0f}% | "
            f"Runs: {self._runs_detected} | "
            f"Counters: {sum(self._counter_attacks.values())}"
        )

    def get_summary(self) -> MatchAnalysisSummary:
        """
        Get summary statistics for all analysis.

        Call this after processing is complete.
        """
        summary = MatchAnalysisSummary(frames_analyzed=self._frame_count)

        # Possession
        total_possession = sum(self._possession_frames.values())
        if total_possession > 0:
            summary.possession_home = self._possession_frames[0] / total_possession * 100
            summary.possession_away = self._possession_frames[1] / total_possession * 100

        # Most common formations
        for team_id in [0, 1]:
            formations = self._formation_history[team_id]
            if formations:
                from collections import Counter
                most_common = Counter(formations).most_common(1)
                if most_common:
                    if team_id == 0:
                        summary.home_primary_formation = most_common[0][0]
                    else:
                        summary.away_primary_formation = most_common[0][0]

                # Count formation changes
                changes = sum(1 for i in range(1, len(formations))
                             if formations[i] != formations[i-1])
                if team_id == 0:
                    summary.formation_changes_home = changes
                else:
                    summary.formation_changes_away = changes

        # Average team shape
        for team_id in [0, 1]:
            shapes = self._team_shape_history[team_id]
            if shapes:
                avg_compactness = sum(s.get("compactness", 0) for s in shapes) / len(shapes)
                avg_width = sum(s.get("width", 0) for s in shapes) / len(shapes)
                if team_id == 0:
                    summary.home_avg_compactness = avg_compactness
                    summary.home_avg_width = avg_width
                else:
                    summary.away_avg_compactness = avg_compactness
                    summary.away_avg_width = avg_width

        # xG
        if self.xg_tracker:
            summary.home_xg = self.xg_tracker.get_team_xg(0)
            summary.away_xg = self.xg_tracker.get_team_xg(1)
            summary.home_shots = len(self.xg_tracker.shots.get(0, []))
            summary.away_shots = len(self.xg_tracker.shots.get(1, []))

        # Pressing (PPDA from pressing analyzer)
        if self.pressing_analyzer:
            ppda_stats = self.pressing_analyzer.get_match_ppda()
            summary.home_ppda = ppda_stats.get("home", 0.0)
            summary.away_ppda = ppda_stats.get("away", 0.0)

        # Counter attacks
        summary.home_counter_attacks = self._counter_attacks[0]
        summary.away_counter_attacks = self._counter_attacks[1]

        # Space creation
        summary.total_runs_detected = self._runs_detected

        return summary

    def get_formation_timeline(self, team_id: int) -> List[FormationType]:
        """Get formation changes over time for a team."""
        return self._formation_history.get(team_id, [])

    def get_xg_timeline(self, team_id: int) -> List[Tuple[float, float]]:
        """Get cumulative xG over time for a team."""
        if self.xg_tracker:
            return self.xg_tracker.get_xg_timeline(team_id)
        return []

    def get_fatigue_levels(self) -> Dict[int, float]:
        """Get current fatigue levels for all tracked players."""
        if self.fatigue_detector:
            return self.fatigue_detector.get_all_fatigue_levels()
        return {}

    def reset(self):
        """Reset all analysis state for a new video."""
        self._frame_count = 0
        self._last_possession_team = None
        self._possession_frames = {0: 0, 1: 0}
        self._formation_history = {0: [], 1: []}
        self._team_shape_history = {0: [], 1: []}
        self._counter_attacks = {0: 0, 1: 0}
        self._runs_detected = 0

        # Reset individual analyzers
        if self.formation_detector:
            self.formation_detector.reset()
        if self.pressing_analyzer:
            self.pressing_analyzer.reset()
        if self.xg_tracker:
            self.xg_tracker.reset()
        if self.fatigue_detector:
            self.fatigue_detector.reset()

        logger.info("AnalysisPipeline reset")
