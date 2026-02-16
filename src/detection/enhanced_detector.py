"""
Enhanced Soccer Detection
Fixes referee detection, team assignment consistency, and ball tracking
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict, deque
from loguru import logger

from .detector import (
    SoccerDetector, PlayerDetection, BallDetection, FrameDetections,
    TeamClassifier, SUPERVISION_AVAILABLE
)
from .pitch_detector import PitchDetector, PitchBoundary


# Common referee colors in RGB
REFEREE_COLORS = {
    'black': (30, 30, 30),
    'bright_yellow': (255, 255, 0),
    'bright_green': (0, 255, 100),
    'bright_pink': (255, 100, 180),
    'bright_orange': (255, 150, 0),
    'red': (255, 50, 50),
}


@dataclass
class TrackedPerson:
    """Persistent tracking info for a detected person.

    Tracks team and role assignments over time and stabilises them via
    majority voting.  Once stable, assignments are locked — a player
    classified as "player on team X" for 5+ frames won't be reclassified
    as a referee unless the tracker disappears and reappears.

    Goalkeepers are locked even more aggressively: once a person is stably
    classified as GK, they remain GK for the rest of the match (GKs rarely
    change in-game).
    """
    tracker_id: int
    team_history: List[int] = field(default_factory=list)
    role_history: List[str] = field(default_factory=list)  # "player", "referee", "goalkeeper"
    color_history: List[Tuple[int, int, int]] = field(default_factory=list)
    last_seen_frame: int = 0
    stable_team: Optional[int] = None
    stable_role: Optional[str] = None

    # Once True, the role is permanently locked (for GKs)
    role_locked: bool = False

    def add_observation(
        self,
        frame: int,
        team_id: Optional[int],
        role: str,
        color: Optional[Tuple[int, int, int]]
    ):
        """Add a new observation."""
        self.last_seen_frame = frame

        if team_id is not None:
            self.team_history.append(team_id)

        # If role is locked, ignore incoming role changes
        if self.role_locked:
            self.role_history.append(self.stable_role or role)
        elif role:
            self.role_history.append(role)

        if color:
            self.color_history.append(color)

        # Keep history limited
        max_history = 60
        self.team_history = self.team_history[-max_history:]
        self.role_history = self.role_history[-max_history:]
        self.color_history = self.color_history[-max_history:]

        # Update stable assignments
        self._update_stable_assignments()

    def _update_stable_assignments(self):
        """Update stable team/role based on history (5-frame threshold)."""

        # ---- Team stability ----
        if len(self.team_history) >= 5:
            team_counts = defaultdict(int)
            for t in self.team_history[-30:]:
                team_counts[t] += 1
            if team_counts:
                self.stable_team = max(team_counts.items(), key=lambda x: x[1])[0]

        # ---- Role stability ----
        if self.role_locked:
            return  # Already permanently locked

        if len(self.role_history) >= 5:
            role_counts = defaultdict(int)
            for r in self.role_history[-30:]:
                role_counts[r] += 1
            if role_counts:
                best_role = max(role_counts.items(), key=lambda x: x[1])[0]
                self.stable_role = best_role

                # Lock goalkeeper role permanently once stable
                if best_role == "goalkeeper":
                    self.role_locked = True

    def get_team(self) -> Optional[int]:
        """Get the most likely team assignment."""
        if self.stable_team is not None:
            return self.stable_team
        if self.team_history:
            return self.team_history[-1]
        return None

    def get_role(self) -> str:
        """Get the most likely role."""
        if self.stable_role is not None:
            return self.stable_role
        if self.role_history:
            return self.role_history[-1]
        return "player"


class EnhancedDetector(SoccerDetector):
    """
    Enhanced soccer detector with:
    - Improved referee detection
    - Consistent team assignment using tracking
    - Pitch boundary filtering
    - Better ball detection with Kalman filter
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pitch boundary detector
        self.pitch_detector = PitchDetector()

        # Tracking persistence
        self.tracked_persons: Dict[int, TrackedPerson] = {}

        # Auto-detected referee colors
        self.auto_referee_colors: List[Tuple[int, int, int]] = []

        # Ball tracking history (deque for O(1) append/trim)
        self.ball_history: deque = deque(maxlen=90)  # (frame, x, y) - 3 seconds at 30fps

        # Kalman filter for ball tracking
        self._ball_kalman = self._create_ball_kalman()
        self._kalman_initialized = False
        self._kalman_frames_without_measurement = 0
        self._kalman_max_predict_frames = 30  # Predict for up to 1 second (30fps)
        # Kalman prediction is accessed by parent's _select_best_ball()
        self._kalman_prediction: Optional[Tuple[float, float]] = None

        # Statistics
        self.stats = {
            'home_possession_frames': 0,
            'away_possession_frames': 0,
            'total_frames': 0,
            'shots_detected': 0,
            'passes_detected': 0,
        }

        # Initialize with common referee colors
        self._init_referee_colors()

    @staticmethod
    def _create_ball_kalman() -> cv2.KalmanFilter:
        """Create a Kalman filter for ball position tracking.

        State: [x, y, vx, vy]  (position + velocity)
        Measurement: [x, y]     (position only)
        """
        kf = cv2.KalmanFilter(4, 2)  # 4 state vars, 2 measurement vars

        # Transition matrix (constant velocity model):
        # x'  = x + vx*dt  (dt=1 frame)
        # y'  = y + vy*dt
        # vx' = vx
        # vy' = vy
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix: we only observe [x, y]
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise (how much we trust the model vs measurements)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5.0

        # Measurement noise (how noisy the YOLO detections are)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0

        # Initial state covariance (high uncertainty)
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        return kf

    def _init_referee_colors(self):
        """Initialize with common referee colors."""
        # Start with empty - let manual colors or learning populate this
        # DO NOT add generic dark colors like black/gray - they match shadows
        self.auto_referee_colors = []

    def set_referee_colors(self, colors: List[Tuple[int, int, int]]):
        """
        Override to clear auto-detected colors when manual colors are set.
        This ensures only the user-specified colors are used.
        """
        # Call parent method
        super().set_referee_colors(colors)
        # Clear auto-detected colors - use only manual colors
        self.auto_referee_colors = []
        logger.info(f"Cleared auto-detected referee colors, using manual: {colors}")

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        fps: float = 30.0,
        detect_pitch: bool = True,
        track_objects: bool = True,
        confidence_threshold: Optional[float] = None
    ) -> FrameDetections:
        """Enhanced detection with pitch filtering, persistent tracking, and Kalman ball filter."""

        # First, detect pitch boundary
        if detect_pitch:
            self.pitch_detector.detect(frame, frame_number)

        # ---- Kalman prediction BEFORE base detection ----
        # This makes the prediction available to _select_best_ball() in the parent class.
        if self._kalman_initialized:
            predicted = self._ball_kalman.predict()
            pred_x, pred_y = float(predicted[0]), float(predicted[1])
            self._kalman_prediction = (pred_x, pred_y)
        else:
            self._kalman_prediction = None

        # Run base detection (uses _kalman_prediction for best-ball selection)
        results = super().detect_frame(
            frame, frame_number, fps, False, track_objects, confidence_threshold
        )

        # Filter out-of-bounds detections
        results = self._filter_out_of_bounds(results, frame)

        # Improve referee detection
        results = self._enhance_referee_detection(frame, results)

        # Apply persistent team assignments
        results = self._apply_persistent_tracking(frame_number, results)

        # ---- Kalman filter update ----
        if results.ball is not None:
            cx, cy = results.ball.center
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])

            if not self._kalman_initialized:
                # First detection — initialize Kalman state
                self._ball_kalman.statePost = np.array(
                    [[np.float32(cx)], [np.float32(cy)], [0.0], [0.0]],
                    dtype=np.float32
                )
                self._kalman_initialized = True
            else:
                # Correct with measurement
                self._ball_kalman.correct(measurement)

            self._kalman_frames_without_measurement = 0
            self.ball_history.append((frame_number, cx, cy))
        else:
            # No ball detected — try Kalman prediction as fallback
            self._kalman_frames_without_measurement += 1

            if (self._kalman_initialized and
                    self._kalman_frames_without_measurement <= self._kalman_max_predict_frames):
                # Use Kalman prediction (already computed above)
                if self._kalman_prediction is not None:
                    pred_x, pred_y = self._kalman_prediction
                    # Clamp to frame bounds
                    h, w = frame.shape[:2]
                    pred_x = max(0, min(pred_x, w))
                    pred_y = max(0, min(pred_y, h))

                    size = 12
                    results.ball = BallDetection(
                        bbox=(pred_x - size, pred_y - size,
                              pred_x + size, pred_y + size),
                        confidence=max(0.1, 0.4 - 0.01 * self._kalman_frames_without_measurement),
                        class_id=32,
                        class_name="ball_predicted",
                        tracker_id=None
                    )
                    self.ball_history.append((frame_number, pred_x, pred_y))
            elif self._kalman_frames_without_measurement > self._kalman_max_predict_frames:
                # Lost the ball for too long — reset Kalman
                self._kalman_initialized = False
                self._ball_kalman = self._create_ball_kalman()

            # Last resort: color-based fallback
            if results.ball is None:
                results.ball = self._detect_ball_by_color(frame)

        # Update statistics
        self.stats['total_frames'] += 1

        return results

    def _filter_out_of_bounds(self, results: FrameDetections, frame: np.ndarray = None) -> FrameDetections:
        """Filter detections that are outside the pitch boundary."""
        boundary = self.pitch_detector._cached_boundary

        # Get frame dimensions from actual frame if available
        if frame is not None:
            frame_height, frame_width = frame.shape[:2]
        else:
            # Use typical HD dimensions as fallback
            frame_height = 1080
            frame_width = 1920

        # Additional heuristic: filter based on frame position
        # Sideline people are typically at very top/bottom or far left/right edges
        def is_likely_sideline_or_bench(bbox):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            height = y2 - y1

            # Check if in technical area (coaches box)
            if boundary and boundary.is_in_technical_area(cx, y2):
                logger.debug(f"Filtered technical area (coaches box): bbox={bbox}")
                return True

            # People at the very top of frame are likely in stands/sideline
            if y1 < frame_height * 0.08:
                logger.debug(f"Filtered sideline (top): bbox={bbox}")
                return True

            # People at extreme left/right edges (ball boys, coaches)
            if cx < frame_width * 0.03 or cx > frame_width * 0.97:
                logger.debug(f"Filtered sideline (edge): bbox={bbox}")
                return True

            # Very small detections at edges are likely distant sideline people
            if height < 40 and (cx < frame_width * 0.1 or cx > frame_width * 0.9):
                logger.debug(f"Filtered sideline (small edge): bbox={bbox}")
                return True

            # People just outside the pitch boundary in center region are likely bench/coaches
            if boundary:
                # Check if just outside pitch but not a goalkeeper area
                if not boundary.contains_point(cx, cy, margin=0):
                    # Outside the pitch
                    if not boundary.is_in_goal_area(cx, cy):
                        # Not in goal area - likely sideline personnel
                        # Check if they're close to the sideline (within margin)
                        if boundary.contains_point(cx, cy, margin=80):
                            logger.debug(f"Filtered bench area: bbox={bbox}")
                            return True

            return False

        # Filter players
        filtered_players = []
        for player in results.players:
            # First check sideline heuristics
            if is_likely_sideline_or_bench(player.bbox):
                continue
            # Then check pitch boundary
            if boundary is None or boundary.contains_bbox(player.bbox):
                filtered_players.append(player)
            else:
                logger.debug(f"Filtered out-of-bounds player: {player.bbox}")

        results.players = filtered_players

        # Filter goalkeepers (they should be on pitch too)
        filtered_gks = []
        for gk in results.goalkeepers:
            if is_likely_sideline_or_bench(gk.bbox):
                continue
            if boundary is None or boundary.contains_bbox(gk.bbox):
                filtered_gks.append(gk)

        results.goalkeepers = filtered_gks

        # Referees can be on or slightly off pitch
        filtered_refs = []
        for ref in results.referees:
            if is_likely_sideline_or_bench(ref.bbox):
                continue
            if boundary is None:
                filtered_refs.append(ref)
            else:
                # Allow referees with some margin
                x1, y1, x2, y2 = ref.bbox
                cx = (x1 + x2) / 2
                if boundary.contains_point(cx, y2, margin=50):  # 50px margin
                    filtered_refs.append(ref)

        results.referees = filtered_refs

        return results

    def _enhance_referee_detection(
        self,
        frame: np.ndarray,
        results: FrameDetections
    ) -> FrameDetections:
        """Improve referee detection using color analysis."""
        # Combine auto-detected and manually set referee colors
        all_ref_colors = self.auto_referee_colors + self.referee_colors

        # Check each player to see if they might be a referee
        players_to_remove = []
        for i, player in enumerate(results.players):
            if player.dominant_color is None:
                continue

            is_referee = self._is_likely_referee(player.dominant_color, all_ref_colors)

            if is_referee:
                # Convert to referee
                ref = PlayerDetection(
                    bbox=player.bbox,
                    confidence=player.confidence,
                    class_id=player.class_id,
                    class_name="referee",
                    tracker_id=player.tracker_id,
                    is_referee=True,
                    dominant_color=player.dominant_color
                )
                results.referees.append(ref)
                players_to_remove.append(i)
                logger.debug(f"Reclassified player as referee: color={player.dominant_color}")

        # Remove reclassified players
        for i in sorted(players_to_remove, reverse=True):
            results.players.pop(i)

        return results

    def _is_likely_referee(
        self,
        color: Tuple[int, int, int],
        ref_colors: List[Tuple[int, int, int]]
    ) -> bool:
        """Check if a color matches referee patterns."""
        # Only match if we have specific referee colors set
        if not ref_colors:
            return False

        r, g, b = color

        # Check against known referee colors with TIGHT LAB threshold
        # Only match if very close to a known referee color
        for ref_color in ref_colors:
            distance = self._color_distance(color, ref_color)
            if distance < 35:  # Tight LAB match - must be close to actual ref color
                return True

        # DO NOT use heuristics for dark colors - they match shadows/skin tones
        # Only use explicit referee colors set by user or detected from actual refs

        return False

    def _apply_persistent_tracking(
        self,
        frame_number: int,
        results: FrameDetections
    ) -> FrameDetections:
        """
        Apply persistent team/role assignments based on tracking history.

        Key behaviours:
        - Once a tracker_id is stably classified as "player" on team X,
          it stays that way even if a single frame's color is ambiguous.
        - If a player was classified as referee but their stable role is
          "player", move them back to the players list.
        - Goalkeepers are locked permanently once stable (5+ frames).
        """

        # ---------- Process players ----------
        for player in results.players:
            if player.tracker_id is None:
                continue

            tid = player.tracker_id

            if tid not in self.tracked_persons:
                self.tracked_persons[tid] = TrackedPerson(tracker_id=tid)

            tracked = self.tracked_persons[tid]

            # Add current observation
            tracked.add_observation(
                frame=frame_number,
                team_id=player.team_id,
                role="player",
                color=player.dominant_color
            )

            # Apply stable team assignment
            stable_team = tracked.get_team()
            if stable_team is not None:
                player.team_id = stable_team

        # ---------- Process goalkeepers ----------
        for gk in results.goalkeepers:
            if gk.tracker_id is None:
                continue

            tid = gk.tracker_id

            if tid not in self.tracked_persons:
                self.tracked_persons[tid] = TrackedPerson(tracker_id=tid)

            tracked = self.tracked_persons[tid]
            tracked.add_observation(
                frame=frame_number,
                team_id=gk.team_id,
                role="goalkeeper",
                color=gk.dominant_color
            )

        # ---------- Process referees ----------
        refs_to_move_back = []
        for i, ref in enumerate(results.referees):
            if ref.tracker_id is None:
                continue

            tid = ref.tracker_id

            if tid not in self.tracked_persons:
                self.tracked_persons[tid] = TrackedPerson(tracker_id=tid)

            tracked = self.tracked_persons[tid]
            tracked.add_observation(
                frame=frame_number,
                team_id=None,
                role="referee",
                color=ref.dominant_color
            )

            # If this tracker's stable role is actually "player", move it back
            if tracked.stable_role == "player":
                logger.debug(f"Tracker {tid} stable role is 'player' — "
                            f"moving back from referees list")
                ref.is_referee = False
                ref.class_name = "player"
                # Give it the stable team
                stable_team = tracked.get_team()
                if stable_team is not None:
                    ref.team_id = stable_team
                refs_to_move_back.append(i)

        # Move mis-classified referees back to players
        for i in sorted(refs_to_move_back, reverse=True):
            moved_player = results.referees.pop(i)
            results.players.append(moved_player)

        # ---------- Apply stable role overrides for players ----------
        players_to_move = []
        for i, player in enumerate(results.players):
            if player.tracker_id is None:
                continue
            tid = player.tracker_id
            if tid in self.tracked_persons:
                tracked = self.tracked_persons[tid]
                # If stably a goalkeeper, move to goalkeepers list
                if tracked.stable_role == "goalkeeper":
                    player.is_goalkeeper = True
                    player.class_name = "goalkeeper"
                    players_to_move.append(i)

        for i in sorted(players_to_move, reverse=True):
            moved_gk = results.players.pop(i)
            results.goalkeepers.append(moved_gk)

        # ---------- Clean up old tracked persons ----------
        stale_threshold = frame_number - 300  # 10 seconds at 30fps
        stale_ids = [
            tid for tid, tracked in self.tracked_persons.items()
            if tracked.last_seen_frame < stale_threshold
        ]
        for tid in stale_ids:
            del self.tracked_persons[tid]

        return results

    def _detect_ball_fallback(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> Optional[BallDetection]:
        """
        Fallback ball detection using motion and color analysis.
        Used when YOLO fails to detect the ball.
        """
        # Method 1: Look for small white/orange circular objects
        ball = self._detect_ball_by_color(frame)
        if ball is not None:
            return ball

        # Method 2: Interpolate from recent history
        if len(self.ball_history) >= 2:
            ball = self._interpolate_ball_position(frame_number)
            if ball is not None:
                return ball

        return None

    def _detect_ball_by_color(self, frame: np.ndarray) -> Optional[BallDetection]:
        """Detect ball using color and shape analysis."""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # White ball detection
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            # Orange ball detection (some leagues use orange balls)
            lower_orange = np.array([5, 150, 150])
            upper_orange = np.array([20, 255, 255])
            orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

            # Combine masks
            ball_mask = cv2.bitwise_or(white_mask, orange_mask)

            # Clean up
            kernel = np.ones((3, 3), np.uint8)
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Look for ball-sized circular contours
            for contour in contours:
                area = cv2.contourArea(contour)

                # Ball should be relatively small
                if 50 < area < 2000:
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)

                        if circularity > 0.6:  # Reasonably circular
                            x, y, w, h = cv2.boundingRect(contour)

                            # Check aspect ratio
                            aspect = w / h if h > 0 else 0
                            if 0.7 < aspect < 1.4:
                                # Found a ball candidate
                                cx = x + w / 2
                                cy = y + h / 2

                                # Verify it's on the pitch
                                if self.pitch_detector._cached_boundary:
                                    if not self.pitch_detector._cached_boundary.contains_point(cx, cy):
                                        continue

                                return BallDetection(
                                    bbox=(x, y, x + w, y + h),
                                    confidence=0.5,  # Lower confidence for color-based detection
                                    class_id=32,
                                    class_name="ball",
                                    tracker_id=None
                                )

        except Exception as e:
            logger.debug(f"Color-based ball detection failed: {e}")

        return None

    def _interpolate_ball_position(self, frame_number: int) -> Optional[BallDetection]:
        """Interpolate ball position from recent history."""
        if len(self.ball_history) < 2:
            return None

        # Get last few known positions (convert deque slice to list)
        history_list = list(self.ball_history)
        recent = history_list[-5:] if len(history_list) >= 5 else history_list

        # Check if positions are recent enough
        last_frame, last_x, last_y = recent[-1]
        if frame_number - last_frame > 15:  # More than 0.5 seconds old
            return None

        # Simple linear extrapolation
        if len(recent) >= 2:
            prev_frame, prev_x, prev_y = recent[-2]
            frame_diff = last_frame - prev_frame
            if frame_diff > 0:
                vx = (last_x - prev_x) / frame_diff
                vy = (last_y - prev_y) / frame_diff

                frames_ahead = frame_number - last_frame
                pred_x = last_x + vx * frames_ahead
                pred_y = last_y + vy * frames_ahead

                # Create interpolated detection
                size = 15
                return BallDetection(
                    bbox=(pred_x - size, pred_y - size, pred_x + size, pred_y + size),
                    confidence=0.3,  # Low confidence for interpolated
                    class_id=32,
                    class_name="ball_interpolated",
                    tracker_id=None
                )

        return None

    def learn_referee_colors(self, frame: np.ndarray, results: FrameDetections):
        """
        Learn referee colors from current detections.
        Call this when you know referees are visible.
        """
        for ref in results.referees:
            if ref.dominant_color is not None:
                # Add to auto-detected colors if not already present
                is_new = True
                for existing in self.auto_referee_colors:
                    if self._color_distance(ref.dominant_color, existing) < 30:
                        is_new = False
                        break

                if is_new:
                    self.auto_referee_colors.append(ref.dominant_color)
                    logger.info(f"Learned referee color: {ref.dominant_color}")

    def learn_team_colors_from_frame(
        self,
        frame: np.ndarray,
        results: FrameDetections
    ) -> Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
        """
        Learn team colors from current frame's player detections.

        Returns (home_color, away_color) or (None, None) if not enough data.
        """
        player_colors = [
            p.dominant_color for p in results.players
            if p.dominant_color is not None and not p.is_referee
        ]

        if len(player_colors) < 6:  # Need enough players
            return None, None

        # Filter out referee-like colors
        filtered_colors = []
        for color in player_colors:
            if not self._is_likely_referee(color, self.auto_referee_colors):
                filtered_colors.append(color)

        if len(filtered_colors) < 4:
            return None, None

        # Cluster into 2 teams (sklearn version compatible)
        from sklearn.cluster import KMeans
        import sklearn

        color_array = np.array(filtered_colors)

        sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        if sklearn_version >= (1, 2):
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42, algorithm='lloyd')
        else:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(color_array)

        team_colors = kmeans.cluster_centers_.astype(int)

        return tuple(team_colors[0]), tuple(team_colors[1])

    def get_possession_stats(self) -> Dict[str, float]:
        """Get current possession statistics."""
        total = self.stats['home_possession_frames'] + self.stats['away_possession_frames']
        if total == 0:
            return {'home': 50.0, 'away': 50.0}

        home_pct = (self.stats['home_possession_frames'] / total) * 100
        away_pct = (self.stats['away_possession_frames'] / total) * 100

        return {
            'home': round(home_pct, 1),
            'away': round(away_pct, 1)
        }

    def reset(self):
        """Reset all tracking state."""
        super().reset_tracker()
        self.tracked_persons.clear()
        self.ball_history.clear()
        self.pitch_detector.reset()

        # Reset Kalman filter
        self._ball_kalman = self._create_ball_kalman()
        self._kalman_initialized = False
        self._kalman_frames_without_measurement = 0
        self._kalman_prediction = None

        self.stats = {
            'home_possession_frames': 0,
            'away_possession_frames': 0,
            'total_frames': 0,
            'shots_detected': 0,
            'passes_detected': 0,
        }
