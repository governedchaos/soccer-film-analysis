"""
Soccer Film Analysis - Detection Pipeline
Uses local YOLO models for player, ball, and pitch detection
No API key required - runs entirely offline
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict, deque
from pathlib import Path
from loguru import logger

# Ultralytics YOLO - Local models
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")

# Supervision for tracking and utilities
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    logger.warning("Supervision not available. Install with: pip install supervision")

# Import sports library if available (optional - for pitch visualization)
try:
    from sports.configs.soccer import SoccerFieldConfiguration
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    SPORTS_AVAILABLE = True
except ImportError:
    SPORTS_AVAILABLE = False
    logger.debug("Roboflow Sports not available (optional). Install with: pip install git+https://github.com/roboflow/sports.git")

from config import settings


# ============================================
# Data Classes for Detection Results
# ============================================

@dataclass
class Detection:
    """Single object detection result"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    tracker_id: Optional[int] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class PlayerDetection(Detection):
    """Player-specific detection with additional attributes"""
    team_id: Optional[int] = None  # 0=home, 1=away, None=unknown
    jersey_number: Optional[int] = None
    is_goalkeeper: bool = False
    is_referee: bool = False
    
    # Color information for team classification
    dominant_color: Optional[Tuple[int, int, int]] = None  # RGB


@dataclass
class BallDetection(Detection):
    """Ball detection with velocity tracking"""
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy) pixels per frame


@dataclass
class PitchKeypoint:
    """Single pitch keypoint detection"""
    point_id: int
    x: float
    y: float
    confidence: float


@dataclass
class FrameDetections:
    """All detections for a single frame"""
    frame_number: int
    timestamp_seconds: float
    
    players: List[PlayerDetection] = field(default_factory=list)
    ball: Optional[BallDetection] = None
    referees: List[PlayerDetection] = field(default_factory=list)
    goalkeepers: List[PlayerDetection] = field(default_factory=list)
    pitch_keypoints: List[PitchKeypoint] = field(default_factory=list)
    
    # Raw supervision detections for advanced processing
    raw_detections: Optional[Any] = None

    # Possession team (0=home, 1=away, None=contested/unknown)
    possession_team: Optional[int] = None


# ============================================
# Detection Pipeline
# ============================================

class SoccerDetector:
    """
    Main detection class using local YOLO models for soccer analysis.
    No API key required - runs entirely offline.

    Uses:
    - YOLOv8 for person detection (players, goalkeepers, referees)
    - YOLOv8 for ball detection
    - Color-based classification for team assignment
    - Position heuristics for role classification (goalkeeper, referee)
    """

    # YOLO COCO class IDs we care about
    COCO_PERSON = 0
    COCO_SPORTS_BALL = 32

    # Available YOLO model sizes (downloaded automatically on first use)
    MODEL_SIZES = {
        "nano": "yolov8n.pt",      # Fastest, least accurate (3.2M params)
        "small": "yolov8s.pt",     # Good balance (11.2M params)
        "medium": "yolov8m.pt",    # Better accuracy (25.9M params)
        "large": "yolov8l.pt",     # High accuracy (43.7M params)
        "xlarge": "yolov8x.pt",    # Best accuracy (68.2M params)
    }

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        models_dir: Optional[Path] = None
    ):
        """
        Initialize the detector with local YOLO models.

        Args:
            model_size: YOLO model size - "nano", "small", "medium", "large", "xlarge"
                       If None, auto-selects based on hardware (nano for CPU, configured size for GPU)
            device: Compute device (cuda, mps, cpu) - auto-detected if None
            models_dir: Directory to store downloaded models
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not installed. Run: pip install ultralytics")

        self.device = device or settings.get_device()
        # Auto-select model size based on hardware if not explicitly specified
        self.model_size = model_size or settings.get_effective_yolo_model_size()
        self.models_dir = models_dir or settings.get_models_dir()

        # Get model filename
        if model_size in self.MODEL_SIZES:
            self.model_file = self.MODEL_SIZES[model_size]
        else:
            # Allow custom model path
            self.model_file = model_size

        # Models (lazy loaded)
        self._model = None

        # Tracking
        self._tracker = None

        # Team classifier
        self._team_classifier = None

        # Referee/goalkeeper color hints (can be set manually)
        self.referee_colors: List[Tuple[int, int, int]] = []  # RGB colors
        self.goalkeeper_colors: Dict[int, Tuple[int, int, int]] = {}  # team_id -> RGB

        # Field boundary for position-based classification
        self.field_bounds: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2

        # Multi-observation color cache for tracked players.
        # Stores the last N color observations per tracker_id and returns the
        # median color, which smooths out frame-to-frame lighting variations.
        # tracker_id -> deque of RGB tuples (max 5 observations)
        self._color_cache: OrderedDict[int, deque] = OrderedDict()
        self._color_cache_max_size = 30
        self._color_cache_obs_count = 5  # Observations to store per tracker
        self._color_cache_hits = 0
        self._color_cache_misses = 0

        # Pitch configuration
        self.pitch_config = None
        if SPORTS_AVAILABLE:
            self.pitch_config = SoccerFieldConfiguration()

        logger.info(f"SoccerDetector initialized (model: {self.model_file}, device: {self.device})")

    @property
    def model(self):
        """Lazy load YOLO model"""
        if self._model is None:
            model_path = self.models_dir / self.model_file

            # If model doesn't exist locally, YOLO will download it automatically
            if model_path.exists():
                logger.info(f"Loading local YOLO model: {model_path}")
                self._model = YOLO(str(model_path))
            else:
                logger.info(f"Downloading YOLO model: {self.model_file}")
                self._model = YOLO(self.model_file)

                # Save to models directory for future use
                try:
                    import shutil
                    # The model is cached in ultralytics cache, copy it
                    cache_path = Path.home() / ".cache" / "ultralytics" / self.model_file
                    if cache_path.exists():
                        shutil.copy(cache_path, model_path)
                        logger.info(f"Model saved to: {model_path}")
                except Exception as e:
                    logger.debug(f"Could not copy model to models dir: {e}")

            # Set device
            self._model.to(self.device)

        return self._model

    @property
    def tracker(self):
        """Get ByteTrack tracker for object tracking"""
        if not SUPERVISION_AVAILABLE:
            return None

        if self._tracker is None:
            self._tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30
            )
        return self._tracker
    
    def _downscale_for_inference(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Downscale frame for faster YOLO inference if inference_resolution is set.

        Args:
            frame: Original BGR frame

        Returns:
            (possibly_resized_frame, scale_factor)
            scale_factor is original_width / resized_width (>= 1.0)
        """
        inference_res = settings.get_effective_inference_resolution()
        if inference_res <= 0 or frame.shape[1] <= inference_res:
            return frame, 1.0

        original_w = frame.shape[1]
        scale = inference_res / original_w
        new_w = inference_res
        new_h = int(frame.shape[0] * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, original_w / new_w

    def _scale_detections_back(
        self,
        detections: Any,
        scale_factor: float
    ) -> Any:
        """
        Scale detection bounding boxes back to original resolution.

        Args:
            detections: sv.Detections object
            scale_factor: Factor to multiply coordinates by (original / resized)

        Returns:
            Detections with scaled bounding boxes
        """
        if scale_factor == 1.0 or detections is None:
            return detections
        if hasattr(detections, 'xyxy') and len(detections) > 0:
            detections.xyxy = detections.xyxy * scale_factor
        return detections

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        fps: float = 30.0,
        detect_pitch: bool = False,  # Pitch detection requires separate model
        track_objects: bool = True,
        confidence_threshold: Optional[float] = None
    ) -> FrameDetections:
        """
        Run detection on a single frame using local YOLO models.

        Args:
            frame: BGR image as numpy array
            frame_number: Current frame number
            fps: Video frames per second
            detect_pitch: Whether to also detect pitch keypoints (requires sports lib)
            track_objects: Whether to apply tracking for persistent IDs
            confidence_threshold: Override default confidence threshold

        Returns:
            FrameDetections object with all detected objects
        """
        player_conf = confidence_threshold or settings.player_confidence_threshold
        ball_conf = settings.ball_confidence_threshold
        # Use the lower threshold to catch both, then filter by class
        min_conf = min(player_conf, ball_conf, 0.15)  # Ball needs low confidence
        timestamp = frame_number / fps

        # Initialize results
        results = FrameDetections(
            frame_number=frame_number,
            timestamp_seconds=timestamp
        )

        # Downscale for faster inference (bboxes scaled back to original res)
        inference_frame, scale_factor = self._downscale_for_inference(frame)

        # Run YOLO detection
        try:
            # Run inference with lower threshold to catch balls
            yolo_results = self.model.predict(
                inference_frame,
                conf=min_conf,
                verbose=False,
                classes=[self.COCO_PERSON, self.COCO_SPORTS_BALL]  # Only detect persons and sports balls
            )[0]

            # Convert to supervision Detections for tracking
            if SUPERVISION_AVAILABLE:
                detections = sv.Detections.from_ultralytics(yolo_results)

                # Scale bboxes back to original resolution before tracking
                detections = self._scale_detections_back(detections, scale_factor)

                # Apply tracking if enabled
                if track_objects and len(detections) > 0 and self.tracker is not None:
                    detections = self.tracker.update_with_detections(detections)

                # Store raw detections
                results.raw_detections = detections

                # Parse detections into typed objects (uses original-res frame for color)
                self._parse_yolo_detections(frame, detections, results)
            else:
                # Fallback without supervision - parse YOLO results directly
                # Scale bboxes back manually
                if scale_factor != 1.0 and yolo_results.boxes is not None:
                    yolo_results.boxes.xyxy *= scale_factor
                self._parse_yolo_results_direct(frame, yolo_results, results)

        except Exception as e:
            logger.error(f"Detection failed on frame {frame_number}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return results

    def _parse_yolo_detections(
        self,
        frame: np.ndarray,
        detections: Any,  # sv.Detections
        results: FrameDetections
    ):
        """Parse supervision Detections from YOLO into typed detection objects"""

        for i in range(len(detections)):
            bbox = tuple(detections.xyxy[i])
            confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            class_id = int(detections.class_id[i]) if detections.class_id is not None else 0
            tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None

            # Extract dominant color from detection region (for persons)
            # Multi-observation cache: store last N colors per tracker_id and
            # return the median to smooth out lighting variations.
            dominant_color = None
            if class_id == self.COCO_PERSON:
                # Always extract a fresh color observation
                fresh_color = self._extract_dominant_color(frame, bbox)

                if tracker_id is not None and tracker_id in self._color_cache:
                    # Known tracker - add new observation and return median
                    self._color_cache.move_to_end(tracker_id)
                    if fresh_color is not None:
                        self._color_cache[tracker_id].append(fresh_color)
                    # Compute median color from stored observations
                    obs = self._color_cache[tracker_id]
                    if len(obs) > 0:
                        arr = np.array(list(obs))
                        dominant_color = tuple(np.median(arr, axis=0).astype(int))
                    else:
                        dominant_color = fresh_color
                    self._color_cache_hits += 1
                else:
                    # New tracker or no tracking - start fresh observation list
                    dominant_color = fresh_color
                    self._color_cache_misses += 1
                    if tracker_id is not None and fresh_color is not None:
                        # LRU eviction - remove oldest if at capacity
                        if len(self._color_cache) >= self._color_cache_max_size:
                            self._color_cache.popitem(last=False)
                        obs_deque = deque(maxlen=self._color_cache_obs_count)
                        obs_deque.append(fresh_color)
                        self._color_cache[tracker_id] = obs_deque

            # Handle ball detection — collect all candidates, pick best later
            if class_id == self.COCO_SPORTS_BALL:
                # Size validation: reject implausible detections
                ball_w = bbox[2] - bbox[0]
                ball_h = bbox[3] - bbox[1]
                ball_diameter = max(ball_w, ball_h)
                frame_h = frame.shape[0]

                # At 1080p from press box, the ball is ~6-30px diameter.
                # Scale thresholds proportionally for other resolutions.
                min_ball = max(3, frame_h * 0.004)   # ~4px at 1080p
                max_ball = frame_h * 0.04             # ~43px at 1080p

                if ball_diameter < min_ball or ball_diameter > max_ball:
                    logger.debug(f"Ball candidate rejected (size {ball_diameter:.0f}px "
                                f"outside [{min_ball:.0f}, {max_ball:.0f}])")
                else:
                    logger.info(f"BALL CANDIDATE: conf={confidence:.3f}, "
                               f"bbox={bbox}, size={ball_diameter:.0f}px")
                    ball = BallDetection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        class_name="ball",
                        tracker_id=tracker_id
                    )
                    # Collect candidates — best-ball selection in _select_best_ball()
                    if not hasattr(results, '_ball_candidates'):
                        results._ball_candidates = []
                    results._ball_candidates.append(ball)

            # Handle person detection - classify as player, goalkeeper, or referee
            elif class_id == self.COCO_PERSON:
                role = self._classify_person_role(frame, bbox, dominant_color)

                if role == "referee":
                    ref = PlayerDetection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        class_name="referee",
                        tracker_id=tracker_id,
                        is_referee=True,
                        dominant_color=dominant_color
                    )
                    results.referees.append(ref)

                elif role == "goalkeeper":
                    gk = PlayerDetection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        class_name="goalkeeper",
                        tracker_id=tracker_id,
                        is_goalkeeper=True,
                        dominant_color=dominant_color
                    )
                    results.goalkeepers.append(gk)

                else:  # player
                    player = PlayerDetection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        class_name="player",
                        tracker_id=tracker_id,
                        dominant_color=dominant_color
                    )
                    results.players.append(player)

        # ---- Best-ball selection from candidates ----
        candidates = getattr(results, '_ball_candidates', [])
        if candidates:
            results.ball = self._select_best_ball(candidates)
            logger.debug(f"Selected best ball from {len(candidates)} candidates")

    def _select_best_ball(self, candidates: List[BallDetection]) -> BallDetection:
        """
        Select the best ball detection from multiple candidates.

        Priority:
        1. If a Kalman prediction is available (set by EnhancedDetector),
           pick the candidate closest to the prediction.
        2. Otherwise, pick the highest-confidence candidate.
        """
        if len(candidates) == 1:
            return candidates[0]

        # Check for Kalman prediction (set by EnhancedDetector subclass)
        kalman_pred = getattr(self, '_kalman_prediction', None)
        if kalman_pred is not None:
            pred_x, pred_y = kalman_pred
            best = min(candidates, key=lambda b: (
                (b.center[0] - pred_x) ** 2 + (b.center[1] - pred_y) ** 2
            ))
            logger.debug(f"Best ball selected by Kalman proximity: "
                        f"pred=({pred_x:.0f},{pred_y:.0f}), "
                        f"selected=({best.center[0]:.0f},{best.center[1]:.0f})")
            return best

        # Fallback: highest confidence
        return max(candidates, key=lambda b: b.confidence)

    def _parse_yolo_results_direct(
        self,
        frame: np.ndarray,
        yolo_results: Any,
        results: FrameDetections
    ):
        """Parse YOLO results directly without supervision (fallback)"""
        if yolo_results.boxes is None:
            return

        boxes = yolo_results.boxes
        for i in range(len(boxes)):
            bbox = tuple(boxes.xyxy[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())

            # Extract dominant color for persons
            dominant_color = None
            if class_id == self.COCO_PERSON:
                dominant_color = self._extract_dominant_color(frame, bbox)

            # Handle ball
            if class_id == self.COCO_SPORTS_BALL:
                ball = BallDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name="ball",
                    tracker_id=None
                )
                results.ball = ball

            # Handle person
            elif class_id == self.COCO_PERSON:
                role = self._classify_person_role(frame, bbox, dominant_color)
                player = PlayerDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=role,
                    tracker_id=None,
                    is_referee=(role == "referee"),
                    is_goalkeeper=(role == "goalkeeper"),
                    dominant_color=dominant_color
                )
                if role == "referee":
                    results.referees.append(player)
                elif role == "goalkeeper":
                    results.goalkeepers.append(player)
                else:
                    results.players.append(player)

    def _classify_person_role(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        dominant_color: Optional[Tuple[int, int, int]]
    ) -> str:
        """
        Classify a detected person as player, goalkeeper, or referee.

        Uses color matching against known referee/goalkeeper colors and
        position heuristics.

        Args:
            frame: The video frame
            bbox: Bounding box (x1, y1, x2, y2)
            dominant_color: Extracted jersey color (RGB)

        Returns:
            "player", "goalkeeper", or "referee"
        """
        if dominant_color is None:
            return "player"

        # LAB distance thresholds (perceptually calibrated):
        #   ~25 = very close match (same jersey, lighting variation)
        #   ~40 = reasonable match (kit under variable lighting)
        #   ~55 = loose match (very different lighting or slight kit variation)
        GK_THRESHOLD = 40        # Matches goalkeeper kit colors
        GK_TEAM_MIN_DIST = 35    # GK must be at least this far from team colors
        REFEREE_THRESHOLD = 40   # Matches referee kit colors
        REFEREE_VS_TEAM = 10     # Referee must be this much FURTHER from team than team match

        # ---- Step 1: Check goalkeeper colors FIRST ----
        # GKs wear unique kits — if color matches a GK kit and does NOT match
        # the corresponding team kit, classify as goalkeeper immediately.
        for team_id, gk_color in self.goalkeeper_colors.items():
            gk_distance = self._color_distance(dominant_color, gk_color)
            logger.debug(f"GK color check: team={team_id}, color={dominant_color}, "
                        f"gk_color={gk_color}, distance={gk_distance:.1f}")
            if gk_distance < GK_THRESHOLD:
                # Verify it doesn't also match a team color (avoid false positives)
                matches_team = False
                if hasattr(self, '_team_classifier') and self._team_classifier and self._team_classifier.team_colors is not None:
                    for team_color in self._team_classifier.team_colors:
                        team_dist = self._color_distance(dominant_color, tuple(team_color))
                        if team_dist < gk_distance:
                            matches_team = True
                            break

                if not matches_team:
                    logger.info(f"Classified as GOALKEEPER: team={team_id}, "
                               f"color={dominant_color}, distance={gk_distance:.1f}")
                    return "goalkeeper"

        # ---- Step 2: Compute team distances (used by referee check) ----
        min_team_distance = float('inf')
        if hasattr(self, '_team_classifier') and self._team_classifier and self._team_classifier.team_colors is not None:
            for team_color in self._team_classifier.team_colors:
                dist = self._color_distance(dominant_color, tuple(team_color))
                min_team_distance = min(min_team_distance, dist)

        # ---- Step 3: Check referee colors AFTER team distance is known ----
        # A person is only classified as referee if their color is closer to a
        # referee color than to EITHER team color. This prevents shadow/skin
        # contaminated players from being misclassified as referees.
        if self.referee_colors:
            for ref_color in self.referee_colors:
                ref_distance = self._color_distance(dominant_color, ref_color)
                logger.debug(f"Referee color check: detected={dominant_color}, "
                            f"ref={ref_color}, ref_dist={ref_distance:.1f}, "
                            f"min_team_dist={min_team_distance:.1f}")
                if ref_distance < REFEREE_THRESHOLD:
                    # Only classify as referee if the ref color is a BETTER match
                    # than any team color (prevents player → referee misclassification)
                    if ref_distance < min_team_distance - REFEREE_VS_TEAM:
                        logger.info(f"Classified as REFEREE: color={dominant_color}, "
                                   f"ref_dist={ref_distance:.1f}, team_dist={min_team_distance:.1f}")
                        return "referee"
                    else:
                        logger.debug(f"Ref color matched but team color is closer — keeping as player")

        # ---- Step 4: Position-based GK heuristic (fallback) ----
        if self.field_bounds and dominant_color:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            field_x1, field_y1, field_x2, field_y2 = self.field_bounds
            field_width = field_x2 - field_x1
            field_height = field_y2 - field_y1

            # Horizontal: outer 15% of field width (goal area)
            edge_threshold = field_width * 0.15
            in_goal_area_x = cx < field_x1 + edge_threshold or cx > field_x2 - edge_threshold

            # Vertical: middle 70% of field height (not on touchline near corner flag)
            middle_margin = field_height * 0.15
            in_goal_area_y = cy > field_y1 + middle_margin and cy < field_y2 - middle_margin

            if in_goal_area_x and in_goal_area_y:
                # In goal area and color is very different from team colors → goalkeeper
                if min_team_distance > GK_TEAM_MIN_DIST:
                    logger.info(f"Classified as GOALKEEPER (position+color): "
                               f"color={dominant_color}, min_team_dist={min_team_distance:.1f}")
                    return "goalkeeper"

        return "player"

    @staticmethod
    def _color_distance_rgb(
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """Calculate Euclidean distance between two RGB colors (legacy)"""
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

    @staticmethod
    def _color_distance_lab(
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """
        Calculate perceptual color distance using CIE LAB color space.

        LAB distance correlates much better with human color perception than RGB.
        Two colors that look similar to a human will have a small LAB distance,
        even if their RGB values are far apart (e.g. two different blues under
        different lighting).
        """
        # Convert single-pixel RGB images to LAB
        c1 = np.array([[color1]], dtype=np.uint8)
        c2 = np.array([[color2]], dtype=np.uint8)
        lab1 = cv2.cvtColor(c1, cv2.COLOR_RGB2Lab).astype(np.float32)[0, 0]
        lab2 = cv2.cvtColor(c2, cv2.COLOR_RGB2Lab).astype(np.float32)[0, 0]
        return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))

    def _color_distance(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """
        Calculate perceptual color distance (LAB by default).

        Uses CIE LAB for more accurate human-perception-aligned color matching.
        NOTE: LAB distances are roughly 0-200 range (vs 0-441 for RGB).
        Thresholds throughout this code are calibrated for LAB distances.
        """
        return self._color_distance_lab(color1, color2)

    def set_referee_colors(self, colors: List[Tuple[int, int, int]]):
        """Set known referee jersey colors (RGB)"""
        self.referee_colors = colors
        logger.info(f"Referee colors set: {colors}")

    def set_goalkeeper_colors(
        self,
        home_gk_color: Optional[Tuple[int, int, int]] = None,
        away_gk_color: Optional[Tuple[int, int, int]] = None
    ):
        """Set known goalkeeper jersey colors (RGB)"""
        if home_gk_color:
            self.goalkeeper_colors[0] = home_gk_color
        if away_gk_color:
            self.goalkeeper_colors[1] = away_gk_color
        logger.info(f"Goalkeeper colors set: {self.goalkeeper_colors}")
    
    def _extract_dominant_color(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[Tuple[int, int, int]]:
        """
        Extract dominant jersey color from a player detection.
        Uses improved algorithm with grass filtering and K-means clustering.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            width = x2 - x1
            height = y2 - y1

            # Focus on center of jersey area (avoid edges/background)
            # For distant players (small bboxes), use larger region
            if height < 80:  # Small/distant player
                jersey_y1 = y1 + int(height * 0.20)  # Skip head
                jersey_y2 = y1 + int(height * 0.55)  # Larger torso region
                jersey_x1 = x1 + int(width * 0.15)   # Less edge cropping
                jersey_x2 = x2 - int(width * 0.15)
            else:  # Normal/close player
                jersey_y1 = y1 + int(height * 0.18)  # Skip head
                jersey_y2 = y1 + int(height * 0.50)  # Upper body
                jersey_x1 = x1 + int(width * 0.20)   # Avoid edges
                jersey_x2 = x2 - int(width * 0.20)

            # Ensure valid coordinates
            jersey_y1 = max(0, min(jersey_y1, frame.shape[0] - 1))
            jersey_y2 = max(0, min(jersey_y2, frame.shape[0] - 1))
            jersey_x1 = max(0, min(jersey_x1, frame.shape[1] - 1))
            jersey_x2 = max(0, min(jersey_x2, frame.shape[1] - 1))

            if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
                return None

            roi = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]

            if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
                return None

            # Convert to HSV to filter out grass
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Create mask to exclude grass-colored pixels (green hues)
            # Grass is typically H: 30-90, high S
            grass_mask = (
                (roi_hsv[:, :, 0] >= 30) & (roi_hsv[:, :, 0] <= 90) &
                (roi_hsv[:, :, 1] >= 40)
            )

            # Exclude skin tones (H: 0-25 or 340-360, low-medium S, medium-high V)
            # In OpenCV HSV, H is 0-180, so skin is roughly H: 0-15 or 170-180
            skin_mask = (
                ((roi_hsv[:, :, 0] <= 15) | (roi_hsv[:, :, 0] >= 170)) &
                (roi_hsv[:, :, 1] >= 20) & (roi_hsv[:, :, 1] <= 150) &
                (roi_hsv[:, :, 2] >= 50) & (roi_hsv[:, :, 2] <= 230)
            )

            # Also exclude very dark pixels (shadows) and very bright (overexposed)
            dark_mask = roi_hsv[:, :, 2] < 40  # Slightly higher threshold
            bright_mask = roi_hsv[:, :, 2] > 250

            # Combined mask: exclude grass, skin, shadows, and overexposed
            exclude_mask = grass_mask | skin_mask | dark_mask | bright_mask

            # Get valid pixels
            valid_pixels = roi_rgb[~exclude_mask]

            if len(valid_pixels) < 10:
                # Fall back to simple mean if too few valid pixels
                pixels = roi_rgb.reshape(-1, 3).astype(np.float32)
                mean_color = np.mean(pixels, axis=0).astype(int)
                return tuple(mean_color)

            # Fast histogram-based dominant color extraction (O(n) vs K-means O(n*k*i))
            # Quantize colors to 16x16x16 bins (4096 bins total) for better discrimination
            pixels = valid_pixels.astype(np.int32)

            # Quantize each channel to 16 levels (0-255 -> 0-15)
            quantized = pixels // 16  # 256/16 = 16 levels per channel

            # Create single index for each color bin: r*256 + g*16 + b
            bin_indices = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]

            # Count occurrences of each bin
            bin_counts = np.bincount(bin_indices, minlength=4096)

            # Find the most common bin
            dominant_bin = np.argmax(bin_counts)

            # Get all pixels that fall into this bin
            mask = bin_indices == dominant_bin
            dominant_pixels = pixels[mask]

            if len(dominant_pixels) > 0:
                # Return mean of pixels in dominant bin
                dominant_color = np.mean(dominant_pixels, axis=0).astype(int)
            else:
                # Fallback to overall mean
                dominant_color = np.mean(pixels, axis=0).astype(int)

            return tuple(dominant_color)

        except Exception as e:
            logger.debug(f"Color extraction failed: {e}")
            return None
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        fps: float = 30.0,
        detect_pitch: bool = False,
        track_objects: bool = True,
        confidence_threshold: Optional[float] = None
    ) -> List[FrameDetections]:
        """
        Run detection on a batch of frames using GPU parallelism.

        More efficient than calling detect_frame repeatedly when using GPU,
        as it processes all frames in a single YOLO inference call.

        Args:
            frames: List of BGR images as numpy arrays
            frame_numbers: Corresponding frame numbers
            fps: Video frames per second
            detect_pitch: Whether to detect pitch keypoints
            track_objects: Whether to apply tracking for persistent IDs
            confidence_threshold: Override default confidence threshold

        Returns:
            List of FrameDetections objects, one per input frame
        """
        if len(frames) == 0:
            return []

        if len(frames) != len(frame_numbers):
            raise ValueError("frames and frame_numbers must have same length")

        player_conf = confidence_threshold or settings.player_confidence_threshold
        ball_conf = settings.ball_confidence_threshold
        min_conf = min(player_conf, ball_conf, 0.15)

        # Initialize results list
        results_list: List[FrameDetections] = []

        try:
            # Downscale all frames for faster inference
            inference_frames = []
            scale_factor = 1.0
            for f in frames:
                inf_f, sf = self._downscale_for_inference(f)
                inference_frames.append(inf_f)
                scale_factor = sf  # Same for all frames (same resolution)

            # Run batch YOLO inference - much more efficient on GPU
            yolo_results_batch = self.model.predict(
                inference_frames,
                conf=min_conf,
                verbose=False,
                classes=[self.COCO_PERSON, self.COCO_SPORTS_BALL]
            )

            # Process each frame's results
            for i, (frame, frame_num, yolo_results) in enumerate(
                zip(frames, frame_numbers, yolo_results_batch)
            ):
                timestamp = frame_num / fps
                frame_detections = FrameDetections(
                    frame_number=frame_num,
                    timestamp_seconds=timestamp
                )

                if SUPERVISION_AVAILABLE:
                    detections = sv.Detections.from_ultralytics(yolo_results)

                    # Scale bboxes back to original resolution
                    detections = self._scale_detections_back(detections, scale_factor)

                    # Apply tracking if enabled (must be done sequentially)
                    if track_objects and len(detections) > 0 and self.tracker is not None:
                        detections = self.tracker.update_with_detections(detections)

                    frame_detections.raw_detections = detections
                    # Use original full-res frame for color extraction
                    self._parse_yolo_detections(frame, detections, frame_detections)
                else:
                    if scale_factor != 1.0 and yolo_results.boxes is not None:
                        yolo_results.boxes.xyxy *= scale_factor
                    self._parse_yolo_results_direct(frame, yolo_results, frame_detections)

                results_list.append(frame_detections)

            logger.debug(f"Batch detection completed: {len(frames)} frames")

        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

            # Fallback: return empty detections for each frame
            for frame_num in frame_numbers:
                results_list.append(FrameDetections(
                    frame_number=frame_num,
                    timestamp_seconds=frame_num / fps
                ))

        return results_list

    def reset_tracker(self):
        """Reset the object tracker and color cache (call between videos)"""
        self._tracker = None
        self._color_cache.clear()
        self._color_cache_hits = 0
        self._color_cache_misses = 0
        logger.debug("Tracker and color cache reset")

    def get_color_cache_stats(self) -> Dict[str, Any]:
        """Get color cache statistics for performance monitoring"""
        total = self._color_cache_hits + self._color_cache_misses
        hit_rate = (self._color_cache_hits / total * 100) if total > 0 else 0.0
        return {
            "cache_size": len(self._color_cache),
            "cache_hits": self._color_cache_hits,
            "cache_misses": self._color_cache_misses,
            "hit_rate_percent": hit_rate
        }


# ============================================
# Team Classifier
# ============================================

class TeamClassifier:
    """
    Classifies players into teams based on jersey color.

    Two modes:
    - **Auto mode** (default): K-Means clustering on collected player colors.
    - **Locked mode**: When the user manually sets team colors via
      ``set_team_colors()``, the classifier uses pure LAB-space distance to
      the two locked centers. This prevents K-Means from drifting due to
      lighting changes and makes user-specified colors authoritative.
    """

    def __init__(self, n_teams: int = 2):
        """
        Initialize team classifier.

        Args:
            n_teams: Number of teams (default 2)
        """
        from sklearn.cluster import KMeans
        import sklearn

        self.n_teams = n_teams

        # Handle sklearn version compatibility
        sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        if sklearn_version >= (1, 2):
            self.kmeans = KMeans(
                n_clusters=n_teams, n_init=10, random_state=42, algorithm='lloyd'
            )
        else:
            self.kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=42)

        self._is_fitted = False
        self._team_colors = None

        # Locked mode: user-specified colors are authoritative
        self._locked = False
        self._locked_colors: Optional[List[Tuple[int, int, int]]] = None

        # Team classification cache (tracker_id -> team_id)
        self._team_cache: Dict[int, int] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def fit(self, colors: List[Tuple[int, int, int]]):
        """
        Fit the classifier on a collection of jersey colors.

        Only runs K-Means if not in locked mode.

        Args:
            colors: List of RGB tuples
        """
        # In locked mode, ignore auto-fitting — user colors are authoritative
        if self._locked:
            logger.debug("Team classifier is locked to user colors — ignoring fit()")
            return

        if len(colors) < self.n_teams:
            logger.warning(f"Not enough colors to classify ({len(colors)} < {self.n_teams})")
            return

        color_array = np.array(colors)
        self.kmeans.fit(color_array)
        self._team_colors = self.kmeans.cluster_centers_.astype(int)
        self._is_fitted = True

        logger.info(f"Team classifier fitted with colors: {self._team_colors}")

    def classify(self, color: Tuple[int, int, int]) -> int:
        """
        Classify a single color to a team.

        In locked mode, uses LAB distance to locked centers.
        In auto mode, uses K-Means prediction.

        Args:
            color: RGB tuple

        Returns:
            Team ID (0 or 1)
        """
        if self._locked and self._locked_colors:
            # Locked mode — pure LAB distance to user-specified team colors
            distances = [
                SoccerDetector._color_distance_lab(color, tc)
                for tc in self._locked_colors
            ]
            return int(np.argmin(distances))

        if not self._is_fitted:
            return -1

        color_array = np.array([color])
        team_id = self.kmeans.predict(color_array)[0]
        return int(team_id)

    def classify_players(self, players: List[PlayerDetection]) -> List[PlayerDetection]:
        """
        Classify all players in a list, using cache for tracked players.

        Args:
            players: List of PlayerDetection objects with dominant_color set

        Returns:
            Same list with team_id assigned
        """
        for player in players:
            # Check cache first for tracked players
            if player.tracker_id is not None and player.tracker_id in self._team_cache:
                player.team_id = self._team_cache[player.tracker_id]
                self._cache_hits += 1
            elif player.dominant_color is not None:
                # Cache miss - classify and cache the result
                player.team_id = self.classify(player.dominant_color)
                self._cache_misses += 1
                if player.tracker_id is not None and player.team_id >= 0:
                    self._team_cache[player.tracker_id] = player.team_id

        return players

    @property
    def team_colors(self) -> Optional[np.ndarray]:
        """Get the current team colors (locked or learned)"""
        return self._team_colors

    @property
    def is_locked(self) -> bool:
        """Whether team colors are locked to user-specified values"""
        return self._locked

    def set_team_colors(
        self,
        home_color: Tuple[int, int, int],
        away_color: Tuple[int, int, int]
    ):
        """
        Manually set team colors and lock the classifier.

        In locked mode, classify() uses LAB distance to these exact colors
        instead of K-Means prediction. This prevents cluster drift and makes
        user-specified colors authoritative.

        Args:
            home_color: RGB tuple for home team
            away_color: RGB tuple for away team
        """
        self._team_colors = np.array([home_color, away_color])
        self._locked = True
        self._locked_colors = [home_color, away_color]

        # Still fit KMeans so that code expecting fitted state doesn't break,
        # but classify() will bypass it when locked.
        fake_data = np.array([home_color, away_color], dtype=np.float64)
        self.kmeans.fit(fake_data)
        self.kmeans.cluster_centers_ = self._team_colors.astype(np.float64)
        self._is_fitted = True

        # Clear cache when team colors change
        self._team_cache.clear()

        logger.info(f"Team colors LOCKED: Home={home_color}, Away={away_color}")

    def unlock(self):
        """Unlock the classifier to allow auto-fitting via K-Means again."""
        self._locked = False
        self._locked_colors = None
        self._team_cache.clear()
        logger.info("Team classifier unlocked — will auto-fit on next fit() call")

    def reset_cache(self):
        """Reset the team classification cache (call between videos)"""
        self._team_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Team classification cache reset")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for performance monitoring"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0.0
        return {
            "cache_size": len(self._team_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": hit_rate
        }


# ============================================
# Possession Calculator
# ============================================

class PossessionCalculator:
    """
    Calculates ball possession based on player proximity to the ball.
    """

    def __init__(self, proximity_threshold: float = 100.0):
        """
        Args:
            proximity_threshold: Maximum distance (pixels) for possession attribution
        """
        self.proximity_threshold = proximity_threshold
        self.possession_history: List[Optional[int]] = []  # team_id or None
        self.home_frames = 0
        self.away_frames = 0
        self.contested_frames = 0

    def calculate_possession(self, detections: FrameDetections) -> Optional[int]:
        """
        Determine which team has possession based on player proximity to ball.

        Args:
            detections: Frame detections with players and ball

        Returns:
            Team ID (0=home, 1=away) or None if no possession
        """
        if detections.ball is None:
            self.possession_history.append(None)
            self.contested_frames += 1
            return None

        ball_center = detections.ball.center

        # Find closest player to ball
        closest_player = None
        min_distance = float('inf')

        all_players = detections.players + detections.goalkeepers
        for player in all_players:
            player_center = player.center
            distance = np.sqrt(
                (player_center[0] - ball_center[0]) ** 2 +
                (player_center[1] - ball_center[1]) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_player = player

        # Attribute possession if player is close enough
        if closest_player and min_distance < self.proximity_threshold:
            team_id = closest_player.team_id
            if team_id is not None and team_id >= 0:
                self.possession_history.append(team_id)
                if team_id == 0:
                    self.home_frames += 1
                else:
                    self.away_frames += 1
                return team_id

        self.possession_history.append(None)
        self.contested_frames += 1
        return None

    def get_possession_percentage(self) -> Tuple[float, float]:
        """
        Get possession percentage for each team.

        Returns:
            (home_percentage, away_percentage)
        """
        total = self.home_frames + self.away_frames
        if total == 0:
            return (50.0, 50.0)

        home_pct = (self.home_frames / total) * 100
        away_pct = (self.away_frames / total) * 100
        return (home_pct, away_pct)

    def reset(self):
        """Reset possession tracking"""
        self.possession_history.clear()
        self.home_frames = 0
        self.away_frames = 0
        self.contested_frames = 0


# ============================================
# Pitch Transformer
# ============================================

class PitchTransformer:
    """
    Transforms video coordinates to real-world pitch coordinates
    using homography from detected pitch keypoints.
    """
    
    # Standard soccer pitch dimensions (in meters)
    PITCH_LENGTH = 105.0  # Length
    PITCH_WIDTH = 68.0    # Width
    
    def __init__(self):
        self.homography_matrix = None
        self._is_calibrated = False
    
    def calibrate(
        self,
        video_points: List[Tuple[float, float]],
        pitch_points: List[Tuple[float, float]]
    ) -> bool:
        """
        Calibrate the transformer using corresponding points.
        
        Args:
            video_points: Points in video frame coordinates
            pitch_points: Corresponding points in pitch coordinates (meters)
            
        Returns:
            True if calibration successful
        """
        if len(video_points) < 4 or len(pitch_points) < 4:
            logger.warning("Need at least 4 points for calibration")
            return False
        
        src = np.array(video_points, dtype=np.float32)
        dst = np.array(pitch_points, dtype=np.float32)
        
        self.homography_matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        
        if self.homography_matrix is not None:
            self._is_calibrated = True
            logger.info("Pitch transformer calibrated successfully")
            return True
        else:
            logger.error("Homography calculation failed")
            return False
    
    def transform_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Transform a single point from video to pitch coordinates.
        
        Args:
            x, y: Video coordinates
            
        Returns:
            Pitch coordinates (meters) or None if not calibrated
        """
        if not self._is_calibrated:
            return None
        
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        
        return (float(transformed[0][0][0]), float(transformed[0][0][1]))
    
    def transform_points(
        self,
        points: List[Tuple[float, float]]
    ) -> List[Optional[Tuple[float, float]]]:
        """Transform multiple points"""
        return [self.transform_point(x, y) for x, y in points]
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated


# ============================================
# Utility Functions
# ============================================

def draw_detections(
    frame: np.ndarray,
    detections: FrameDetections,
    team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    draw_ball: bool = True,
    draw_referees: bool = True,
    draw_labels: bool = True,
    simplified: Optional[bool] = None
) -> np.ndarray:
    """
    Draw detection boxes and labels on a frame.

    Args:
        frame: BGR image
        detections: FrameDetections object
        team_colors: Dict mapping team_id to BGR color
        draw_ball: Whether to draw ball
        draw_referees: Whether to draw referees
        draw_labels: Whether to draw text labels
        simplified: Use dots instead of rectangles+text for faster rendering.
                    If None, reads from settings.simplified_annotations.

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    # Determine whether to use simplified annotations
    use_simplified = simplified if simplified is not None else settings.simplified_annotations

    # Default colors (BGR)
    default_colors = {
        0: (0, 255, 255),   # Home - Yellow
        1: (0, 0, 255),     # Away - Red
        -1: (128, 128, 128) # Unknown - Gray
    }
    team_colors = team_colors or default_colors

    ball_color = (0, 165, 255)      # Orange
    referee_color = (50, 50, 50)    # Dark gray
    goalkeeper_color = (0, 255, 0)  # Green

    if use_simplified:
        # Fast path: colored dots at feet (bottom-center of bbox) — no text, no rectangles
        for player in detections.players:
            color = team_colors.get(player.team_id, team_colors.get(-1, (128, 128, 128)))
            x1, y1, x2, y2 = map(int, player.bbox)
            cx = (x1 + x2) // 2
            cv2.circle(annotated, (cx, y2), 8, color, -1)
            cv2.circle(annotated, (cx, y2), 9, (255, 255, 255), 1)

        for gk in detections.goalkeepers:
            x1, y1, x2, y2 = map(int, gk.bbox)
            cx = (x1 + x2) // 2
            cv2.circle(annotated, (cx, y2), 10, goalkeeper_color, -1)
            cv2.circle(annotated, (cx, y2), 11, (255, 255, 255), 2)

        if draw_referees:
            for ref in detections.referees:
                x1, y1, x2, y2 = map(int, ref.bbox)
                cx = (x1 + x2) // 2
                cv2.circle(annotated, (cx, y2), 6, referee_color, -1)

        if draw_ball and detections.ball:
            cx, cy = detections.ball.center
            cv2.circle(annotated, (int(cx), int(cy)), 12, ball_color, -1)
            cv2.circle(annotated, (int(cx), int(cy)), 14, (255, 255, 255), 2)

        return annotated

    # Full annotation path: rectangles + labels
    # Draw players
    for player in detections.players:
        color = team_colors.get(player.team_id, team_colors.get(-1, (128, 128, 128)))
        _draw_detection_box(annotated, player, color, draw_labels)

    # Draw goalkeepers
    for gk in detections.goalkeepers:
        _draw_detection_box(annotated, gk, goalkeeper_color, draw_labels, prefix="GK")

    # Draw referees
    if draw_referees:
        for ref in detections.referees:
            _draw_detection_box(annotated, ref, referee_color, draw_labels, prefix="REF")

    # Draw ball
    if draw_ball:
        if detections.ball:
            ball = detections.ball
            cx, cy = ball.center
            logger.debug(f"[DRAW] Drawing ball at ({int(cx)}, {int(cy)}) conf={ball.confidence:.2f}")
            # Draw larger ball marker for visibility
            cv2.circle(annotated, (int(cx), int(cy)), 15, ball_color, -1)
            cv2.circle(annotated, (int(cx), int(cy)), 17, (255, 255, 255), 3)
            # Add "BALL" label
            cv2.putText(annotated, "BALL", (int(cx) - 20, int(cy) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            logger.debug(f"[DRAW] No ball in detections for frame {detections.frame_number}")

    return annotated


def _draw_detection_box(
    frame: np.ndarray,
    detection: Detection,
    color: Tuple[int, int, int],
    draw_label: bool = True,
    prefix: str = ""
):
    """Draw a single detection box with optional label"""
    x1, y1, x2, y2 = map(int, detection.bbox)
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    if draw_label:
        label_parts = []
        if prefix:
            label_parts.append(prefix)
        if detection.tracker_id is not None:
            label_parts.append(f"#{detection.tracker_id}")
        if hasattr(detection, 'jersey_number') and detection.jersey_number:
            label_parts.append(f"#{detection.jersey_number}")
        
        label = " ".join(label_parts)
        if label:
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
