"""
Soccer Film Analysis - Detection Pipeline
Uses local YOLO models for player, ball, and pitch detection
No API key required - runs entirely offline
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
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
        model_size: str = "small",
        device: Optional[str] = None,
        models_dir: Optional[Path] = None
    ):
        """
        Initialize the detector with local YOLO models.

        Args:
            model_size: YOLO model size - "nano", "small", "medium", "large", "xlarge"
            device: Compute device (cuda, mps, cpu) - auto-detected if None
            models_dir: Directory to store downloaded models
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not installed. Run: pip install ultralytics")

        self.device = device or settings.get_device()
        self.model_size = model_size
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

        # Run YOLO detection
        try:
            # Run inference with lower threshold to catch balls
            yolo_results = self.model.predict(
                frame,
                conf=min_conf,
                verbose=False,
                classes=[self.COCO_PERSON, self.COCO_SPORTS_BALL]  # Only detect persons and sports balls
            )[0]

            # Convert to supervision Detections for tracking
            if SUPERVISION_AVAILABLE:
                detections = sv.Detections.from_ultralytics(yolo_results)

                # Apply tracking if enabled
                if track_objects and len(detections) > 0 and self.tracker is not None:
                    detections = self.tracker.update_with_detections(detections)

                # Store raw detections
                results.raw_detections = detections

                # Parse detections into typed objects
                self._parse_yolo_detections(frame, detections, results)
            else:
                # Fallback without supervision - parse YOLO results directly
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
            dominant_color = None
            if class_id == self.COCO_PERSON:
                dominant_color = self._extract_dominant_color(frame, bbox)

            # Handle ball detection
            if class_id == self.COCO_SPORTS_BALL:
                ball = BallDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name="ball",
                    tracker_id=tracker_id
                )
                results.ball = ball

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

        # Check if color matches referee colors (use looser threshold of 100)
        if self.referee_colors:
            for ref_color in self.referee_colors:
                distance = self._color_distance(dominant_color, ref_color)
                logger.debug(f"Referee color check: detected={dominant_color}, ref={ref_color}, distance={distance}")
                if distance < 100:  # Increased from 60 to 100 for better matching
                    logger.debug(f"Classified as REFEREE: color distance {distance}")
                    return "referee"

        # Check if color matches goalkeeper colors
        for team_id, gk_color in self.goalkeeper_colors.items():
            distance = self._color_distance(dominant_color, gk_color)
            if distance < 100:  # Increased threshold
                logger.debug(f"Classified as GOALKEEPER: color distance {distance}")
                return "goalkeeper"

        # Position-based heuristic: goalkeepers are often near the edges
        if self.field_bounds:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            field_x1, _, field_x2, _ = self.field_bounds
            field_width = field_x2 - field_x1

            # If person is in the outer 15% of the field width, might be goalkeeper
            edge_threshold = field_width * 0.15
            if cx < field_x1 + edge_threshold or cx > field_x2 - edge_threshold:
                # Additional check: is the color very different from typical player colors?
                # This is a simple heuristic, real implementation would use learned classifier
                pass

        return "player"

    def _color_distance(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """Calculate Euclidean distance between two RGB colors"""
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

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
            jersey_y1 = y1 + int(height * 0.15)  # Skip head
            jersey_y2 = y1 + int(height * 0.45)  # Upper body only
            jersey_x1 = x1 + int(width * 0.2)    # Avoid edges
            jersey_x2 = x2 - int(width * 0.2)

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

            # Also exclude very dark pixels (shadows) and very bright (overexposed)
            dark_mask = roi_hsv[:, :, 2] < 30
            bright_mask = roi_hsv[:, :, 2] > 250

            # Combined mask: exclude grass, shadows, and overexposed
            exclude_mask = grass_mask | dark_mask | bright_mask

            # Get valid pixels
            valid_pixels = roi_rgb[~exclude_mask]

            if len(valid_pixels) < 10:
                # Fall back to simple mean if too few valid pixels
                pixels = roi_rgb.reshape(-1, 3).astype(np.float32)
                mean_color = np.mean(pixels, axis=0).astype(int)
                return tuple(mean_color)

            # Use K-means clustering to find dominant color (k=3 to handle variations)
            pixels = valid_pixels.astype(np.float32)

            # K-means with k=3 clusters
            from sklearn.cluster import KMeans
            n_clusters = min(3, len(pixels))
            kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42, max_iter=50)
            kmeans.fit(pixels)

            # Find the largest cluster (most common color)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_idx = labels[np.argmax(counts)]
            dominant_color = kmeans.cluster_centers_[dominant_idx].astype(int)

            return tuple(dominant_color)

        except Exception as e:
            logger.debug(f"Color extraction failed: {e}")
            return None
    
    def reset_tracker(self):
        """Reset the object tracker (call between videos)"""
        self._tracker = None
        logger.debug("Tracker reset")


# ============================================
# Team Classifier
# ============================================

class TeamClassifier:
    """
    Classifies players into teams based on jersey color.
    Uses K-means clustering on player colors.
    """
    
    def __init__(self, n_teams: int = 2):
        """
        Initialize team classifier.
        
        Args:
            n_teams: Number of teams (default 2)
        """
        from sklearn.cluster import KMeans
        
        self.n_teams = n_teams
        self.kmeans = KMeans(n_clusters=n_teams, n_init=10, random_state=42)
        self._is_fitted = False
        self._team_colors = None
    
    def fit(self, colors: List[Tuple[int, int, int]]):
        """
        Fit the classifier on a collection of jersey colors.
        
        Args:
            colors: List of RGB tuples
        """
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
        
        Args:
            color: RGB tuple
            
        Returns:
            Team ID (0 or 1)
        """
        if not self._is_fitted:
            return -1
        
        color_array = np.array([color])
        team_id = self.kmeans.predict(color_array)[0]
        return int(team_id)
    
    def classify_players(self, players: List[PlayerDetection]) -> List[PlayerDetection]:
        """
        Classify all players in a list.
        
        Args:
            players: List of PlayerDetection objects with dominant_color set
            
        Returns:
            Same list with team_id assigned
        """
        for player in players:
            if player.dominant_color is not None:
                player.team_id = self.classify(player.dominant_color)
        
        return players
    
    @property
    def team_colors(self) -> Optional[np.ndarray]:
        """Get the learned team colors"""
        return self._team_colors
    
    def set_team_colors(
        self,
        home_color: Tuple[int, int, int],
        away_color: Tuple[int, int, int]
    ):
        """
        Manually set team colors instead of learning them.
        
        Args:
            home_color: RGB tuple for home team
            away_color: RGB tuple for away team
        """
        self._team_colors = np.array([home_color, away_color])
        self.kmeans.cluster_centers_ = self._team_colors.astype(np.float64)
        self._is_fitted = True
        
        logger.info(f"Team colors manually set: Home={home_color}, Away={away_color}")


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
    draw_labels: bool = True
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
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
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
    
    # Draw players
    for player in detections.players:
        color = team_colors.get(player.team_id, team_colors[-1])
        _draw_detection_box(annotated, player, color, draw_labels)
    
    # Draw goalkeepers
    for gk in detections.goalkeepers:
        _draw_detection_box(annotated, gk, goalkeeper_color, draw_labels, prefix="GK")
    
    # Draw referees
    if draw_referees:
        for ref in detections.referees:
            _draw_detection_box(annotated, ref, referee_color, draw_labels, prefix="REF")
    
    # Draw ball
    if draw_ball and detections.ball:
        ball = detections.ball
        cx, cy = ball.center
        cv2.circle(annotated, (int(cx), int(cy)), 10, ball_color, -1)
        cv2.circle(annotated, (int(cx), int(cy)), 12, (255, 255, 255), 2)
    
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
