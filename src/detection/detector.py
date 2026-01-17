"""
Soccer Film Analysis - Detection Pipeline
Uses Roboflow Sports library for player, ball, and pitch detection
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from loguru import logger

# Roboflow & Supervision imports
try:
    import supervision as sv
    from inference import get_model
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    logger.warning("Roboflow/Supervision not available. Install with: pip install supervision inference roboflow")

# Import sports library if available
try:
    from sports.configs.soccer import SoccerFieldConfiguration
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    SPORTS_AVAILABLE = True
except ImportError:
    SPORTS_AVAILABLE = False
    logger.warning("Roboflow Sports not available. Install with: pip install git+https://github.com/roboflow/sports.git")

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
    Main detection class using Roboflow models for soccer analysis.
    
    Uses:
    - Player detection model (detects players, goalkeepers, referees, ball)
    - Pitch detection model (detects field lines and keypoints)
    """
    
    # Class mappings for player detection model
    CLASS_NAMES = {
        0: "ball",
        1: "goalkeeper",
        2: "player",
        3: "referee"
    }
    
    def __init__(
        self,
        player_model_id: Optional[str] = None,
        pitch_model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the detector with Roboflow models.
        
        Args:
            player_model_id: Roboflow model ID for player detection
            pitch_model_id: Roboflow model ID for pitch keypoint detection  
            api_key: Roboflow API key (uses settings if not provided)
            device: Compute device (cuda, mps, cpu)
        """
        if not ROBOFLOW_AVAILABLE:
            raise ImportError("Roboflow dependencies not installed. Run: pip install supervision inference roboflow")
        
        self.api_key = api_key or settings.roboflow_api_key
        self.device = device or settings.get_device()
        
        # Model IDs
        self.player_model_id = player_model_id or settings.player_detection_model
        self.pitch_model_id = pitch_model_id or settings.pitch_detection_model
        
        # Models (lazy loaded)
        self._player_model = None
        self._pitch_model = None
        
        # Tracking
        self._tracker = None
        
        # Team classifier
        self._team_classifier = None
        
        # Pitch configuration
        self.pitch_config = None
        if SPORTS_AVAILABLE:
            self.pitch_config = SoccerFieldConfiguration()
        
        logger.info(f"SoccerDetector initialized (device: {self.device})")
    
    @property
    def player_model(self):
        """Lazy load player detection model"""
        if self._player_model is None:
            logger.info(f"Loading player detection model: {self.player_model_id}")
            self._player_model = get_model(
                model_id=self.player_model_id,
                api_key=self.api_key
            )
        return self._player_model
    
    @property
    def pitch_model(self):
        """Lazy load pitch detection model"""
        if self._pitch_model is None:
            logger.info(f"Loading pitch detection model: {self.pitch_model_id}")
            self._pitch_model = get_model(
                model_id=self.pitch_model_id,
                api_key=self.api_key
            )
        return self._pitch_model
    
    @property
    def tracker(self):
        """Get ByteTrack tracker for object tracking"""
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
        detect_pitch: bool = True,
        track_objects: bool = True,
        confidence_threshold: Optional[float] = None
    ) -> FrameDetections:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image as numpy array
            frame_number: Current frame number
            fps: Video frames per second
            detect_pitch: Whether to also detect pitch keypoints
            track_objects: Whether to apply tracking for persistent IDs
            confidence_threshold: Override default confidence threshold
            
        Returns:
            FrameDetections object with all detected objects
        """
        conf = confidence_threshold or settings.player_confidence_threshold
        timestamp = frame_number / fps
        
        # Initialize results
        results = FrameDetections(
            frame_number=frame_number,
            timestamp_seconds=timestamp
        )
        
        # Run player detection
        try:
            player_results = self.player_model.infer(frame, confidence=conf)[0]
            detections = sv.Detections.from_inference(player_results)
            
            # Apply tracking if enabled
            if track_objects and len(detections) > 0:
                detections = self.tracker.update_with_detections(detections)
            
            # Store raw detections
            results.raw_detections = detections
            
            # Parse detections into typed objects
            self._parse_player_detections(frame, detections, results)
            
        except Exception as e:
            logger.error(f"Player detection failed on frame {frame_number}: {e}")
        
        # Run pitch detection if requested
        if detect_pitch:
            try:
                pitch_results = self.pitch_model.infer(frame, confidence=settings.pitch_confidence_threshold)[0]
                self._parse_pitch_detections(pitch_results, results)
            except Exception as e:
                logger.debug(f"Pitch detection failed on frame {frame_number}: {e}")
        
        return results
    
    def _parse_player_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        results: FrameDetections
    ):
        """Parse supervision detections into typed detection objects"""
        
        for i in range(len(detections)):
            bbox = tuple(detections.xyxy[i])
            confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            class_id = int(detections.class_id[i]) if detections.class_id is not None else 2
            tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None
            
            class_name = self.CLASS_NAMES.get(class_id, "unknown")
            
            # Extract dominant color from detection region
            dominant_color = self._extract_dominant_color(frame, bbox)
            
            if class_name == "ball":
                ball = BallDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    tracker_id=tracker_id
                )
                results.ball = ball
                
            elif class_name == "referee":
                ref = PlayerDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    tracker_id=tracker_id,
                    is_referee=True,
                    dominant_color=dominant_color
                )
                results.referees.append(ref)
                
            elif class_name == "goalkeeper":
                gk = PlayerDetection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
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
                    class_name=class_name,
                    tracker_id=tracker_id,
                    dominant_color=dominant_color
                )
                results.players.append(player)
    
    def _parse_pitch_detections(self, pitch_results: Any, results: FrameDetections):
        """Parse pitch keypoint detection results"""
        if hasattr(pitch_results, 'keypoints') and pitch_results.keypoints:
            for i, kp in enumerate(pitch_results.keypoints):
                results.pitch_keypoints.append(PitchKeypoint(
                    point_id=i,
                    x=kp.x,
                    y=kp.y,
                    confidence=kp.confidence if hasattr(kp, 'confidence') else 1.0
                ))
    
    def _extract_dominant_color(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[Tuple[int, int, int]]:
        """
        Extract dominant color from the upper body region of a detection.
        This is used for team classification.
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get upper portion (jersey area)
            height = y2 - y1
            jersey_y1 = y1 + int(height * 0.1)  # Skip head area
            jersey_y2 = y1 + int(height * 0.5)  # Upper body only
            
            # Ensure valid coordinates
            jersey_y1 = max(0, min(jersey_y1, frame.shape[0] - 1))
            jersey_y2 = max(0, min(jersey_y2, frame.shape[0] - 1))
            x1 = max(0, min(x1, frame.shape[1] - 1))
            x2 = max(0, min(x2, frame.shape[1] - 1))
            
            if jersey_y2 <= jersey_y1 or x2 <= x1:
                return None
            
            roi = frame[jersey_y1:jersey_y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # Convert to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Reshape and find dominant color using k-means
            pixels = roi_rgb.reshape(-1, 3).astype(np.float32)
            
            # Simple approach: use mean color (faster)
            # For more accuracy, use k-means clustering
            mean_color = np.mean(pixels, axis=0).astype(int)
            
            return tuple(mean_color)
            
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
