"""
Pitch Line Detection and Boundary Analysis
Detects pitch boundaries to filter out-of-play detections
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from loguru import logger


@dataclass
class PitchBoundary:
    """Detected pitch boundary information."""
    # Bounding rectangle of the playing field
    min_x: int
    min_y: int
    max_x: int
    max_y: int

    # Corner points (if detected)
    corners: Optional[List[Tuple[int, int]]] = None

    # Convex hull of the pitch area
    hull: Optional[np.ndarray] = None

    # Confidence score
    confidence: float = 0.0

    def contains_point(self, x: float, y: float, margin: int = 0) -> bool:
        """Check if a point is within the pitch boundary."""
        return (self.min_x - margin <= x <= self.max_x + margin and
                self.min_y - margin <= y <= self.max_y + margin)

    def is_in_goal_area(self, x: float, y: float) -> bool:
        """
        Check if a point is likely in a goal area (left or right 12% of pitch).
        Goalkeepers typically operate in this zone.
        """
        pitch_width = self.max_x - self.min_x
        goal_zone_width = pitch_width * 0.12  # ~12% on each side

        # Check if in left goal area
        if self.min_x <= x <= self.min_x + goal_zone_width:
            return True
        # Check if in right goal area
        if self.max_x - goal_zone_width <= x <= self.max_x:
            return True
        return False

    def get_goal_areas(self) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Get bounding boxes for estimated goal areas.
        Returns: (left_goal_area, right_goal_area) as (x1, y1, x2, y2)
        """
        pitch_width = self.max_x - self.min_x
        goal_zone_width = int(pitch_width * 0.12)

        # Goal areas are typically in the center vertical portion
        pitch_height = self.max_y - self.min_y
        goal_y1 = self.min_y + int(pitch_height * 0.25)
        goal_y2 = self.max_y - int(pitch_height * 0.25)

        left_goal = (self.min_x, goal_y1, self.min_x + goal_zone_width, goal_y2)
        right_goal = (self.max_x - goal_zone_width, goal_y1, self.max_x, goal_y2)

        return left_goal, right_goal

    def get_technical_areas(self) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """
        Estimate technical areas (coaches boxes) on the sidelines.

        Technical areas are on the touchline, roughly centered between penalty areas.
        Based on FIFA regulations: extends 1m each side of bench, within 1m of touchline.

        Returns: (top_technical_area, bottom_technical_area) as (x1, y1, x2, y2)
        """
        pitch_width = self.max_x - self.min_x
        pitch_height = self.max_y - self.min_y

        # Technical areas span roughly 20% of pitch width, centered
        ta_width = int(pitch_width * 0.20)
        center_x = (self.min_x + self.max_x) // 2

        # Technical area extends outside the pitch boundary
        ta_depth = 60  # pixels outside the pitch

        # Top sideline technical area (above pitch)
        top_ta = (
            center_x - ta_width // 2,
            self.min_y - ta_depth,
            center_x + ta_width // 2,
            self.min_y
        )

        # Bottom sideline technical area (below pitch)
        bottom_ta = (
            center_x - ta_width // 2,
            self.max_y,
            center_x + ta_width // 2,
            self.max_y + ta_depth
        )

        return top_ta, bottom_ta

    def is_in_technical_area(self, x: float, y: float) -> bool:
        """Check if a point is in a technical area (coaches box)."""
        top_ta, bottom_ta = self.get_technical_areas()

        def in_box(point_x, point_y, box):
            x1, y1, x2, y2 = box
            return x1 <= point_x <= x2 and y1 <= point_y <= y2

        return in_box(x, y, top_ta) or in_box(x, y, bottom_ta)

    def contains_bbox(self, bbox: Tuple[float, float, float, float], threshold: float = 0.5) -> bool:
        """
        Check if a bounding box is mostly within the pitch.

        Args:
            bbox: (x1, y1, x2, y2)
            threshold: Fraction of bbox that must be inside (0-1)
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Check if center is inside
        if not self.contains_point(cx, cy):
            return False

        # Check if bottom of bbox (feet) is inside - most important
        foot_y = y2
        foot_x = cx
        if not self.contains_point(foot_x, foot_y, margin=20):
            return False

        return True


class PitchDetector:
    """
    Detects soccer pitch boundaries using line detection and color analysis.
    """

    # Common grass colors in HSV
    GRASS_HSV_LOWER = np.array([30, 30, 30])   # Lower bound
    GRASS_HSV_UPPER = np.array([90, 255, 255]) # Upper bound

    # White line detection
    WHITE_THRESHOLD = 200  # Minimum brightness for white lines

    def __init__(
        self,
        min_line_length: int = 100,
        max_line_gap: int = 30,
        detection_interval: int = 30  # Only run every N frames
    ):
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.detection_interval = detection_interval

        # Cached boundary
        self._cached_boundary: Optional[PitchBoundary] = None
        self._last_detection_frame: int = -1000
        self._boundary_history: List[PitchBoundary] = []

    def detect(self, frame: np.ndarray, frame_number: int = 0, force: bool = False) -> Optional[PitchBoundary]:
        """
        Detect pitch boundaries in a frame.

        Args:
            frame: BGR image
            frame_number: Current frame number (for caching)
            force: Force detection even if cached

        Returns:
            PitchBoundary or None if detection failed
        """
        # Use cached result if recent
        if not force and self._cached_boundary is not None:
            if frame_number - self._last_detection_frame < self.detection_interval:
                return self._cached_boundary

        try:
            boundary = self._detect_boundary(frame)

            if boundary is not None:
                self._cached_boundary = boundary
                self._last_detection_frame = frame_number
                self._boundary_history.append(boundary)

                # Keep only recent history
                if len(self._boundary_history) > 10:
                    self._boundary_history = self._boundary_history[-10:]

                # Smooth with history
                boundary = self._smooth_boundary()

            return boundary

        except Exception as e:
            logger.debug(f"Pitch detection failed: {e}")
            return self._cached_boundary

    def _detect_boundary(self, frame: np.ndarray) -> Optional[PitchBoundary]:
        """Internal boundary detection using multiple methods."""
        height, width = frame.shape[:2]

        # Method 1: Grass color segmentation
        grass_mask = self._detect_grass(frame)

        # Method 2: White line detection
        lines = self._detect_white_lines(frame)

        # Combine methods
        if grass_mask is not None:
            # Find contours in grass mask
            contours, _ = cv2.findContours(
                grass_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Get largest contour (should be the pitch)
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                # Pitch should be a significant portion of the frame
                if area > (width * height * 0.3):
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest)

                    # Get convex hull for more accurate boundary
                    hull = cv2.convexHull(largest)

                    # Find corners
                    corners = self._find_corners(hull)

                    return PitchBoundary(
                        min_x=x,
                        min_y=y,
                        max_x=x + w,
                        max_y=y + h,
                        corners=corners,
                        hull=hull,
                        confidence=min(1.0, area / (width * height))
                    )

        # Fallback: use line-based detection
        if lines is not None and len(lines) > 4:
            boundary = self._boundary_from_lines(lines, width, height)
            if boundary is not None:
                return boundary

        # Last fallback: assume pitch is central 80% of frame
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.1)

        return PitchBoundary(
            min_x=margin_x,
            min_y=margin_y,
            max_x=width - margin_x,
            max_y=height - margin_y,
            confidence=0.3
        )

    def _detect_grass(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect grass regions using color segmentation."""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create mask for grass colors
            mask = cv2.inRange(hsv, self.GRASS_HSV_LOWER, self.GRASS_HSV_UPPER)

            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Fill holes
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.erode(mask, kernel, iterations=2)

            return mask

        except Exception as e:
            logger.debug(f"Grass detection failed: {e}")
            return None

    def _detect_white_lines(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect white pitch lines using edge detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Threshold for white
            _, white_mask = cv2.threshold(gray, self.WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Combine with white mask
            combined = cv2.bitwise_and(edges, white_mask)

            # Detect lines
            lines = cv2.HoughLinesP(
                combined,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )

            return lines

        except Exception as e:
            logger.debug(f"Line detection failed: {e}")
            return None

    def _boundary_from_lines(
        self,
        lines: np.ndarray,
        width: int,
        height: int
    ) -> Optional[PitchBoundary]:
        """Estimate pitch boundary from detected lines."""
        if lines is None or len(lines) == 0:
            return None

        # Collect all line endpoints
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])

        if len(points) < 4:
            return None

        points = np.array(points)

        # Get bounding box of all line points
        min_x = max(0, np.min(points[:, 0]) - 20)
        min_y = max(0, np.min(points[:, 1]) - 20)
        max_x = min(width, np.max(points[:, 0]) + 20)
        max_y = min(height, np.max(points[:, 1]) + 20)

        return PitchBoundary(
            min_x=int(min_x),
            min_y=int(min_y),
            max_x=int(max_x),
            max_y=int(max_y),
            confidence=0.6
        )

    def _find_corners(self, hull: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Find corner points of the pitch from convex hull."""
        if hull is None or len(hull) < 4:
            return None

        # Approximate hull to polygon
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4:
            # Perfect quadrilateral
            corners = [(int(p[0][0]), int(p[0][1])) for p in approx]
            return corners

        # Find 4 extreme points
        hull_points = hull.reshape(-1, 2)

        # Top-left, top-right, bottom-right, bottom-left
        sum_xy = hull_points[:, 0] + hull_points[:, 1]
        diff_xy = hull_points[:, 0] - hull_points[:, 1]

        top_left = hull_points[np.argmin(sum_xy)]
        bottom_right = hull_points[np.argmax(sum_xy)]
        top_right = hull_points[np.argmax(diff_xy)]
        bottom_left = hull_points[np.argmin(diff_xy)]

        return [
            (int(top_left[0]), int(top_left[1])),
            (int(top_right[0]), int(top_right[1])),
            (int(bottom_right[0]), int(bottom_right[1])),
            (int(bottom_left[0]), int(bottom_left[1]))
        ]

    def _smooth_boundary(self) -> PitchBoundary:
        """Smooth boundary using history."""
        if len(self._boundary_history) < 2:
            return self._cached_boundary

        # Average recent boundaries
        recent = self._boundary_history[-5:]

        avg_min_x = int(np.mean([b.min_x for b in recent]))
        avg_min_y = int(np.mean([b.min_y for b in recent]))
        avg_max_x = int(np.mean([b.max_x for b in recent]))
        avg_max_y = int(np.mean([b.max_y for b in recent]))
        avg_conf = np.mean([b.confidence for b in recent])

        return PitchBoundary(
            min_x=avg_min_x,
            min_y=avg_min_y,
            max_x=avg_max_x,
            max_y=avg_max_y,
            confidence=avg_conf
        )

    def is_on_pitch(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if a detection is on the pitch."""
        if self._cached_boundary is None:
            return True  # Assume on pitch if no boundary detected

        return self._cached_boundary.contains_bbox(bbox)

    def filter_detections(
        self,
        bboxes: List[Tuple[float, float, float, float]]
    ) -> List[bool]:
        """
        Filter a list of detections by pitch boundary.

        Returns:
            List of booleans indicating if each detection is on pitch
        """
        return [self.is_on_pitch(bbox) for bbox in bboxes]

    def reset(self):
        """Reset cached boundary."""
        self._cached_boundary = None
        self._boundary_history.clear()
        self._last_detection_frame = -1000

    def draw_boundary(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected pitch boundary on frame."""
        if self._cached_boundary is None:
            return frame

        annotated = frame.copy()
        boundary = self._cached_boundary

        # Draw bounding rectangle
        cv2.rectangle(
            annotated,
            (boundary.min_x, boundary.min_y),
            (boundary.max_x, boundary.max_y),
            (0, 255, 0), 2
        )

        # Draw hull if available
        if boundary.hull is not None:
            cv2.drawContours(annotated, [boundary.hull], 0, (255, 255, 0), 2)

        # Draw corners if available
        if boundary.corners:
            for corner in boundary.corners:
                cv2.circle(annotated, corner, 8, (0, 0, 255), -1)

        # Draw confidence
        cv2.putText(
            annotated,
            f"Pitch: {boundary.confidence:.1%}",
            (boundary.min_x, boundary.min_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        return annotated

    def detect_goal_posts(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect goal posts using white vertical line detection.

        Goal posts are white vertical structures at the far left/right of the pitch.
        Uses color filtering + line detection in goal areas.

        Returns:
            List of detected goal posts with 'side' ('left'/'right'), 'bbox', 'confidence'
        """
        if self._cached_boundary is None:
            return []

        goals = []
        height, width = frame.shape[:2]
        boundary = self._cached_boundary

        # Get goal areas
        left_goal_area, right_goal_area = boundary.get_goal_areas()

        for side, goal_area in [('left', left_goal_area), ('right', right_goal_area)]:
            x1, y1, x2, y2 = goal_area

            # Ensure bounds are within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Extract goal area ROI
            roi = frame[y1:y2, x1:x2]

            # Detect white regions (goal posts are white)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # Detect vertical lines
            edges = cv2.Canny(white_mask, 50, 150)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=30,
                maxLineGap=10
            )

            if lines is not None:
                # Find vertical lines (goal posts)
                vertical_lines = []
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    # Check if line is mostly vertical (angle > 70 degrees)
                    dx = abs(lx2 - lx1)
                    dy = abs(ly2 - ly1)
                    if dy > 0 and dx / dy < 0.4:  # More vertical than horizontal
                        vertical_lines.append(line[0])

                if vertical_lines:
                    # Find the tallest vertical line (likely the post)
                    best_line = max(vertical_lines, key=lambda l: abs(l[3] - l[1]))
                    lx1, ly1, lx2, ly2 = best_line

                    # Convert back to frame coordinates
                    post_x = x1 + (lx1 + lx2) // 2
                    post_y1 = y1 + min(ly1, ly2)
                    post_y2 = y1 + max(ly1, ly2)

                    goals.append({
                        'side': side,
                        'bbox': (post_x - 5, post_y1, post_x + 5, post_y2),
                        'confidence': 0.7,
                        'line': (post_x, post_y1, post_x, post_y2)
                    })

        logger.debug(f"Detected {len(goals)} goal posts")
        return goals

    def draw_goals(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected goals on frame."""
        annotated = frame.copy()

        goals = self.detect_goal_posts(frame)

        for goal in goals:
            x1, y1, x2, y2 = goal['bbox']
            side = goal['side']

            # Draw goal post as cyan rectangle
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)

            # Label
            label = f"GOAL ({side})"
            cv2.putText(
                annotated, label, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

        # Also draw goal areas
        if self._cached_boundary:
            left_area, right_area = self._cached_boundary.get_goal_areas()

            for area, label in [(left_area, "L-GOAL"), (right_area, "R-GOAL")]:
                ax1, ay1, ax2, ay2 = area
                cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 200, 200), 1)

        return annotated
