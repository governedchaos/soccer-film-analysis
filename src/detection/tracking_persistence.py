"""
Soccer Film Analysis - Enhanced Player Tracking Persistence
Improves player ID stability across frames using multiple cues
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


@dataclass
class PlayerTrackHistory:
    """Stores historical data for a tracked player"""
    tracker_id: int
    team_id: Optional[int] = None
    jersey_number: Optional[int] = None

    # Position history (normalized coordinates)
    positions: deque = field(default_factory=lambda: deque(maxlen=30))

    # Appearance features (average color histogram)
    color_histogram: Optional[np.ndarray] = None

    # Tracking stats
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    total_frames_seen: int = 0
    consecutive_missing: int = 0

    # Velocity for prediction
    velocity: Tuple[float, float] = (0.0, 0.0)

    def update_position(self, x: float, y: float, frame_num: int):
        """Update position history and calculate velocity"""
        self.positions.append((x, y, frame_num))
        self.last_seen_frame = frame_num
        self.total_frames_seen += 1
        self.consecutive_missing = 0

        # Calculate velocity from recent positions
        if len(self.positions) >= 2:
            p1 = self.positions[-2]
            p2 = self.positions[-1]
            dt = max(1, p2[2] - p1[2])  # Frame difference
            self.velocity = (
                (p2[0] - p1[0]) / dt,
                (p2[1] - p1[1]) / dt
            )

    def predict_position(self, frame_num: int) -> Tuple[float, float]:
        """Predict position at given frame based on velocity"""
        if not self.positions:
            return (0.5, 0.5)

        last_pos = self.positions[-1]
        frames_since = frame_num - last_pos[2]

        predicted_x = last_pos[0] + self.velocity[0] * frames_since
        predicted_y = last_pos[1] + self.velocity[1] * frames_since

        # Clamp to valid range
        predicted_x = max(0, min(1, predicted_x))
        predicted_y = max(0, min(1, predicted_y))

        return (predicted_x, predicted_y)

    def is_stale(self, current_frame: int, max_missing: int = 60) -> bool:
        """Check if track is too old to recover"""
        return (current_frame - self.last_seen_frame) > max_missing


class TrackingPersistenceManager:
    """
    Manages player tracking with improved ID persistence.

    Uses multiple strategies:
    1. Position-based matching with velocity prediction
    2. Appearance-based matching (color histograms)
    3. Jersey number matching when available
    4. Team constraint matching (IDs stay within team)
    """

    def __init__(
        self,
        max_missing_frames: int = 60,
        position_threshold: float = 0.1,
        appearance_threshold: float = 0.7,
        velocity_weight: float = 0.3
    ):
        """
        Args:
            max_missing_frames: Maximum frames a track can be missing before deletion
            position_threshold: Maximum normalized distance for position match
            appearance_threshold: Minimum similarity for appearance match
            velocity_weight: Weight of predicted position vs last position
        """
        self.max_missing_frames = max_missing_frames
        self.position_threshold = position_threshold
        self.appearance_threshold = appearance_threshold
        self.velocity_weight = velocity_weight

        # Active tracks by ID
        self.tracks: Dict[int, PlayerTrackHistory] = {}

        # Recently lost tracks (for recovery)
        self.lost_tracks: Dict[int, PlayerTrackHistory] = {}

        # ID mapping from external tracker to our stable ID
        self.id_mapping: Dict[int, int] = {}

        # Next available stable ID
        self._next_id = 1

        # Team track counts for validation
        self.team_counts: Dict[int, int] = {0: 0, 1: 0}

    def update(
        self,
        detections: List[Dict],
        frame_num: int,
        frame: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Update tracks with new detections.

        Args:
            detections: List of player detections with bbox, tracker_id, team_id
            frame_num: Current frame number
            frame: Optional frame for appearance features

        Returns:
            Updated detections with stable IDs
        """
        # Mark all current tracks as potentially missing
        for track in self.tracks.values():
            track.consecutive_missing += 1

        updated_detections = []
        matched_track_ids: Set[int] = set()
        unmatched_detections = []

        # First pass: match using external tracker ID if we have mapping
        for det in detections:
            external_id = det.get('tracker_id')
            if external_id in self.id_mapping:
                stable_id = self.id_mapping[external_id]
                if stable_id in self.tracks:
                    track = self.tracks[stable_id]
                    self._update_track(track, det, frame_num, frame)
                    det['stable_id'] = stable_id
                    updated_detections.append(det)
                    matched_track_ids.add(stable_id)
                else:
                    unmatched_detections.append(det)
            else:
                unmatched_detections.append(det)

        # Second pass: try to match unmatched detections to lost or unmatched tracks
        remaining_detections = []
        for det in unmatched_detections:
            stable_id = self._find_best_match(det, frame_num, matched_track_ids, frame)

            if stable_id is not None:
                # Matched to existing or lost track
                if stable_id in self.lost_tracks:
                    # Recover lost track
                    self.tracks[stable_id] = self.lost_tracks.pop(stable_id)

                track = self.tracks[stable_id]
                self._update_track(track, det, frame_num, frame)

                # Update ID mapping
                external_id = det.get('tracker_id')
                if external_id:
                    self.id_mapping[external_id] = stable_id

                det['stable_id'] = stable_id
                updated_detections.append(det)
                matched_track_ids.add(stable_id)
            else:
                remaining_detections.append(det)

        # Third pass: create new tracks for truly new detections
        for det in remaining_detections:
            stable_id = self._create_new_track(det, frame_num, frame)
            det['stable_id'] = stable_id
            updated_detections.append(det)

            external_id = det.get('tracker_id')
            if external_id:
                self.id_mapping[external_id] = stable_id

        # Move stale tracks to lost
        self._cleanup_stale_tracks(frame_num)

        return updated_detections

    def _update_track(
        self,
        track: PlayerTrackHistory,
        det: Dict,
        frame_num: int,
        frame: Optional[np.ndarray]
    ):
        """Update a track with new detection"""
        bbox = det.get('bbox', [0, 0, 0, 0])
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2

        # Normalize to 0-1 range (assuming frame coordinates)
        # In practice, you'd divide by frame dimensions
        track.update_position(x, y, frame_num)

        # Update team if detected
        if det.get('team_id') is not None:
            track.team_id = det['team_id']

        # Update jersey number if detected
        if det.get('jersey_number') is not None:
            track.jersey_number = det['jersey_number']

        # Update appearance if frame available
        if frame is not None:
            self._update_appearance(track, bbox, frame)

    def _update_appearance(self, track: PlayerTrackHistory, bbox: List, frame: np.ndarray):
        """Update color histogram for appearance matching"""
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]

                # Calculate color histogram
                import cv2
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                if track.color_histogram is None:
                    track.color_histogram = hist
                else:
                    # Exponential moving average
                    track.color_histogram = 0.8 * track.color_histogram + 0.2 * hist
        except Exception as e:
            pass  # Ignore appearance update failures

    def _find_best_match(
        self,
        det: Dict,
        frame_num: int,
        already_matched: Set[int],
        frame: Optional[np.ndarray]
    ) -> Optional[int]:
        """Find best matching track for a detection"""
        bbox = det.get('bbox', [0, 0, 0, 0])
        det_x = (bbox[0] + bbox[2]) / 2
        det_y = (bbox[1] + bbox[3]) / 2
        det_team = det.get('team_id')
        det_jersey = det.get('jersey_number')

        best_id = None
        best_score = 0.0

        # Check active tracks first
        candidates = list(self.tracks.items()) + list(self.lost_tracks.items())

        for track_id, track in candidates:
            if track_id in already_matched:
                continue

            # Team constraint
            if det_team is not None and track.team_id is not None:
                if det_team != track.team_id:
                    continue

            # Jersey number matching (strong signal)
            if det_jersey is not None and track.jersey_number is not None:
                if det_jersey == track.jersey_number:
                    return track_id  # Immediate match

            # Position matching with velocity prediction
            predicted = track.predict_position(frame_num)
            last_pos = track.positions[-1][:2] if track.positions else predicted

            # Weighted average of last position and prediction
            match_x = (1 - self.velocity_weight) * last_pos[0] + self.velocity_weight * predicted[0]
            match_y = (1 - self.velocity_weight) * last_pos[1] + self.velocity_weight * predicted[1]

            dist = np.sqrt((det_x - match_x)**2 + (det_y - match_y)**2)

            if dist > self.position_threshold:
                continue

            # Position score (closer = higher)
            position_score = 1.0 - (dist / self.position_threshold)

            # Appearance score
            appearance_score = 0.5  # Default
            if frame is not None and track.color_histogram is not None:
                try:
                    import cv2
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                        hist = cv2.normalize(hist, hist).flatten()

                        # Compare histograms
                        similarity = cv2.compareHist(
                            track.color_histogram.astype(np.float32),
                            hist.astype(np.float32),
                            cv2.HISTCMP_CORREL
                        )
                        appearance_score = max(0, similarity)
                except Exception:
                    pass

            # Combined score
            total_score = 0.6 * position_score + 0.4 * appearance_score

            # Bonus for same team
            if det_team is not None and track.team_id == det_team:
                total_score *= 1.1

            if total_score > best_score:
                best_score = total_score
                best_id = track_id

        return best_id if best_score > 0.4 else None

    def _create_new_track(
        self,
        det: Dict,
        frame_num: int,
        frame: Optional[np.ndarray]
    ) -> int:
        """Create a new track for an unmatched detection"""
        stable_id = self._next_id
        self._next_id += 1

        bbox = det.get('bbox', [0, 0, 0, 0])

        track = PlayerTrackHistory(
            tracker_id=stable_id,
            team_id=det.get('team_id'),
            jersey_number=det.get('jersey_number'),
            first_seen_frame=frame_num
        )

        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        track.update_position(x, y, frame_num)

        if frame is not None:
            self._update_appearance(track, bbox, frame)

        self.tracks[stable_id] = track

        return stable_id

    def _cleanup_stale_tracks(self, current_frame: int):
        """Move stale tracks to lost and remove very old tracks"""
        stale_ids = []

        for track_id, track in self.tracks.items():
            if track.consecutive_missing > 5:  # Missing for 5+ frames
                stale_ids.append(track_id)

        for track_id in stale_ids:
            track = self.tracks.pop(track_id)
            if not track.is_stale(current_frame, self.max_missing_frames):
                self.lost_tracks[track_id] = track

        # Remove very stale lost tracks
        very_stale = [
            tid for tid, t in self.lost_tracks.items()
            if t.is_stale(current_frame, self.max_missing_frames)
        ]
        for tid in very_stale:
            del self.lost_tracks[tid]

    def get_track_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'active_tracks': len(self.tracks),
            'lost_tracks': len(self.lost_tracks),
            'total_ids_assigned': self._next_id - 1,
            'id_mappings': len(self.id_mapping)
        }

    def get_player_track(self, stable_id: int) -> Optional[PlayerTrackHistory]:
        """Get track history for a player"""
        return self.tracks.get(stable_id) or self.lost_tracks.get(stable_id)

    def get_all_positions(self, stable_id: int) -> List[Tuple[float, float, int]]:
        """Get all recorded positions for a player"""
        track = self.get_player_track(stable_id)
        return list(track.positions) if track else []


class IDStabilizer:
    """
    Simple wrapper to stabilize IDs in existing detection pipeline.
    Can be plugged into the video processor.
    """

    def __init__(self):
        self.manager = TrackingPersistenceManager()

    def stabilize(self, detections, frame_num: int, frame=None) -> Dict[int, int]:
        """
        Stabilize tracker IDs for a frame.

        Args:
            detections: FrameDetections object
            frame_num: Current frame
            frame: Optional video frame for appearance matching

        Returns:
            Dict mapping original tracker_id to stable_id
        """
        # Convert detections to dict format
        det_list = []
        for player in detections.players:
            det_list.append({
                'tracker_id': player.tracker_id,
                'bbox': player.bbox,
                'team_id': player.team_id,
                'jersey_number': getattr(player, 'jersey_number', None)
            })

        # Update and get stable IDs
        updated = self.manager.update(det_list, frame_num, frame)

        # Build mapping
        mapping = {}
        for det in updated:
            if det.get('tracker_id') and det.get('stable_id'):
                mapping[det['tracker_id']] = det['stable_id']

        return mapping
