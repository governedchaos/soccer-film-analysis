"""
Soccer Film Analysis - Advanced Analytics
Heatmaps, pass networks, formations, possession sequences, shot detection, xG
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available for visualizations")

try:
    from mplsoccer import Pitch, VerticalPitch
    MPLSOCCER_AVAILABLE = True
except ImportError:
    MPLSOCCER_AVAILABLE = False
    logger.debug("mplsoccer not available, using basic pitch drawing")


# ============================================
# Data Classes
# ============================================

@dataclass
class PlayerPosition:
    """Player position at a specific time"""
    tracker_id: int
    team_id: int
    x: float  # 0-100 (pitch percentage)
    y: float  # 0-100 (pitch percentage)
    frame: int
    timestamp: float


@dataclass
class PassEvent:
    """A pass between two players"""
    from_player_id: int
    to_player_id: int
    from_pos: Tuple[float, float]
    to_pos: Tuple[float, float]
    frame: int
    timestamp: float
    team_id: int
    successful: bool = True


@dataclass
class ShotEvent:
    """A shot attempt"""
    player_id: int
    team_id: int
    position: Tuple[float, float]  # x, y on pitch (0-100)
    frame: int
    timestamp: float
    outcome: str  # 'goal', 'saved', 'blocked', 'missed'
    xg: float = 0.0  # Expected goals value


@dataclass
class PossessionSequence:
    """A sequence of possession by one team"""
    team_id: int
    start_frame: int
    end_frame: int
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    passes: List[PassEvent] = field(default_factory=list)
    shot: Optional[ShotEvent] = None


# ============================================
# Player Heatmaps
# ============================================

class HeatmapGenerator:
    """
    Generates player position heatmaps.
    """

    def __init__(self, pitch_width: int = 105, pitch_height: int = 68):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.position_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    def add_position(self, tracker_id: int, x: float, y: float):
        """Add a position observation for a player"""
        # Normalize to pitch coordinates
        self.position_history[tracker_id].append((x, y))

    def add_frame_detections(self, detections, frame_width: int, frame_height: int):
        """Add all player positions from a frame"""
        for player in detections.players + detections.goalkeepers:
            if player.tracker_id is not None:
                # Convert frame coordinates to pitch percentage
                x_pct = (player.center[0] / frame_width) * 100
                y_pct = (player.center[1] / frame_height) * 100
                self.add_position(player.tracker_id, x_pct, y_pct)

    def generate_heatmap(
        self,
        tracker_id: Optional[int] = None,
        team_id: Optional[int] = None,
        resolution: int = 50
    ) -> np.ndarray:
        """
        Generate heatmap array.

        Args:
            tracker_id: Specific player (None for all)
            team_id: Specific team (None for all)
            resolution: Grid resolution

        Returns:
            2D numpy array of heat values
        """
        heatmap = np.zeros((resolution, resolution))

        positions = []
        if tracker_id is not None:
            positions = self.position_history.get(tracker_id, [])
        else:
            for tid, pos_list in self.position_history.items():
                positions.extend(pos_list)

        for x, y in positions:
            # Convert percentage to grid cell
            gx = int((x / 100) * (resolution - 1))
            gy = int((y / 100) * (resolution - 1))
            gx = max(0, min(gx, resolution - 1))
            gy = max(0, min(gy, resolution - 1))
            heatmap[gy, gx] += 1

        # Apply Gaussian blur for smoothing
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=2)

        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def plot_heatmap(
        self,
        tracker_id: Optional[int] = None,
        title: str = "Player Heatmap",
        save_path: Optional[str] = None
    ):
        """Plot heatmap visualization"""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for plotting")
            return None

        heatmap = self.generate_heatmap(tracker_id)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw pitch
        self._draw_pitch(ax)

        # Overlay heatmap
        cmap = LinearSegmentedColormap.from_list('heat', ['green', 'yellow', 'red'])
        im = ax.imshow(heatmap, extent=[0, 100, 0, 100], origin='lower',
                       cmap=cmap, alpha=0.6, aspect='auto')

        plt.colorbar(im, ax=ax, label='Frequency')
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Heatmap saved to: {save_path}")

        return fig

    def _draw_pitch(self, ax):
        """Draw basic pitch outline"""
        # Pitch outline
        ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], 'white', lw=2)
        # Center line
        ax.plot([50, 50], [0, 100], 'white', lw=1)
        # Center circle
        circle = plt.Circle((50, 50), 9.15, fill=False, color='white', lw=1)
        ax.add_patch(circle)
        # Penalty areas
        ax.plot([0, 16.5, 16.5, 0], [21, 21, 79, 79], 'white', lw=1)
        ax.plot([100, 83.5, 83.5, 100], [21, 21, 79, 79], 'white', lw=1)
        ax.set_facecolor('green')


# ============================================
# Pass Network
# ============================================

class PassNetworkAnalyzer:
    """
    Analyzes and visualizes passing networks.
    """

    def __init__(self):
        self.passes: List[PassEvent] = []
        self.player_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    def detect_pass(
        self,
        prev_detections,
        curr_detections,
        ball_proximity_threshold: float = 50.0
    ) -> Optional[PassEvent]:
        """
        Detect a pass between frames.

        A pass is detected when the ball moves from proximity of one player
        to another player on the same team.
        """
        if not prev_detections or not curr_detections:
            return None

        if not prev_detections.ball or not curr_detections.ball:
            return None

        prev_ball = prev_detections.ball.center
        curr_ball = curr_detections.ball.center

        # Find player closest to ball in previous frame
        prev_player = self._find_closest_player(prev_detections, prev_ball)
        # Find player closest to ball in current frame
        curr_player = self._find_closest_player(curr_detections, curr_ball)

        if prev_player and curr_player:
            if (prev_player.tracker_id != curr_player.tracker_id and
                prev_player.team_id == curr_player.team_id and
                prev_player.team_id is not None):

                # This is a pass
                pass_event = PassEvent(
                    from_player_id=prev_player.tracker_id,
                    to_player_id=curr_player.tracker_id,
                    from_pos=prev_player.center,
                    to_pos=curr_player.center,
                    frame=curr_detections.frame_number,
                    timestamp=curr_detections.timestamp_seconds,
                    team_id=prev_player.team_id
                )
                self.passes.append(pass_event)
                return pass_event

        return None

    def _find_closest_player(self, detections, point: Tuple[float, float], max_dist: float = 100):
        """Find player closest to a point"""
        closest = None
        min_dist = max_dist

        for player in detections.players + detections.goalkeepers:
            dist = np.sqrt((player.center[0] - point[0])**2 +
                          (player.center[1] - point[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest = player

        return closest

    def get_pass_matrix(self, team_id: Optional[int] = None) -> Dict[Tuple[int, int], int]:
        """Get pass count matrix between players"""
        matrix = defaultdict(int)
        for p in self.passes:
            if team_id is None or p.team_id == team_id:
                matrix[(p.from_player_id, p.to_player_id)] += 1
        return dict(matrix)

    def get_player_pass_stats(self, tracker_id: int) -> Dict:
        """Get passing stats for a specific player"""
        passes_made = [p for p in self.passes if p.from_player_id == tracker_id]
        passes_received = [p for p in self.passes if p.to_player_id == tracker_id]

        return {
            "passes_made": len(passes_made),
            "passes_received": len(passes_received),
            "pass_accuracy": sum(1 for p in passes_made if p.successful) / max(1, len(passes_made)),
            "most_frequent_target": self._most_common_partner(passes_made, 'to'),
            "most_frequent_source": self._most_common_partner(passes_received, 'from')
        }

    def _most_common_partner(self, passes: List[PassEvent], direction: str) -> Optional[int]:
        """Find most common pass partner"""
        counts = defaultdict(int)
        for p in passes:
            partner = p.to_player_id if direction == 'to' else p.from_player_id
            counts[partner] += 1
        if counts:
            return max(counts, key=counts.get)
        return None

    def plot_pass_network(
        self,
        team_id: int,
        avg_positions: Dict[int, Tuple[float, float]],
        save_path: Optional[str] = None
    ):
        """Plot pass network visualization"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor('green')

        # Draw pitch
        ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], 'white', lw=2)
        ax.plot([50, 50], [0, 100], 'white', lw=1)

        # Get pass matrix
        pass_matrix = self.get_pass_matrix(team_id)

        # Find max passes for scaling
        max_passes = max(pass_matrix.values()) if pass_matrix else 1

        # Draw connections
        for (from_id, to_id), count in pass_matrix.items():
            if from_id in avg_positions and to_id in avg_positions:
                x1, y1 = avg_positions[from_id]
                x2, y2 = avg_positions[to_id]
                width = (count / max_passes) * 5 + 0.5
                ax.plot([x1, x2], [y1, y2], 'white', lw=width, alpha=0.6)

        # Draw player positions
        for player_id, (x, y) in avg_positions.items():
            passes_made = len([p for p in self.passes
                              if p.from_player_id == player_id and p.team_id == team_id])
            size = 100 + passes_made * 20
            ax.scatter(x, y, s=size, c='yellow', edgecolors='black', zorder=5)
            ax.annotate(f'#{player_id}', (x, y), ha='center', va='center',
                       fontsize=8, fontweight='bold')

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_title(f'Pass Network - Team {team_id}')
        ax.set_aspect('equal')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


# ============================================
# Formation Detection
# ============================================

class FormationDetector:
    """
    Detects team formations (4-4-2, 4-3-3, etc.)
    """

    COMMON_FORMATIONS = {
        "4-4-2": [(1, 0.5), (2, 0.2), (2, 0.4), (2, 0.6), (2, 0.8),
                  (3, 0.2), (3, 0.4), (3, 0.6), (3, 0.8),
                  (4, 0.35), (4, 0.65)],
        "4-3-3": [(1, 0.5), (2, 0.2), (2, 0.4), (2, 0.6), (2, 0.8),
                  (3, 0.25), (3, 0.5), (3, 0.75),
                  (4, 0.2), (4, 0.5), (4, 0.8)],
        "3-5-2": [(1, 0.5), (2, 0.25), (2, 0.5), (2, 0.75),
                  (3, 0.1), (3, 0.3), (3, 0.5), (3, 0.7), (3, 0.9),
                  (4, 0.35), (4, 0.65)],
        "4-2-3-1": [(1, 0.5), (2, 0.2), (2, 0.4), (2, 0.6), (2, 0.8),
                    (3, 0.35), (3, 0.65),
                    (3.5, 0.2), (3.5, 0.5), (3.5, 0.8),
                    (4, 0.5)],
        "5-3-2": [(1, 0.5), (2, 0.1), (2, 0.3), (2, 0.5), (2, 0.7), (2, 0.9),
                  (3, 0.25), (3, 0.5), (3, 0.75),
                  (4, 0.35), (4, 0.65)],
    }

    def __init__(self):
        self.position_samples: Dict[int, List[List[Tuple[float, float]]]] = {0: [], 1: []}

    def add_frame(self, detections, frame_width: int, frame_height: int):
        """Add player positions from a frame"""
        for team_id in [0, 1]:
            positions = []
            for player in detections.players:
                if player.team_id == team_id:
                    x = player.center[0] / frame_width
                    y = player.center[1] / frame_height
                    positions.append((x, y))

            # Add goalkeeper
            for gk in detections.goalkeepers:
                if gk.team_id == team_id:
                    x = gk.center[0] / frame_width
                    y = gk.center[1] / frame_height
                    positions.append((x, y))

            if len(positions) >= 10:  # Need at least 10 outfield players
                self.position_samples[team_id].append(positions)

    def detect_formation(self, team_id: int) -> Tuple[str, float]:
        """
        Detect the most likely formation for a team.

        Returns:
            (formation_name, confidence)
        """
        samples = self.position_samples.get(team_id, [])
        if not samples:
            return ("Unknown", 0.0)

        # Average positions
        avg_positions = self._calculate_average_positions(samples)

        if len(avg_positions) < 10:
            return ("Unknown", 0.0)

        # Compare to known formations
        best_match = "Unknown"
        best_score = 0.0

        for formation_name, template in self.COMMON_FORMATIONS.items():
            score = self._match_formation(avg_positions, template)
            if score > best_score:
                best_score = score
                best_match = formation_name

        return (best_match, best_score)

    def _calculate_average_positions(
        self,
        samples: List[List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """Calculate average player positions from samples"""
        if not samples:
            return []

        # Use clustering to find consistent positions (sklearn version compatible)
        from sklearn.cluster import KMeans
        import sklearn

        all_positions = []
        for sample in samples:
            all_positions.extend(sample)

        if len(all_positions) < 11:
            return []

        # Cluster into 11 positions
        sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        if sklearn_version >= (1, 2):
            kmeans = KMeans(n_clusters=11, n_init=10, random_state=42, algorithm='lloyd')
        else:
            kmeans = KMeans(n_clusters=11, n_init=10, random_state=42)
        kmeans.fit(all_positions)

        return [tuple(c) for c in kmeans.cluster_centers_]

    def _match_formation(
        self,
        positions: List[Tuple[float, float]],
        template: List[Tuple[float, float]]
    ) -> float:
        """Calculate how well positions match a formation template"""
        if len(positions) != len(template):
            return 0.0

        # Sort positions by x coordinate (defense to attack)
        sorted_pos = sorted(positions, key=lambda p: p[0])

        # Convert template to normalized coordinates
        template_norm = [(t[0] / 5, t[1]) for t in template]
        template_sorted = sorted(template_norm, key=lambda p: p[0])

        # Calculate distance score
        total_dist = 0
        for pos, temp in zip(sorted_pos, template_sorted):
            dist = np.sqrt((pos[0] - temp[0])**2 + (pos[1] - temp[1])**2)
            total_dist += dist

        # Convert to similarity score (0-1)
        avg_dist = total_dist / len(positions)
        score = max(0, 1 - avg_dist * 2)

        return score


# ============================================
# Shot Detection & xG
# ============================================

class ShotDetector:
    """
    Detects shots and calculates expected goals (xG).
    """

    # xG model based on shot position (simplified)
    # Real xG models use more features like angle, defender positions, etc.

    def __init__(self):
        self.shots: List[ShotEvent] = []
        self.prev_ball_positions: List[Tuple[float, float]] = []

    def add_ball_position(self, x: float, y: float):
        """Track ball position for velocity calculation"""
        self.prev_ball_positions.append((x, y))
        if len(self.prev_ball_positions) > 10:
            self.prev_ball_positions.pop(0)

    def detect_shot(
        self,
        detections,
        frame_width: int,
        frame_height: int,
        goal_y_range: Tuple[float, float] = (0.35, 0.65)
    ) -> Optional[ShotEvent]:
        """
        Detect if a shot occurred based on ball movement toward goal.
        """
        if not detections.ball:
            return None

        ball_x = detections.ball.center[0] / frame_width
        ball_y = detections.ball.center[1] / frame_height

        self.add_ball_position(ball_x, ball_y)

        if len(self.prev_ball_positions) < 3:
            return None

        # Calculate ball velocity
        vx, vy = self._calculate_velocity()

        # Detect shot: high velocity toward goal area
        # Goal is at x=0 or x=1 depending on team
        is_shot = False
        shooting_team = None

        if abs(vx) > 0.05:  # Significant horizontal movement
            if vx < -0.05 and ball_x < 0.25:  # Moving toward left goal
                is_shot = True
                shooting_team = 1  # Away team shooting at home goal
            elif vx > 0.05 and ball_x > 0.75:  # Moving toward right goal
                is_shot = True
                shooting_team = 0  # Home team shooting at away goal

        if is_shot:
            # Find shooter (closest player to ball before shot)
            shooter = self._find_shooter(detections, shooting_team)

            shot = ShotEvent(
                player_id=shooter.tracker_id if shooter else -1,
                team_id=shooting_team,
                position=(ball_x * 100, ball_y * 100),
                frame=detections.frame_number,
                timestamp=detections.timestamp_seconds,
                outcome='pending',
                xg=self.calculate_xg(ball_x, ball_y, shooting_team)
            )
            self.shots.append(shot)
            return shot

        return None

    def _calculate_velocity(self) -> Tuple[float, float]:
        """Calculate ball velocity from recent positions"""
        if len(self.prev_ball_positions) < 2:
            return (0, 0)

        x1, y1 = self.prev_ball_positions[-2]
        x2, y2 = self.prev_ball_positions[-1]

        return (x2 - x1, y2 - y1)

    def _find_shooter(self, detections, team_id: int):
        """Find the player who likely took the shot"""
        if not detections.ball:
            return None

        ball_pos = detections.ball.center
        closest = None
        min_dist = float('inf')

        for player in detections.players:
            if player.team_id == team_id:
                dist = np.sqrt((player.center[0] - ball_pos[0])**2 +
                              (player.center[1] - ball_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest = player

        return closest

    def calculate_xg(self, x: float, y: float, team_id: int) -> float:
        """
        Calculate expected goals (xG) based on shot position.

        Simplified model - real xG uses many more features.

        Args:
            x, y: Position (0-1 normalized)
            team_id: Team taking the shot

        Returns:
            xG value (0-1)
        """
        # Determine which goal
        if team_id == 0:
            goal_x = 1.0  # Shooting at right goal
        else:
            goal_x = 0.0  # Shooting at left goal

        # Distance to goal center
        goal_y = 0.5
        distance = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

        # Angle to goal (simplified)
        angle = abs(y - 0.5)

        # Base xG from distance (exponential decay)
        base_xg = np.exp(-distance * 3)

        # Penalty for wide angles
        angle_factor = 1 - angle

        # Final xG
        xg = base_xg * angle_factor

        # Clamp to reasonable range
        return max(0.01, min(0.95, xg))

    def get_team_xg(self, team_id: int) -> float:
        """Get total xG for a team"""
        return sum(s.xg for s in self.shots if s.team_id == team_id)


# ============================================
# Speed & Distance Tracking
# ============================================

class SpeedDistanceTracker:
    """
    Tracks player speeds and distances covered.
    """

    def __init__(self, fps: float = 30.0, pixels_per_meter: float = 10.0):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.position_history: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)

    def add_positions(self, detections, frame_num: int):
        """Add player positions from a frame"""
        for player in detections.players + detections.goalkeepers:
            if player.tracker_id is not None:
                self.position_history[player.tracker_id].append(
                    (player.center[0], player.center[1], frame_num)
                )

    def get_distance_covered(self, tracker_id: int) -> float:
        """Get total distance covered by a player in meters"""
        positions = self.position_history.get(tracker_id, [])
        if len(positions) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(positions)):
            x1, y1, _ = positions[i-1]
            x2, y2, _ = positions[i]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += dist

        return total_distance / self.pixels_per_meter

    def get_max_speed(self, tracker_id: int) -> float:
        """Get maximum speed reached by a player in m/s"""
        positions = self.position_history.get(tracker_id, [])
        if len(positions) < 2:
            return 0.0

        max_speed = 0.0
        for i in range(1, len(positions)):
            x1, y1, f1 = positions[i-1]
            x2, y2, f2 = positions[i]

            if f2 == f1:
                continue

            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / self.pixels_per_meter
            time = (f2 - f1) / self.fps
            speed = dist / time

            max_speed = max(max_speed, speed)

        return max_speed

    def get_average_speed(self, tracker_id: int) -> float:
        """Get average speed of a player in m/s"""
        positions = self.position_history.get(tracker_id, [])
        if len(positions) < 2:
            return 0.0

        total_dist = self.get_distance_covered(tracker_id)
        first_frame = positions[0][2]
        last_frame = positions[-1][2]
        total_time = (last_frame - first_frame) / self.fps

        if total_time <= 0:
            return 0.0

        return total_dist / total_time

    def get_sprint_count(self, tracker_id: int, sprint_threshold: float = 7.0) -> int:
        """Count number of sprints (speed > threshold m/s)"""
        positions = self.position_history.get(tracker_id, [])
        if len(positions) < 2:
            return 0

        sprints = 0
        in_sprint = False

        for i in range(1, len(positions)):
            x1, y1, f1 = positions[i-1]
            x2, y2, f2 = positions[i]

            if f2 == f1:
                continue

            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / self.pixels_per_meter
            time = (f2 - f1) / self.fps
            speed = dist / time

            if speed > sprint_threshold and not in_sprint:
                sprints += 1
                in_sprint = True
            elif speed <= sprint_threshold:
                in_sprint = False

        return sprints

    def get_all_player_stats(self) -> Dict[int, Dict]:
        """Get stats for all tracked players"""
        stats = {}
        for tracker_id in self.position_history.keys():
            stats[tracker_id] = {
                "distance_m": round(self.get_distance_covered(tracker_id), 1),
                "max_speed_ms": round(self.get_max_speed(tracker_id), 2),
                "avg_speed_ms": round(self.get_average_speed(tracker_id), 2),
                "sprint_count": self.get_sprint_count(tracker_id),
                "distance_km": round(self.get_distance_covered(tracker_id) / 1000, 2)
            }
        return stats


# ============================================
# Possession Sequences
# ============================================

class PossessionSequenceTracker:
    """
    Tracks and analyzes possession sequences.
    """

    def __init__(self):
        self.sequences: List[PossessionSequence] = []
        self.current_sequence: Optional[PossessionSequence] = None
        self.current_team: Optional[int] = None

    def update(self, detections, possession_team: Optional[int]):
        """Update possession tracking"""
        if possession_team is None:
            # No clear possession - end current sequence
            if self.current_sequence:
                self.current_sequence.end_frame = detections.frame_number
                self.sequences.append(self.current_sequence)
                self.current_sequence = None
            return

        if possession_team != self.current_team:
            # Possession changed
            if self.current_sequence:
                self.current_sequence.end_frame = detections.frame_number
                self.sequences.append(self.current_sequence)

            # Start new sequence
            ball_pos = detections.ball.center if detections.ball else (0, 0)
            self.current_sequence = PossessionSequence(
                team_id=possession_team,
                start_frame=detections.frame_number,
                end_frame=detections.frame_number,
                start_position=ball_pos,
                end_position=ball_pos
            )
            self.current_team = possession_team
        else:
            # Continue current sequence
            if self.current_sequence and detections.ball:
                self.current_sequence.end_position = detections.ball.center

    def get_sequence_stats(self, team_id: Optional[int] = None) -> Dict:
        """Get possession sequence statistics"""
        seqs = [s for s in self.sequences if team_id is None or s.team_id == team_id]

        if not seqs:
            return {"total_sequences": 0}

        durations = [s.end_frame - s.start_frame for s in seqs]

        return {
            "total_sequences": len(seqs),
            "avg_duration_frames": np.mean(durations),
            "max_duration_frames": max(durations),
            "sequences_with_shot": sum(1 for s in seqs if s.shot is not None),
            "total_passes": sum(len(s.passes) for s in seqs)
        }


# ============================================
# Offside Line Visualization
# ============================================

class OffsideLine:
    """
    Calculates and draws offside line based on second-last defender.
    """

    def __init__(self):
        pass

    def calculate_offside_line(
        self,
        detections,
        defending_team: int,
        frame_width: int
    ) -> Optional[float]:
        """
        Calculate offside line position (x coordinate).

        The offside line is the position of the second-last defender
        (last defender is usually the goalkeeper).
        """
        defenders = []

        for player in detections.players + detections.goalkeepers:
            if player.team_id == defending_team:
                defenders.append(player.center[0])

        if len(defenders) < 2:
            return None

        # Sort by x position
        defenders.sort()

        # For team 0 (defending left goal), offside is at second-rightmost
        # For team 1 (defending right goal), offside is at second-leftmost
        if defending_team == 0:
            return defenders[-2]  # Second highest x
        else:
            return defenders[1]   # Second lowest x

    def draw_offside_line(
        self,
        frame: np.ndarray,
        offside_x: float,
        color: Tuple[int, int, int] = (255, 0, 255)
    ) -> np.ndarray:
        """Draw offside line on frame"""
        h, w = frame.shape[:2]
        x = int(offside_x)
        cv2.line(frame, (x, 0), (x, h), color, 2)
        cv2.putText(frame, "OFFSIDE LINE", (x + 5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
