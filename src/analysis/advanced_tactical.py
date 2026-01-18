"""
Soccer Film Analysis - Advanced Tactical Analysis
Team shape, duels, passing lanes, xT, view transformation, and more
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle, FancyArrowPatch
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial import ConvexHull, Delaunay
    from scipy.ndimage import gaussian_filter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ==================== TEAM SHAPE ANALYSIS ====================

@dataclass
class TeamShapeMetrics:
    """Metrics describing team shape"""
    frame: int
    team_id: int

    # Dimensions (in meters, normalized to pitch size)
    width: float = 0.0  # Horizontal spread
    length: float = 0.0  # Vertical spread (depth)
    area: float = 0.0  # Convex hull area

    # Compactness
    compactness: float = 0.0  # 0-1, higher = more compact
    avg_player_distance: float = 0.0  # Average distance between players

    # Center of mass
    centroid_x: float = 0.0
    centroid_y: float = 0.0

    # Line metrics
    defensive_line_y: float = 0.0
    offensive_line_y: float = 0.0
    midfield_line_y: float = 0.0

    # Shape alerts
    is_stretched: bool = False
    is_compact: bool = False
    defensive_line_broken: bool = False


class TeamShapeAnalyzer:
    """
    Analyzes team shape, compactness, width, and length throughout the match.
    Generates alerts when shape is compromised.
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        compact_threshold: float = 25.0,  # meters
        stretched_threshold: float = 45.0  # meters
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.compact_threshold = compact_threshold
        self.stretched_threshold = stretched_threshold

        # Historical data
        self.home_shapes: List[TeamShapeMetrics] = []
        self.away_shapes: List[TeamShapeMetrics] = []

        # Alert tracking
        self.shape_alerts: List[Dict] = []

    def analyze_frame(
        self,
        frame_num: int,
        home_positions: List[Tuple[float, float]],
        away_positions: List[Tuple[float, float]]
    ) -> Tuple[TeamShapeMetrics, TeamShapeMetrics]:
        """
        Analyze team shapes for a single frame.

        Args:
            frame_num: Current frame number
            home_positions: List of (x, y) normalized positions for home team
            away_positions: List of (x, y) normalized positions for away team

        Returns:
            Tuple of (home_metrics, away_metrics)
        """
        home_metrics = self._calculate_shape(frame_num, 0, home_positions)
        away_metrics = self._calculate_shape(frame_num, 1, away_positions)

        self.home_shapes.append(home_metrics)
        self.away_shapes.append(away_metrics)

        # Check for alerts
        self._check_alerts(home_metrics)
        self._check_alerts(away_metrics)

        return home_metrics, away_metrics

    def _calculate_shape(
        self,
        frame: int,
        team_id: int,
        positions: List[Tuple[float, float]]
    ) -> TeamShapeMetrics:
        """Calculate shape metrics for a team"""
        if len(positions) < 3:
            return TeamShapeMetrics(frame=frame, team_id=team_id)

        positions_array = np.array(positions)

        # Convert to meters
        positions_meters = positions_array * np.array([self.pitch_length, self.pitch_width])

        # Centroid
        centroid = np.mean(positions_meters, axis=0)

        # Width and length
        width = np.max(positions_meters[:, 1]) - np.min(positions_meters[:, 1])
        length = np.max(positions_meters[:, 0]) - np.min(positions_meters[:, 0])

        # Convex hull area
        try:
            hull = ConvexHull(positions_meters)
            area = hull.volume  # In 2D, volume gives area
        except Exception:
            area = width * length * 0.5

        # Average player distance
        distances = []
        for i, p1 in enumerate(positions_meters):
            for p2 in positions_meters[i+1:]:
                distances.append(np.linalg.norm(p1 - p2))
        avg_distance = np.mean(distances) if distances else 0

        # Compactness (inverse of spread, normalized)
        max_spread = np.sqrt(self.pitch_length**2 + self.pitch_width**2)
        spread = np.sqrt(width**2 + length**2)
        compactness = 1.0 - min(1.0, spread / max_spread)

        # Line positions (sorted by x position)
        sorted_x = np.sort(positions_meters[:, 0])
        if len(sorted_x) >= 4:
            defensive_line = np.mean(sorted_x[:4])  # Back 4
            offensive_line = np.mean(sorted_x[-3:])  # Front 3
            midfield_line = np.mean(sorted_x[4:-3]) if len(sorted_x) > 7 else np.mean(sorted_x)
        else:
            defensive_line = sorted_x[0] if len(sorted_x) > 0 else 0
            offensive_line = sorted_x[-1] if len(sorted_x) > 0 else 0
            midfield_line = np.mean(sorted_x)

        # Shape status
        is_compact = length < self.compact_threshold
        is_stretched = length > self.stretched_threshold

        # Defensive line broken check (large gap between defenders)
        defensive_line_broken = False
        if len(sorted_x) >= 4:
            defender_positions = positions_meters[np.argsort(positions_meters[:, 0])[:4]]
            defender_gaps = np.diff(np.sort(defender_positions[:, 1]))
            if len(defender_gaps) > 0 and np.max(defender_gaps) > 15:  # 15m gap
                defensive_line_broken = True

        return TeamShapeMetrics(
            frame=frame,
            team_id=team_id,
            width=width,
            length=length,
            area=area,
            compactness=compactness,
            avg_player_distance=avg_distance,
            centroid_x=centroid[0] / self.pitch_length,
            centroid_y=centroid[1] / self.pitch_width,
            defensive_line_y=defensive_line / self.pitch_length,
            offensive_line_y=offensive_line / self.pitch_length,
            midfield_line_y=midfield_line / self.pitch_length,
            is_stretched=is_stretched,
            is_compact=is_compact,
            defensive_line_broken=defensive_line_broken
        )

    def _check_alerts(self, metrics: TeamShapeMetrics):
        """Check for shape-related alerts"""
        if metrics.is_stretched:
            self.shape_alerts.append({
                'frame': metrics.frame,
                'team_id': metrics.team_id,
                'alert_type': 'stretched',
                'message': f"Team stretched to {metrics.length:.1f}m length",
                'severity': 'warning'
            })

        if metrics.defensive_line_broken:
            self.shape_alerts.append({
                'frame': metrics.frame,
                'team_id': metrics.team_id,
                'alert_type': 'defensive_gap',
                'message': "Large gap in defensive line",
                'severity': 'high'
            })

    def get_average_shape(self, team_id: int) -> Dict:
        """Get average shape metrics for a team"""
        shapes = self.home_shapes if team_id == 0 else self.away_shapes
        if not shapes:
            return {}

        return {
            'avg_width': np.mean([s.width for s in shapes]),
            'avg_length': np.mean([s.length for s in shapes]),
            'avg_compactness': np.mean([s.compactness for s in shapes]),
            'avg_area': np.mean([s.area for s in shapes]),
            'stretched_percentage': sum(1 for s in shapes if s.is_stretched) / len(shapes) * 100,
            'compact_percentage': sum(1 for s in shapes if s.is_compact) / len(shapes) * 100
        }

    def visualize_shape(
        self,
        positions: List[Tuple[float, float]],
        team_id: int,
        save_path: Optional[str] = None
    ):
        """Create visualization of team shape"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.65)
        ax.set_facecolor('#2e7d32')

        # Draw pitch lines
        ax.plot([0, 1], [0, 0], 'w-', linewidth=2)
        ax.plot([0, 1], [0.65, 0.65], 'w-', linewidth=2)
        ax.plot([0, 0], [0, 0.65], 'w-', linewidth=2)
        ax.plot([1, 1], [0, 0.65], 'w-', linewidth=2)
        ax.plot([0.5, 0.5], [0, 0.65], 'w-', linewidth=1)

        if len(positions) >= 3:
            positions_array = np.array(positions)

            # Draw convex hull
            try:
                hull = ConvexHull(positions_array)
                hull_points = positions_array[hull.vertices]
                hull_polygon = Polygon(hull_points, alpha=0.3,
                                       facecolor='yellow' if team_id == 0 else 'red',
                                       edgecolor='white', linewidth=2)
                ax.add_patch(hull_polygon)
            except Exception:
                pass

            # Draw players
            color = 'yellow' if team_id == 0 else 'red'
            ax.scatter(positions_array[:, 0], positions_array[:, 1],
                       c=color, s=200, edgecolors='white', linewidths=2, zorder=5)

            # Draw centroid
            centroid = np.mean(positions_array, axis=0)
            ax.scatter([centroid[0]], [centroid[1]], c='white', s=100,
                       marker='x', linewidths=3, zorder=6)

        ax.set_title("Team Shape Analysis", fontsize=14)
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')

        return fig


# ==================== DUEL/CHALLENGE STATISTICS ====================

class DuelType(Enum):
    AERIAL = "aerial"
    GROUND = "ground"
    TACKLE = "tackle"
    DRIBBLE = "dribble"


@dataclass
class Duel:
    """Represents a 1v1 duel between players"""
    frame: int
    timestamp_seconds: float
    duel_type: DuelType
    attacker_id: int
    attacker_team: int
    defender_id: int
    defender_team: int
    location: Tuple[float, float]
    winner_id: Optional[int] = None
    winner_team: Optional[int] = None


class DuelAnalyzer:
    """
    Tracks and analyzes 1v1 duels (aerial, ground, tackles).
    """

    def __init__(self, proximity_threshold: float = 0.02):
        """
        Args:
            proximity_threshold: Normalized distance for duel detection
        """
        self.proximity_threshold = proximity_threshold
        self.duels: List[Duel] = []

        # Per-player stats
        self.player_duels: Dict[int, Dict] = defaultdict(lambda: {
            'aerial_won': 0, 'aerial_lost': 0,
            'ground_won': 0, 'ground_lost': 0,
            'tackles_won': 0, 'tackles_lost': 0,
            'dribbles_won': 0, 'dribbles_lost': 0
        })

    def detect_duels(
        self,
        frame: int,
        timestamp: float,
        home_players: List[Dict],  # {'id': int, 'x': float, 'y': float, 'has_ball': bool}
        away_players: List[Dict],
        ball_position: Optional[Tuple[float, float]] = None
    ) -> List[Duel]:
        """Detect duels in the current frame"""
        detected_duels = []

        # Find close player pairs from opposing teams
        for home_p in home_players:
            for away_p in away_players:
                dist = np.sqrt(
                    (home_p['x'] - away_p['x'])**2 +
                    (home_p['y'] - away_p['y'])**2
                )

                if dist < self.proximity_threshold:
                    # Determine duel type
                    attacker = home_p if home_p.get('has_ball') else away_p
                    defender = away_p if home_p.get('has_ball') else home_p

                    if not attacker.get('has_ball') and not defender.get('has_ball'):
                        # Neither has ball - likely aerial duel
                        duel_type = DuelType.AERIAL
                    elif attacker.get('has_ball'):
                        duel_type = DuelType.DRIBBLE
                    else:
                        duel_type = DuelType.TACKLE

                    duel = Duel(
                        frame=frame,
                        timestamp_seconds=timestamp,
                        duel_type=duel_type,
                        attacker_id=attacker['id'],
                        attacker_team=0 if attacker in home_players else 1,
                        defender_id=defender['id'],
                        defender_team=0 if defender in home_players else 1,
                        location=(home_p['x'], home_p['y'])
                    )
                    detected_duels.append(duel)

        return detected_duels

    def record_duel_result(
        self,
        duel: Duel,
        winner_id: int,
        winner_team: int
    ):
        """Record the outcome of a duel"""
        duel.winner_id = winner_id
        duel.winner_team = winner_team
        self.duels.append(duel)

        # Update player stats
        duel_key = f"{duel.duel_type.value}_won"
        duel_lost_key = f"{duel.duel_type.value}_lost"

        if winner_id == duel.attacker_id:
            self.player_duels[duel.attacker_id][duel_key] += 1
            self.player_duels[duel.defender_id][duel_lost_key] += 1
        else:
            self.player_duels[duel.defender_id][duel_key] += 1
            self.player_duels[duel.attacker_id][duel_lost_key] += 1

    def get_player_duel_stats(self, player_id: int) -> Dict:
        """Get duel statistics for a player"""
        stats = self.player_duels.get(player_id, {})

        total_aerial = stats.get('aerial_won', 0) + stats.get('aerial_lost', 0)
        total_ground = stats.get('ground_won', 0) + stats.get('ground_lost', 0)
        total_duels = total_aerial + total_ground

        return {
            'total_duels': total_duels,
            'duels_won': stats.get('aerial_won', 0) + stats.get('ground_won', 0) +
                         stats.get('tackles_won', 0) + stats.get('dribbles_won', 0),
            'aerial_duels': total_aerial,
            'aerial_win_rate': stats.get('aerial_won', 0) / max(1, total_aerial) * 100,
            'ground_duels': total_ground,
            'ground_win_rate': stats.get('ground_won', 0) / max(1, total_ground) * 100,
            'tackles_won': stats.get('tackles_won', 0),
            'dribbles_completed': stats.get('dribbles_won', 0)
        }

    def get_team_duel_stats(self, team_id: int) -> Dict:
        """Get duel statistics for a team"""
        team_duels = [d for d in self.duels
                      if d.attacker_team == team_id or d.defender_team == team_id]

        won = sum(1 for d in team_duels if d.winner_team == team_id)
        total = len(team_duels)

        aerial_duels = [d for d in team_duels if d.duel_type == DuelType.AERIAL]
        aerial_won = sum(1 for d in aerial_duels if d.winner_team == team_id)

        return {
            'total_duels': total,
            'duels_won': won,
            'win_rate': won / max(1, total) * 100,
            'aerial_duels': len(aerial_duels),
            'aerial_win_rate': aerial_won / max(1, len(aerial_duels)) * 100
        }


# ==================== PASSING LANE BLOCKING ====================

@dataclass
class PassingLane:
    """Represents a potential passing lane"""
    passer_id: int
    receiver_id: int
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    is_blocked: bool = False
    blocking_player_id: Optional[int] = None
    lane_quality: float = 1.0  # 0-1, lower = more blocked


class PassingLaneAnalyzer:
    """
    Analyzes how well defenders cut off passing options.
    """

    def __init__(self, blocking_distance: float = 0.03):
        """
        Args:
            blocking_distance: How close a defender must be to block a lane
        """
        self.blocking_distance = blocking_distance
        self.blocked_lanes: List[Dict] = []
        self.open_lanes: List[Dict] = []

    def analyze_passing_lanes(
        self,
        frame: int,
        attacking_players: List[Dict],  # {'id': int, 'x': float, 'y': float}
        defending_players: List[Dict],
        ball_carrier_id: int
    ) -> List[PassingLane]:
        """
        Analyze all passing lanes from the ball carrier.

        Returns:
            List of PassingLane objects
        """
        lanes = []
        ball_carrier = next((p for p in attacking_players if p['id'] == ball_carrier_id), None)
        if not ball_carrier:
            return lanes

        for teammate in attacking_players:
            if teammate['id'] == ball_carrier_id:
                continue

            lane = PassingLane(
                passer_id=ball_carrier_id,
                receiver_id=teammate['id'],
                start_pos=(ball_carrier['x'], ball_carrier['y']),
                end_pos=(teammate['x'], teammate['y'])
            )

            # Check if any defender blocks this lane
            lane_blocked, blocker_id, quality = self._check_lane_blocked(
                lane.start_pos, lane.end_pos, defending_players
            )

            lane.is_blocked = lane_blocked
            lane.blocking_player_id = blocker_id
            lane.lane_quality = quality

            lanes.append(lane)

            # Record for statistics
            if lane_blocked:
                self.blocked_lanes.append({
                    'frame': frame,
                    'passer': ball_carrier_id,
                    'receiver': teammate['id'],
                    'blocker': blocker_id
                })
            else:
                self.open_lanes.append({
                    'frame': frame,
                    'passer': ball_carrier_id,
                    'receiver': teammate['id'],
                    'quality': quality
                })

        return lanes

    def _check_lane_blocked(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        defenders: List[Dict]
    ) -> Tuple[bool, Optional[int], float]:
        """Check if a passing lane is blocked"""
        lane_vector = np.array([end[0] - start[0], end[1] - start[1]])
        lane_length = np.linalg.norm(lane_vector)

        if lane_length < 0.001:
            return False, None, 1.0

        lane_unit = lane_vector / lane_length

        min_distance = float('inf')
        closest_blocker = None

        for defender in defenders:
            # Vector from start to defender
            to_defender = np.array([
                defender['x'] - start[0],
                defender['y'] - start[1]
            ])

            # Project onto lane
            projection = np.dot(to_defender, lane_unit)

            # Only consider defenders between passer and receiver
            if projection < 0 or projection > lane_length:
                continue

            # Perpendicular distance to lane
            projected_point = np.array(start) + projection * lane_unit
            distance = np.linalg.norm(
                np.array([defender['x'], defender['y']]) - projected_point
            )

            if distance < min_distance:
                min_distance = distance
                closest_blocker = defender['id']

        is_blocked = min_distance < self.blocking_distance
        quality = min(1.0, min_distance / self.blocking_distance) if min_distance < float('inf') else 1.0

        return is_blocked, closest_blocker if is_blocked else None, quality

    def get_blocking_stats(self, team_id: int, player_stats: Dict[int, int]) -> Dict:
        """Get passing lane blocking statistics"""
        # player_stats maps player_id to team_id
        team_blocks = [b for b in self.blocked_lanes
                       if player_stats.get(b['blocker']) == team_id]

        player_blocks = defaultdict(int)
        for block in team_blocks:
            player_blocks[block['blocker']] += 1

        return {
            'total_lanes_blocked': len(team_blocks),
            'top_blockers': sorted(player_blocks.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def visualize_lanes(
        self,
        lanes: List[PassingLane],
        defending_players: List[Dict],
        save_path: Optional[str] = None
    ):
        """Visualize passing lanes"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.65)
        ax.set_facecolor('#2e7d32')

        # Draw passing lanes
        for lane in lanes:
            color = 'red' if lane.is_blocked else 'lime'
            alpha = 0.3 if lane.is_blocked else 0.7
            ax.plot([lane.start_pos[0], lane.end_pos[0]],
                    [lane.start_pos[1], lane.end_pos[1]],
                    color=color, linewidth=2, alpha=alpha)

        # Draw defenders
        for defender in defending_players:
            ax.scatter(defender['x'], defender['y'], c='blue', s=150,
                       edgecolors='white', linewidths=2, zorder=5)

        ax.set_title("Passing Lane Analysis", fontsize=14)
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')

        return fig


# ==================== EXPECTED THREAT (xT) MODEL ====================

class ExpectedThreatModel:
    """
    Expected Threat (xT) model for valuing ball movement.
    Based on the concept that pitch positions have different goal-scoring probabilities.
    """

    # Pre-computed xT grid (12x8 zones)
    # Values represent the probability of scoring from each zone
    XT_GRID = np.array([
        [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.0174506, 0.02122129, 0.02812427, 0.03934859, 0.05494619],
        [0.00750072, 0.00878589, 0.00942382, 0.0105949, 0.01214719, 0.0138454, 0.01611813, 0.01870347, 0.02401521, 0.03545096, 0.05346447, 0.0817432],
        [0.00764987, 0.00904991, 0.00991452, 0.0110948, 0.01262646, 0.01457036, 0.01692039, 0.01982545, 0.02616179, 0.04046498, 0.06590185, 0.11202441],
        [0.00760421, 0.00896527, 0.00977354, 0.01089061, 0.01220282, 0.01411579, 0.0163923, 0.01949896, 0.02613893, 0.04196925, 0.07125946, 0.12958475],
        [0.00759691, 0.00895764, 0.00972685, 0.01085533, 0.01218254, 0.01406879, 0.01637021, 0.01946346, 0.02622361, 0.04225135, 0.07198966, 0.13138691],
        [0.00755102, 0.00895855, 0.00975024, 0.01089831, 0.01232377, 0.01427204, 0.01653789, 0.01943535, 0.02578946, 0.04051479, 0.06671277, 0.11459224],
        [0.0074109, 0.00876494, 0.00944478, 0.01055353, 0.01207726, 0.01381838, 0.01610005, 0.01866743, 0.02397444, 0.03552021, 0.05382859, 0.08320631],
        [0.00632697, 0.00774832, 0.00842208, 0.00974663, 0.01127202, 0.01252514, 0.01480497, 0.01749795, 0.02127918, 0.02816771, 0.03944794, 0.05549316]
    ])

    def __init__(self):
        self.grid_rows = 8
        self.grid_cols = 12

        # Movement values (xT gained by moving the ball)
        self.movement_log: List[Dict] = []

    def get_xt_value(self, x: float, y: float) -> float:
        """
        Get xT value for a position.

        Args:
            x: Normalized x position (0-1, left to right)
            y: Normalized y position (0-1, bottom to top)

        Returns:
            xT value (probability of scoring from this position)
        """
        col = min(int(x * self.grid_cols), self.grid_cols - 1)
        row = min(int(y * self.grid_rows), self.grid_rows - 1)
        return self.XT_GRID[row, col]

    def calculate_xt_gained(
        self,
        start_x: float, start_y: float,
        end_x: float, end_y: float
    ) -> float:
        """
        Calculate xT gained from a ball movement.

        Args:
            start_x, start_y: Starting position
            end_x, end_y: Ending position

        Returns:
            xT difference (positive = threat increased)
        """
        start_xt = self.get_xt_value(start_x, start_y)
        end_xt = self.get_xt_value(end_x, end_y)
        return end_xt - start_xt

    def record_movement(
        self,
        frame: int,
        player_id: int,
        team_id: int,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        action_type: str  # 'pass', 'carry', 'dribble'
    ):
        """Record a ball movement for xT tracking"""
        xt_gained = self.calculate_xt_gained(
            start_pos[0], start_pos[1],
            end_pos[0], end_pos[1]
        )

        self.movement_log.append({
            'frame': frame,
            'player_id': player_id,
            'team_id': team_id,
            'start': start_pos,
            'end': end_pos,
            'action': action_type,
            'xt_gained': xt_gained,
            'start_xt': self.get_xt_value(*start_pos),
            'end_xt': self.get_xt_value(*end_pos)
        })

    def get_player_xt_stats(self, player_id: int) -> Dict:
        """Get xT statistics for a player"""
        player_movements = [m for m in self.movement_log if m['player_id'] == player_id]

        if not player_movements:
            return {'total_xt_gained': 0, 'movements': 0}

        total_xt = sum(m['xt_gained'] for m in player_movements)
        positive_moves = sum(1 for m in player_movements if m['xt_gained'] > 0)

        return {
            'total_xt_gained': total_xt,
            'movements': len(player_movements),
            'avg_xt_per_action': total_xt / len(player_movements),
            'positive_movements': positive_moves,
            'positive_rate': positive_moves / len(player_movements) * 100
        }

    def get_team_xt_stats(self, team_id: int) -> Dict:
        """Get xT statistics for a team"""
        team_movements = [m for m in self.movement_log if m['team_id'] == team_id]

        if not team_movements:
            return {'total_xt_gained': 0}

        total_xt = sum(m['xt_gained'] for m in team_movements)
        by_action = defaultdict(float)
        for m in team_movements:
            by_action[m['action']] += m['xt_gained']

        return {
            'total_xt_gained': total_xt,
            'total_movements': len(team_movements),
            'xt_from_passes': by_action.get('pass', 0),
            'xt_from_carries': by_action.get('carry', 0),
            'xt_from_dribbles': by_action.get('dribble', 0)
        }

    def visualize_xt_grid(self, save_path: Optional[str] = None):
        """Visualize the xT grid"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create heatmap
        im = ax.imshow(self.XT_GRID, cmap='RdYlGn', aspect='auto',
                       extent=[0, 1, 0, 0.65], origin='lower')

        # Add pitch lines
        ax.axvline(x=0.5, color='white', linewidth=2, alpha=0.7)

        plt.colorbar(im, ax=ax, label='xT Value')
        ax.set_title("Expected Threat (xT) Grid", fontsize=14)
        ax.set_xlabel("Pitch Length")
        ax.set_ylabel("Pitch Width")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


# ==================== BIRD'S EYE VIEW TRANSFORMATION ====================

class BirdsEyeViewTransformer:
    """
    Transforms broadcast camera view to tactical bird's eye view.
    Uses homography transformation with calibration points.
    """

    def __init__(
        self,
        output_width: int = 1050,
        output_height: int = 680,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0
    ):
        """
        Args:
            output_width: Output image width in pixels
            output_height: Output image height in pixels
            pitch_length: Real pitch length in meters
            pitch_width: Real pitch width in meters
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for view transformation")

        self.output_width = output_width
        self.output_height = output_height
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        self.homography_matrix: Optional[np.ndarray] = None
        self.calibration_points: List[Tuple] = []

        # Pixels per meter
        self.scale_x = output_width / pitch_length
        self.scale_y = output_height / pitch_width

    def calibrate(
        self,
        source_points: List[Tuple[float, float]],
        destination_points: List[Tuple[float, float]]
    ) -> bool:
        """
        Calibrate the transformation using known point correspondences.

        Args:
            source_points: Points in the source (broadcast) image
            destination_points: Corresponding points in real-world coordinates (meters)

        Returns:
            True if calibration successful
        """
        if len(source_points) < 4 or len(destination_points) < 4:
            logger.error("Need at least 4 points for calibration")
            return False

        # Convert destination points to pixel coordinates
        dst_pixels = [
            (p[0] * self.scale_x, p[1] * self.scale_y)
            for p in destination_points
        ]

        src = np.float32(source_points)
        dst = np.float32(dst_pixels)

        self.homography_matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if self.homography_matrix is None:
            logger.error("Failed to compute homography")
            return False

        self.calibration_points = list(zip(source_points, destination_points))
        logger.info("View transformation calibrated successfully")
        return True

    def auto_calibrate_from_lines(self, frame: np.ndarray) -> bool:
        """
        Attempt automatic calibration by detecting pitch lines.

        Args:
            frame: Input video frame

        Returns:
            True if auto-calibration successful
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect lines using Hough transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                                minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) < 4:
            logger.warning("Could not detect enough lines for auto-calibration")
            return False

        # Find intersection points (simplified - would need more robust detection)
        # This is a placeholder - real implementation would detect specific pitch features
        logger.warning("Auto-calibration requires manual point selection for accuracy")
        return False

    def transform_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform a frame to bird's eye view.

        Args:
            frame: Input video frame

        Returns:
            Transformed frame or None if not calibrated
        """
        if self.homography_matrix is None:
            logger.warning("Transform not calibrated")
            return None

        output = cv2.warpPerspective(
            frame, self.homography_matrix,
            (self.output_width, self.output_height)
        )

        return output

    def transform_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Transform a single point to bird's eye coordinates.

        Args:
            x, y: Point in source image

        Returns:
            Transformed (x, y) in meters, or None if not calibrated
        """
        if self.homography_matrix is None:
            return None

        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)

        tx = transformed[0][0][0] / self.scale_x
        ty = transformed[0][0][1] / self.scale_y

        return (tx, ty)

    def transform_detections(
        self,
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Transform player detections to bird's eye coordinates.

        Args:
            detections: List of detection dicts with 'x', 'y' keys

        Returns:
            Detections with transformed coordinates
        """
        transformed = []
        for det in detections:
            result = self.transform_point(det['x'], det['y'])
            if result:
                det_copy = det.copy()
                det_copy['x_world'] = result[0]
                det_copy['y_world'] = result[1]
                transformed.append(det_copy)

        return transformed

    def draw_tactical_view(
        self,
        home_positions: List[Tuple[float, float]],
        away_positions: List[Tuple[float, float]],
        ball_position: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Draw a clean tactical view with player positions.

        Args:
            home_positions: Home team positions in meters
            away_positions: Away team positions in meters
            ball_position: Ball position in meters

        Returns:
            Tactical view image
        """
        # Create green pitch
        output = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        output[:] = (34, 139, 34)  # Forest green

        # Draw pitch markings
        self._draw_pitch_markings(output)

        # Draw players
        for pos in home_positions:
            px = int(pos[0] * self.scale_x)
            py = int(pos[1] * self.scale_y)
            cv2.circle(output, (px, py), 12, (0, 255, 255), -1)  # Yellow
            cv2.circle(output, (px, py), 12, (255, 255, 255), 2)

        for pos in away_positions:
            px = int(pos[0] * self.scale_x)
            py = int(pos[1] * self.scale_y)
            cv2.circle(output, (px, py), 12, (0, 0, 255), -1)  # Red
            cv2.circle(output, (px, py), 12, (255, 255, 255), 2)

        # Draw ball
        if ball_position:
            bx = int(ball_position[0] * self.scale_x)
            by = int(ball_position[1] * self.scale_y)
            cv2.circle(output, (bx, by), 8, (255, 255, 255), -1)

        return output

    def _draw_pitch_markings(self, img: np.ndarray):
        """Draw pitch markings on the tactical view"""
        white = (255, 255, 255)
        thickness = 2

        w, h = self.output_width, self.output_height

        # Outer boundary
        cv2.rectangle(img, (0, 0), (w-1, h-1), white, thickness)

        # Center line
        cv2.line(img, (w//2, 0), (w//2, h), white, thickness)

        # Center circle (radius ~9.15m)
        center_radius = int(9.15 * self.scale_x)
        cv2.circle(img, (w//2, h//2), center_radius, white, thickness)

        # Penalty areas (16.5m x 40.32m)
        pa_width = int(16.5 * self.scale_x)
        pa_height = int(40.32 * self.scale_y)
        pa_top = (h - pa_height) // 2

        # Left penalty area
        cv2.rectangle(img, (0, pa_top), (pa_width, pa_top + pa_height), white, thickness)
        # Right penalty area
        cv2.rectangle(img, (w - pa_width, pa_top), (w, pa_top + pa_height), white, thickness)

        # Goal areas (5.5m x 18.32m)
        ga_width = int(5.5 * self.scale_x)
        ga_height = int(18.32 * self.scale_y)
        ga_top = (h - ga_height) // 2

        cv2.rectangle(img, (0, ga_top), (ga_width, ga_top + ga_height), white, thickness)
        cv2.rectangle(img, (w - ga_width, ga_top), (w, ga_top + ga_height), white, thickness)


# ==================== OPPONENT TENDENCY PREDICTION ====================

@dataclass
class TendencyPattern:
    """A detected tendency pattern"""
    pattern_type: str  # 'passing_lane', 'attacking_side', 'set_piece', 'pressing_trigger'
    description: str
    confidence: float  # 0-1
    occurrences: int
    examples: List[Dict] = field(default_factory=list)


class OpponentTendencyAnalyzer:
    """
    Uses historical data to predict opponent patterns.
    """

    def __init__(self):
        # Tracked patterns
        self.passing_patterns: Dict[str, int] = defaultdict(int)  # zone->zone counts
        self.attacking_side_counts = {'left': 0, 'center': 0, 'right': 0}
        self.set_piece_outcomes: Dict[str, List[str]] = defaultdict(list)
        self.pressing_triggers: List[Dict] = []

        # Detected tendencies
        self.tendencies: List[TendencyPattern] = []

    def record_pass(self, from_zone: str, to_zone: str, team_id: int):
        """Record a pass for pattern detection"""
        key = f"{from_zone}->{to_zone}"
        self.passing_patterns[key] += 1

    def record_attack(self, side: str, team_id: int):
        """Record attack side preference"""
        if side in self.attacking_side_counts:
            self.attacking_side_counts[side] += 1

    def record_set_piece(self, set_piece_type: str, delivery: str, outcome: str):
        """Record set piece routine"""
        key = f"{set_piece_type}_{delivery}"
        self.set_piece_outcomes[key].append(outcome)

    def record_pressing_trigger(self, trigger_zone: str, trigger_event: str, success: bool):
        """Record when opponent initiates press"""
        self.pressing_triggers.append({
            'zone': trigger_zone,
            'event': trigger_event,
            'success': success
        })

    def analyze_tendencies(self, min_occurrences: int = 5) -> List[TendencyPattern]:
        """Analyze collected data and identify tendencies"""
        self.tendencies = []

        # Passing lane preferences
        total_passes = sum(self.passing_patterns.values())
        if total_passes > 0:
            for pattern, count in sorted(self.passing_patterns.items(),
                                          key=lambda x: x[1], reverse=True)[:3]:
                if count >= min_occurrences:
                    percentage = count / total_passes * 100
                    self.tendencies.append(TendencyPattern(
                        pattern_type='passing_lane',
                        description=f"Frequently passes {pattern} ({percentage:.1f}%)",
                        confidence=min(0.9, percentage / 100 * 2),
                        occurrences=count
                    ))

        # Attacking side preference
        total_attacks = sum(self.attacking_side_counts.values())
        if total_attacks > min_occurrences:
            preferred_side = max(self.attacking_side_counts.items(), key=lambda x: x[1])
            preference_pct = preferred_side[1] / total_attacks * 100
            if preference_pct > 40:  # Clear preference
                self.tendencies.append(TendencyPattern(
                    pattern_type='attacking_side',
                    description=f"Prefers attacking down the {preferred_side[0]} ({preference_pct:.1f}%)",
                    confidence=min(0.9, preference_pct / 100 * 1.5),
                    occurrences=preferred_side[1]
                ))

        # Set piece patterns
        for routine, outcomes in self.set_piece_outcomes.items():
            if len(outcomes) >= min_occurrences:
                goals = sum(1 for o in outcomes if o == 'goal')
                shots = sum(1 for o in outcomes if o in ['goal', 'shot'])
                self.tendencies.append(TendencyPattern(
                    pattern_type='set_piece',
                    description=f"{routine.replace('_', ' ').title()}: {shots}/{len(outcomes)} shots, {goals} goals",
                    confidence=0.7,
                    occurrences=len(outcomes)
                ))

        # Pressing triggers
        if len(self.pressing_triggers) >= min_occurrences:
            zone_counts = defaultdict(int)
            for trigger in self.pressing_triggers:
                zone_counts[trigger['zone']] += 1

            top_zone = max(zone_counts.items(), key=lambda x: x[1])
            success_rate = sum(1 for t in self.pressing_triggers
                               if t['zone'] == top_zone[0] and t['success']) / top_zone[1]

            self.tendencies.append(TendencyPattern(
                pattern_type='pressing_trigger',
                description=f"Often presses when ball in {top_zone[0]} ({success_rate*100:.0f}% success)",
                confidence=0.6,
                occurrences=top_zone[1]
            ))

        return self.tendencies

    def get_scouting_report(self) -> str:
        """Generate a text scouting report"""
        if not self.tendencies:
            self.analyze_tendencies()

        report = ["OPPONENT TENDENCIES REPORT", "=" * 40, ""]

        for tendency in sorted(self.tendencies, key=lambda t: t.confidence, reverse=True):
            confidence_str = "HIGH" if tendency.confidence > 0.7 else "MEDIUM" if tendency.confidence > 0.5 else "LOW"
            report.append(f"[{confidence_str}] {tendency.pattern_type.upper()}")
            report.append(f"    {tendency.description}")
            report.append(f"    Based on {tendency.occurrences} observations")
            report.append("")

        return "\n".join(report)


# ==================== PLAYER FATIGUE DETECTION ====================

@dataclass
class FatigueMetrics:
    """Fatigue metrics for a player"""
    player_id: int
    frame: int
    timestamp_seconds: float

    # Speed metrics
    current_speed: float = 0.0
    avg_speed_last_5min: float = 0.0
    avg_speed_first_15min: float = 0.0
    speed_decline_pct: float = 0.0

    # Sprint metrics
    sprints_last_5min: int = 0
    sprints_first_15min_rate: float = 0.0  # Per 5 min
    sprint_decline_pct: float = 0.0

    # Recovery
    recovery_time_avg: float = 0.0  # Seconds between sprints

    # Fatigue indicators
    fatigue_score: float = 0.0  # 0-100, higher = more fatigued
    substitution_recommended: bool = False


class FatigueDetector:
    """
    Tracks player movement patterns to detect fatigue.
    """

    def __init__(
        self,
        fps: float = 30.0,
        sprint_threshold_kmh: float = 25.0,
        fatigue_threshold: float = 70.0
    ):
        """
        Args:
            fps: Video frame rate
            sprint_threshold_kmh: Speed threshold for sprints
            fatigue_threshold: Fatigue score threshold for substitution alert
        """
        self.fps = fps
        self.sprint_threshold = sprint_threshold_kmh
        self.fatigue_threshold = fatigue_threshold

        # Player tracking data
        self.player_data: Dict[int, Dict] = defaultdict(lambda: {
            'positions': deque(maxlen=int(fps * 60 * 10)),  # 10 min history
            'speeds': deque(maxlen=int(fps * 60 * 10)),
            'sprint_frames': [],
            'first_15min_avg_speed': None,
            'first_15min_sprint_rate': None
        })

        # Fatigue alerts
        self.alerts: List[Dict] = []

    def update_player(
        self,
        player_id: int,
        frame: int,
        x: float, y: float,
        timestamp_seconds: float
    ):
        """Update player position and calculate speed"""
        data = self.player_data[player_id]
        data['positions'].append((x, y, frame, timestamp_seconds))

        # Calculate speed if we have previous position
        if len(data['positions']) >= 2:
            prev = data['positions'][-2]
            curr = data['positions'][-1]

            dx = (curr[0] - prev[0]) * 105  # Convert to meters
            dy = (curr[1] - prev[1]) * 68
            dt = curr[3] - prev[3]

            if dt > 0:
                speed_ms = np.sqrt(dx**2 + dy**2) / dt
                speed_kmh = speed_ms * 3.6
                data['speeds'].append(speed_kmh)

                # Track sprints
                if speed_kmh >= self.sprint_threshold:
                    data['sprint_frames'].append(frame)

        # Calculate baseline metrics after 15 minutes
        if timestamp_seconds >= 900 and data['first_15min_avg_speed'] is None:
            if len(data['speeds']) > 0:
                data['first_15min_avg_speed'] = np.mean(list(data['speeds']))
                # Sprints per 5 minutes
                sprints_in_15 = len([f for f in data['sprint_frames']
                                     if f <= frame])
                data['first_15min_sprint_rate'] = sprints_in_15 / 3

    def get_fatigue_metrics(self, player_id: int, frame: int, timestamp: float) -> FatigueMetrics:
        """Calculate current fatigue metrics for a player"""
        data = self.player_data.get(player_id)
        if not data or len(data['speeds']) < 10:
            return FatigueMetrics(player_id=player_id, frame=frame, timestamp_seconds=timestamp)

        speeds = list(data['speeds'])

        # Current speed (smoothed)
        current_speed = np.mean(speeds[-30:]) if len(speeds) >= 30 else np.mean(speeds)

        # Last 5 minutes average
        last_5min_samples = int(self.fps * 300)
        recent_speeds = speeds[-last_5min_samples:] if len(speeds) >= last_5min_samples else speeds
        avg_speed_last_5min = np.mean(recent_speeds)

        # Speed decline
        baseline_speed = data.get('first_15min_avg_speed', avg_speed_last_5min)
        speed_decline = ((baseline_speed - avg_speed_last_5min) / max(baseline_speed, 1)) * 100

        # Sprint metrics
        recent_sprints = len([f for f in data['sprint_frames']
                              if f > frame - int(self.fps * 300)])
        baseline_sprint_rate = data.get('first_15min_sprint_rate', recent_sprints / 5)
        sprint_rate_now = recent_sprints / 5  # Per 5 min
        sprint_decline = ((baseline_sprint_rate - sprint_rate_now) /
                          max(baseline_sprint_rate, 1)) * 100

        # Recovery time between sprints
        sprint_gaps = []
        sprint_frames = data['sprint_frames']
        for i in range(1, len(sprint_frames)):
            gap = (sprint_frames[i] - sprint_frames[i-1]) / self.fps
            if gap > 5:  # Ignore consecutive sprint frames
                sprint_gaps.append(gap)
        avg_recovery = np.mean(sprint_gaps) if sprint_gaps else 0

        # Calculate fatigue score (0-100)
        fatigue_score = 0
        fatigue_score += max(0, speed_decline) * 0.4
        fatigue_score += max(0, sprint_decline) * 0.4
        fatigue_score += min(30, avg_recovery / 2) * 0.2  # Higher recovery = more fatigued
        fatigue_score = min(100, max(0, fatigue_score))

        # Check for substitution recommendation
        sub_recommended = fatigue_score >= self.fatigue_threshold

        if sub_recommended and player_id not in [a['player_id'] for a in self.alerts]:
            self.alerts.append({
                'player_id': player_id,
                'frame': frame,
                'timestamp': timestamp,
                'fatigue_score': fatigue_score,
                'message': f"Player {player_id} showing signs of fatigue ({fatigue_score:.0f}%)"
            })

        return FatigueMetrics(
            player_id=player_id,
            frame=frame,
            timestamp_seconds=timestamp,
            current_speed=current_speed,
            avg_speed_last_5min=avg_speed_last_5min,
            avg_speed_first_15min=data.get('first_15min_avg_speed', 0),
            speed_decline_pct=speed_decline,
            sprints_last_5min=recent_sprints,
            sprints_first_15min_rate=baseline_sprint_rate,
            sprint_decline_pct=sprint_decline,
            recovery_time_avg=avg_recovery,
            fatigue_score=fatigue_score,
            substitution_recommended=sub_recommended
        )

    def get_team_fatigue_summary(self, player_ids: List[int], frame: int, timestamp: float) -> Dict:
        """Get fatigue summary for a team"""
        fatigued_players = []
        avg_fatigue = 0

        for pid in player_ids:
            metrics = self.get_fatigue_metrics(pid, frame, timestamp)
            avg_fatigue += metrics.fatigue_score
            if metrics.substitution_recommended:
                fatigued_players.append({
                    'player_id': pid,
                    'fatigue_score': metrics.fatigue_score,
                    'speed_decline': metrics.speed_decline_pct
                })

        return {
            'avg_fatigue_score': avg_fatigue / max(1, len(player_ids)),
            'fatigued_players': sorted(fatigued_players,
                                       key=lambda x: x['fatigue_score'],
                                       reverse=True),
            'substitution_alerts': len(fatigued_players)
        }

    def visualize_fatigue(self, player_ids: List[int], frame: int, timestamp: float,
                          save_path: Optional[str] = None):
        """Visualize team fatigue levels"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fatigue_data = []
        for pid in player_ids:
            metrics = self.get_fatigue_metrics(pid, frame, timestamp)
            fatigue_data.append((pid, metrics.fatigue_score))

        fatigue_data.sort(key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        players = [f"Player {p[0]}" for p in fatigue_data]
        scores = [p[1] for p in fatigue_data]
        colors = ['red' if s >= self.fatigue_threshold else 'orange' if s >= 50 else 'green'
                  for s in scores]

        ax.barh(players, scores, color=colors)
        ax.axvline(x=self.fatigue_threshold, color='red', linestyle='--',
                   label=f'Substitution Threshold ({self.fatigue_threshold})')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Fatigue Score')
        ax.set_title('Player Fatigue Levels')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
