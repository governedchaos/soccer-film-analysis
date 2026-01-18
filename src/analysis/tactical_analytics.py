"""
Soccer Film Analysis - Tactical Analytics
Advanced tactical analysis including zones, pressing, set pieces, and more
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from loguru import logger

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, Polygon
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ==================== PITCH ZONES ====================

class PitchZone(Enum):
    """Pitch zones for analysis"""
    # Thirds
    DEFENSIVE_THIRD = "defensive_third"
    MIDDLE_THIRD = "middle_third"
    ATTACKING_THIRD = "attacking_third"

    # Channels
    LEFT_CHANNEL = "left_channel"
    CENTER_CHANNEL = "center_channel"
    RIGHT_CHANNEL = "right_channel"

    # Combined zones (9-zone grid)
    DEF_LEFT = "def_left"
    DEF_CENTER = "def_center"
    DEF_RIGHT = "def_right"
    MID_LEFT = "mid_left"
    MID_CENTER = "mid_center"
    MID_RIGHT = "mid_right"
    ATT_LEFT = "att_left"
    ATT_CENTER = "att_center"
    ATT_RIGHT = "att_right"

    # Special zones
    PENALTY_AREA_HOME = "penalty_area_home"
    PENALTY_AREA_AWAY = "penalty_area_away"
    BOX_HOME = "box_home"
    BOX_AWAY = "box_away"


@dataclass
class ZoneStats:
    """Statistics for a pitch zone"""
    zone: str
    total_time_seconds: float = 0.0
    ball_entries: int = 0
    passes_in: int = 0
    passes_out: int = 0
    passes_completed: int = 0
    shots: int = 0
    tackles: int = 0
    interceptions: int = 0
    possession_percentage: float = 0.0


class PitchZoneAnalyzer:
    """
    Analyzes activity in different pitch zones.
    Divides the pitch into configurable zones and tracks metrics.
    """

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        """
        Args:
            pitch_length: Pitch length in meters (default: 105m)
            pitch_width: Pitch width in meters (default: 68m)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Zone boundaries (normalized 0-1)
        self.third_bounds = [0.0, 0.333, 0.667, 1.0]
        self.channel_bounds = [0.0, 0.333, 0.667, 1.0]

        # Zone statistics per team
        self.home_stats: Dict[str, ZoneStats] = {}
        self.away_stats: Dict[str, ZoneStats] = {}

        # Tracking
        self.ball_positions: List[Tuple[float, float, int]] = []  # (x, y, frame)
        self.last_zone: Optional[str] = None
        self._init_zones()

    def _init_zones(self):
        """Initialize zone statistics"""
        zones = [
            "def_left", "def_center", "def_right",
            "mid_left", "mid_center", "mid_right",
            "att_left", "att_center", "att_right"
        ]
        for zone in zones:
            self.home_stats[zone] = ZoneStats(zone=zone)
            self.away_stats[zone] = ZoneStats(zone=zone)

    def get_zone(self, x: float, y: float, team_id: int = 0) -> str:
        """
        Get the zone name for a position.

        Args:
            x: Normalized x position (0-1, left to right)
            y: Normalized y position (0-1, bottom to top)
            team_id: Team perspective (0 = attacking right, 1 = attacking left)

        Returns:
            Zone name string
        """
        # Flip for away team perspective
        if team_id == 1:
            x = 1.0 - x

        # Determine third
        if x < self.third_bounds[1]:
            third = "def"
        elif x < self.third_bounds[2]:
            third = "mid"
        else:
            third = "att"

        # Determine channel
        if y < self.channel_bounds[1]:
            channel = "left"
        elif y < self.channel_bounds[2]:
            channel = "center"
        else:
            channel = "right"

        return f"{third}_{channel}"

    def add_ball_position(self, x: float, y: float, frame: int, team_with_possession: int):
        """Record ball position for zone analysis"""
        self.ball_positions.append((x, y, frame))

        zone = self.get_zone(x, y, team_with_possession)
        stats = self.home_stats if team_with_possession == 0 else self.away_stats

        if zone in stats:
            stats[zone].total_time_seconds += 1/30  # Assume 30fps

            # Check for zone entry
            if self.last_zone != zone:
                stats[zone].ball_entries += 1
                self.last_zone = zone

    def add_pass(self, start_x: float, start_y: float, end_x: float, end_y: float,
                 team_id: int, completed: bool):
        """Record a pass for zone statistics"""
        start_zone = self.get_zone(start_x, start_y, team_id)
        end_zone = self.get_zone(end_x, end_y, team_id)

        stats = self.home_stats if team_id == 0 else self.away_stats

        if start_zone in stats:
            stats[start_zone].passes_out += 1
        if end_zone in stats:
            stats[end_zone].passes_in += 1
            if completed:
                stats[end_zone].passes_completed += 1

    def add_shot(self, x: float, y: float, team_id: int):
        """Record a shot for zone statistics"""
        zone = self.get_zone(x, y, team_id)
        stats = self.home_stats if team_id == 0 else self.away_stats
        if zone in stats:
            stats[zone].shots += 1

    def add_defensive_action(self, x: float, y: float, team_id: int, action_type: str):
        """Record defensive action (tackle, interception)"""
        zone = self.get_zone(x, y, team_id)
        stats = self.home_stats if team_id == 0 else self.away_stats
        if zone in stats:
            if action_type == "tackle":
                stats[zone].tackles += 1
            elif action_type == "interception":
                stats[zone].interceptions += 1

    def calculate_possession_by_zone(self):
        """Calculate possession percentage for each zone"""
        total_time = sum(s.total_time_seconds for s in self.home_stats.values())
        total_time += sum(s.total_time_seconds for s in self.away_stats.values())

        if total_time > 0:
            for zone in self.home_stats:
                zone_total = (self.home_stats[zone].total_time_seconds +
                              self.away_stats[zone].total_time_seconds)
                if zone_total > 0:
                    self.home_stats[zone].possession_percentage = (
                        self.home_stats[zone].total_time_seconds / zone_total * 100
                    )
                    self.away_stats[zone].possession_percentage = (
                        self.away_stats[zone].total_time_seconds / zone_total * 100
                    )

    def get_zone_summary(self, team_id: int = 0) -> Dict[str, Dict]:
        """Get summary statistics for all zones"""
        self.calculate_possession_by_zone()
        stats = self.home_stats if team_id == 0 else self.away_stats

        return {
            zone: {
                'time_seconds': s.total_time_seconds,
                'ball_entries': s.ball_entries,
                'passes_in': s.passes_in,
                'passes_completed': s.passes_completed,
                'pass_completion_rate': (s.passes_completed / max(1, s.passes_in) * 100),
                'shots': s.shots,
                'defensive_actions': s.tackles + s.interceptions,
                'possession_pct': s.possession_percentage
            }
            for zone, s in stats.items()
        }

    def visualize_zones(self, team_id: int = 0, metric: str = "possession_pct",
                        save_path: Optional[str] = None):
        """Create visualization of zone statistics"""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib required for visualization")
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw pitch
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.65)
        ax.set_facecolor('#2e7d32')
        ax.set_aspect('equal')

        stats = self.home_stats if team_id == 0 else self.away_stats

        # Get values for color scaling
        values = [getattr(stats[z], metric.replace('_pct', '_percentage'), 0)
                  if 'pct' in metric else
                  getattr(stats[z], metric, 0)
                  for z in stats]
        max_val = max(values) if values else 1

        # Draw zones
        zone_positions = {
            'def_left': (0, 0), 'def_center': (0, 0.217), 'def_right': (0, 0.434),
            'mid_left': (0.333, 0), 'mid_center': (0.333, 0.217), 'mid_right': (0.333, 0.434),
            'att_left': (0.667, 0), 'att_center': (0.667, 0.217), 'att_right': (0.667, 0.434),
        }

        for zone, (x, y) in zone_positions.items():
            value = getattr(stats[zone], metric.replace('_pct', '_percentage'), 0) if 'pct' in metric else getattr(stats[zone], metric, 0)
            intensity = value / max(max_val, 1)

            rect = Rectangle((x, y), 0.333, 0.217,
                              facecolor=plt.cm.YlOrRd(intensity),
                              edgecolor='white', linewidth=2, alpha=0.8)
            ax.add_patch(rect)

            # Add value text
            ax.text(x + 0.167, y + 0.108, f"{value:.1f}",
                    ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        # Add labels
        ax.text(0.167, -0.03, "Defensive", ha='center', fontsize=12, color='white')
        ax.text(0.5, -0.03, "Middle", ha='center', fontsize=12, color='white')
        ax.text(0.833, -0.03, "Attacking", ha='center', fontsize=12, color='white')

        ax.set_title(f"Zone Analysis: {metric.replace('_', ' ').title()}", fontsize=16, color='white')
        ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
            logger.info(f"Zone visualization saved to: {save_path}")

        return fig


# ==================== PRESSING INTENSITY ====================

@dataclass
class PressingMetrics:
    """Metrics for pressing intensity"""
    ppda: float = 0.0  # Passes Per Defensive Action
    ppda_attacking_third: float = 0.0
    high_press_sequences: int = 0
    successful_high_presses: int = 0
    press_success_rate: float = 0.0
    avg_press_duration_seconds: float = 0.0
    recoveries_in_att_third: int = 0
    defensive_actions_total: int = 0


class PressingAnalyzer:
    """
    Analyzes pressing intensity and high press effectiveness.
    """

    def __init__(self, high_press_line: float = 0.6):
        """
        Args:
            high_press_line: X position (0-1) considered high press (default: 60%)
        """
        self.high_press_line = high_press_line

        # Team data
        self.home_data = {
            'passes_allowed': 0,
            'defensive_actions': 0,
            'defensive_actions_att_third': 0,
            'recoveries_att_third': 0,
            'press_sequences': [],
            'current_press': None
        }
        self.away_data = {
            'passes_allowed': 0,
            'defensive_actions': 0,
            'defensive_actions_att_third': 0,
            'recoveries_att_third': 0,
            'press_sequences': [],
            'current_press': None
        }

    def record_pass(self, team_id: int, x: float, completed: bool):
        """Record an opponent's pass (for PPDA calculation)"""
        # PPDA is calculated as opponent passes / our defensive actions
        defending_team = 1 - team_id
        data = self.home_data if defending_team == 0 else self.away_data

        # Only count passes in defensive/middle thirds (their attacking play)
        if x > 0.33:  # In middle or attacking third
            data['passes_allowed'] += 1

    def record_defensive_action(self, team_id: int, x: float, y: float,
                                 action_type: str, successful: bool = True):
        """Record a defensive action (tackle, interception, etc.)"""
        data = self.home_data if team_id == 0 else self.away_data

        data['defensive_actions'] += 1

        # Check if in attacking third (opponent's defensive third)
        if x > 0.67:
            data['defensive_actions_att_third'] += 1

            if successful:
                data['recoveries_att_third'] += 1

                # Check if this is part of a high press
                if x > self.high_press_line:
                    if data['current_press'] is None:
                        data['current_press'] = {
                            'start_frame': 0,  # Would be actual frame
                            'actions': []
                        }
                    data['current_press']['actions'].append({
                        'type': action_type,
                        'x': x,
                        'y': y,
                        'successful': successful
                    })

    def end_press_sequence(self, team_id: int, successful: bool, frame: int):
        """End a high press sequence"""
        data = self.home_data if team_id == 0 else self.away_data

        if data['current_press'] is not None:
            data['current_press']['successful'] = successful
            data['current_press']['end_frame'] = frame
            data['press_sequences'].append(data['current_press'])
            data['current_press'] = None

    def calculate_ppda(self, team_id: int) -> float:
        """Calculate Passes Per Defensive Action"""
        data = self.home_data if team_id == 0 else self.away_data

        if data['defensive_actions'] == 0:
            return 0.0

        return data['passes_allowed'] / data['defensive_actions']

    def get_match_ppda(self) -> Dict[str, float]:
        """
        Get PPDA (Passes Per Defensive Action) for both teams for the match.

        Lower PPDA = more intense pressing (fewer passes allowed per action)

        Returns:
            Dict with 'home' and 'away' PPDA values
        """
        return {
            'home': self.calculate_ppda(0),
            'away': self.calculate_ppda(1)
        }

    def get_metrics(self, team_id: int) -> PressingMetrics:
        """Get all pressing metrics for a team"""
        data = self.home_data if team_id == 0 else self.away_data

        ppda = self.calculate_ppda(team_id)

        # Calculate high press stats
        press_sequences = data['press_sequences']
        successful_presses = sum(1 for p in press_sequences if p.get('successful', False))

        return PressingMetrics(
            ppda=ppda,
            ppda_attacking_third=data['passes_allowed'] / max(1, data['defensive_actions_att_third']),
            high_press_sequences=len(press_sequences),
            successful_high_presses=successful_presses,
            press_success_rate=successful_presses / max(1, len(press_sequences)) * 100,
            recoveries_in_att_third=data['recoveries_att_third'],
            defensive_actions_total=data['defensive_actions']
        )


# ==================== GOALKEEPER DISTRIBUTION ====================

@dataclass
class GKDistributionStats:
    """Goalkeeper distribution statistics"""
    total_distributions: int = 0

    # By type
    goal_kicks: int = 0
    throws: int = 0
    punts: int = 0
    passes_from_hands: int = 0

    # By direction
    left_side: int = 0
    center: int = 0
    right_side: int = 0

    # By length
    short_passes: int = 0  # < 25m
    medium_passes: int = 0  # 25-45m
    long_passes: int = 0  # > 45m

    # Success rates
    successful: int = 0
    to_defender: int = 0
    to_midfielder: int = 0
    to_attacker: int = 0

    @property
    def success_rate(self) -> float:
        return self.successful / max(1, self.total_distributions) * 100


class GoalkeeperAnalyzer:
    """
    Analyzes goalkeeper distribution patterns.
    """

    def __init__(self):
        self.home_gk = GKDistributionStats()
        self.away_gk = GKDistributionStats()
        self.distributions: List[Dict] = []

    def record_distribution(
        self,
        team_id: int,
        start_x: float, start_y: float,
        end_x: float, end_y: float,
        dist_type: str,  # goal_kick, throw, punt, pass
        successful: bool,
        recipient_position: Optional[str] = None  # defender, midfielder, attacker
    ):
        """Record a goalkeeper distribution"""
        stats = self.home_gk if team_id == 0 else self.away_gk

        stats.total_distributions += 1

        # Type
        if dist_type == "goal_kick":
            stats.goal_kicks += 1
        elif dist_type == "throw":
            stats.throws += 1
        elif dist_type == "punt":
            stats.punts += 1
        else:
            stats.passes_from_hands += 1

        # Direction (based on end y position)
        if end_y < 0.33:
            stats.left_side += 1
        elif end_y < 0.67:
            stats.center += 1
        else:
            stats.right_side += 1

        # Length (approximate distance)
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) * 105  # Convert to meters
        if distance < 25:
            stats.short_passes += 1
        elif distance < 45:
            stats.medium_passes += 1
        else:
            stats.long_passes += 1

        # Success
        if successful:
            stats.successful += 1

        # Recipient
        if recipient_position == "defender":
            stats.to_defender += 1
        elif recipient_position == "midfielder":
            stats.to_midfielder += 1
        elif recipient_position == "attacker":
            stats.to_attacker += 1

        # Store for visualization
        self.distributions.append({
            'team_id': team_id,
            'start': (start_x, start_y),
            'end': (end_x, end_y),
            'type': dist_type,
            'successful': successful,
            'distance': distance
        })

    def get_stats(self, team_id: int) -> GKDistributionStats:
        """Get distribution stats for a team's goalkeeper"""
        return self.home_gk if team_id == 0 else self.away_gk

    def visualize_distribution(self, team_id: int, save_path: Optional[str] = None):
        """Create visualization of GK distribution patterns"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw half pitch
        ax.set_xlim(-0.05, 0.55)
        ax.set_ylim(-0.05, 0.7)
        ax.set_facecolor('#2e7d32')

        # Pitch markings
        ax.plot([0, 0.5], [0, 0], 'w-', linewidth=2)
        ax.plot([0, 0.5], [0.65, 0.65], 'w-', linewidth=2)
        ax.plot([0, 0], [0, 0.65], 'w-', linewidth=2)
        ax.plot([0.5, 0.5], [0, 0.65], 'w-', linewidth=2)

        # Goal area
        ax.plot([0, 0.06], [0.22, 0.22], 'w-')
        ax.plot([0, 0.06], [0.43, 0.43], 'w-')
        ax.plot([0.06, 0.06], [0.22, 0.43], 'w-')

        # Penalty area
        ax.plot([0, 0.16], [0.13, 0.13], 'w-')
        ax.plot([0, 0.16], [0.52, 0.52], 'w-')
        ax.plot([0.16, 0.16], [0.13, 0.52], 'w-')

        # Draw distributions
        team_dists = [d for d in self.distributions if d['team_id'] == team_id]
        for d in team_dists:
            color = 'lime' if d['successful'] else 'red'
            alpha = 0.7
            ax.arrow(d['start'][0], d['start'][1],
                     d['end'][0] - d['start'][0], d['end'][1] - d['start'][1],
                     head_width=0.02, head_length=0.01, fc=color, ec=color, alpha=alpha)

        stats = self.get_stats(team_id)
        ax.set_title(f"GK Distribution (Success: {stats.success_rate:.0f}%)", fontsize=14, color='white')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')

        return fig


# ==================== SET PIECE ANALYSIS ====================

@dataclass
class SetPieceEvent:
    """A set piece event"""
    frame: int
    timestamp_seconds: float
    set_piece_type: str  # corner, free_kick, throw_in, penalty
    team_id: int
    location: Tuple[float, float]
    delivery_type: Optional[str] = None  # inswing, outswing, short, long
    outcome: Optional[str] = None  # goal, shot, cleared, won_possession
    scorer_id: Optional[int] = None
    assister_id: Optional[int] = None


class SetPieceAnalyzer:
    """
    Analyzes set pieces (corners, free kicks, etc.)
    """

    def __init__(self):
        self.events: List[SetPieceEvent] = []

        # Stats by type and team
        self.stats = {
            0: defaultdict(lambda: {'total': 0, 'goals': 0, 'shots': 0, 'won_possession': 0}),
            1: defaultdict(lambda: {'total': 0, 'goals': 0, 'shots': 0, 'won_possession': 0})
        }

    def add_set_piece(self, event: SetPieceEvent):
        """Add a set piece event"""
        self.events.append(event)

        # Update stats
        team_stats = self.stats[event.team_id][event.set_piece_type]
        team_stats['total'] += 1

        if event.outcome == 'goal':
            team_stats['goals'] += 1
            team_stats['shots'] += 1
        elif event.outcome == 'shot':
            team_stats['shots'] += 1
        elif event.outcome == 'won_possession':
            team_stats['won_possession'] += 1

    def detect_corner(self, ball_x: float, ball_y: float, frame: int,
                      team_with_possession: int, timestamp: float) -> Optional[SetPieceEvent]:
        """Detect if ball position indicates a corner kick"""
        # Corner positions (normalized)
        corner_positions = [
            (0.0, 0.0), (0.0, 1.0),  # Home corners
            (1.0, 0.0), (1.0, 1.0)   # Away corners
        ]

        for cx, cy in corner_positions:
            if abs(ball_x - cx) < 0.02 and abs(ball_y - cy) < 0.02:
                # Determine if attacking corner
                is_attacking = (team_with_possession == 0 and cx > 0.5) or \
                               (team_with_possession == 1 and cx < 0.5)
                if is_attacking:
                    return SetPieceEvent(
                        frame=frame,
                        timestamp_seconds=timestamp,
                        set_piece_type='corner',
                        team_id=team_with_possession,
                        location=(ball_x, ball_y)
                    )
        return None

    def get_corner_stats(self, team_id: int) -> Dict:
        """Get corner kick statistics"""
        stats = self.stats[team_id]['corner']
        return {
            'total': stats['total'],
            'goals': stats['goals'],
            'conversion_rate': stats['goals'] / max(1, stats['total']) * 100,
            'shots': stats['shots'],
            'shot_rate': stats['shots'] / max(1, stats['total']) * 100
        }

    def get_free_kick_stats(self, team_id: int) -> Dict:
        """Get free kick statistics"""
        stats = self.stats[team_id]['free_kick']
        return {
            'total': stats['total'],
            'goals': stats['goals'],
            'direct_goals': sum(1 for e in self.events
                                if e.team_id == team_id and e.set_piece_type == 'free_kick'
                                and e.outcome == 'goal' and e.delivery_type == 'direct'),
            'shots': stats['shots']
        }

    def visualize_corners(self, team_id: int, save_path: Optional[str] = None):
        """Visualize corner kick delivery patterns"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        corners = [e for e in self.events
                   if e.team_id == team_id and e.set_piece_type == 'corner']

        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw penalty area (attacking corner view)
        ax.set_xlim(0.5, 1.05)
        ax.set_ylim(-0.05, 0.7)
        ax.set_facecolor('#2e7d32')

        # Pitch lines
        ax.plot([0.5, 1], [0, 0], 'w-', linewidth=2)
        ax.plot([0.5, 1], [0.65, 0.65], 'w-', linewidth=2)
        ax.plot([1, 1], [0, 0.65], 'w-', linewidth=2)

        # Penalty area
        ax.plot([0.84, 1], [0.13, 0.13], 'w-')
        ax.plot([0.84, 1], [0.52, 0.52], 'w-')
        ax.plot([0.84, 0.84], [0.13, 0.52], 'w-')

        # Goal
        ax.plot([1, 1], [0.27, 0.38], 'w-', linewidth=4)

        # Plot corner deliveries
        for corner in corners:
            color = 'gold' if corner.outcome == 'goal' else \
                    'orange' if corner.outcome == 'shot' else 'white'
            ax.scatter(corner.location[0], corner.location[1],
                       c=color, s=100, alpha=0.7, edgecolors='black')

        stats = self.get_corner_stats(team_id)
        ax.set_title(f"Corners: {stats['total']} | Goals: {stats['goals']} | Shots: {stats['shots']}",
                     fontsize=12, color='white')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')

        return fig


# ==================== COUNTER-ATTACK DETECTION ====================

@dataclass
class CounterAttack:
    """Represents a counter-attack sequence"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    team_id: int
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    ball_recoveries: int = 1
    passes: int = 0
    outcome: str = ""  # goal, shot, lost_possession
    max_speed: float = 0.0
    distance_covered: float = 0.0


class CounterAttackDetector:
    """
    Detects and analyzes counter-attack sequences.

    A counter-attack is defined as:
    1. Possession won in defensive/middle third
    2. Rapid forward ball movement
    3. Attack reaches attacking third within N seconds
    """

    def __init__(
        self,
        max_duration_seconds: float = 10.0,
        min_forward_progress: float = 0.4,
        min_speed_threshold: float = 0.03  # Normalized units per frame
    ):
        self.max_duration = max_duration_seconds
        self.min_forward_progress = min_forward_progress
        self.min_speed = min_speed_threshold

        self.counter_attacks: List[CounterAttack] = []
        self.current_sequences: Dict[int, Dict] = {}  # By team

    def process_frame(
        self,
        frame: int,
        timestamp: float,
        ball_x: float, ball_y: float,
        team_with_possession: int,
        possession_just_won: bool,
        ball_velocity: float
    ):
        """Process a frame to detect counter-attacks"""
        team_id = team_with_possession

        # Check for new counter-attack start
        if possession_just_won and ball_x < 0.5:  # Won in own half
            self.current_sequences[team_id] = {
                'start_frame': frame,
                'start_time': timestamp,
                'start_position': (ball_x, ball_y),
                'max_x': ball_x,
                'max_speed': ball_velocity,
                'passes': 0,
                'positions': [(ball_x, ball_y)]
            }

        # Update existing sequence
        if team_id in self.current_sequences:
            seq = self.current_sequences[team_id]

            # Check duration
            if timestamp - seq['start_time'] > self.max_duration:
                # Sequence too slow, not a counter
                del self.current_sequences[team_id]
                return

            # Update tracking
            seq['positions'].append((ball_x, ball_y))
            seq['max_x'] = max(seq['max_x'], ball_x)
            seq['max_speed'] = max(seq['max_speed'], ball_velocity)

            # Check for successful counter (reached attacking third quickly)
            forward_progress = ball_x - seq['start_position'][0]
            if ball_x > 0.67 and forward_progress >= self.min_forward_progress:
                # Valid counter-attack
                counter = CounterAttack(
                    start_frame=seq['start_frame'],
                    end_frame=frame,
                    start_time=seq['start_time'],
                    end_time=timestamp,
                    team_id=team_id,
                    start_position=seq['start_position'],
                    end_position=(ball_x, ball_y),
                    passes=seq['passes'],
                    max_speed=seq['max_speed'],
                    distance_covered=forward_progress
                )
                self.counter_attacks.append(counter)
                del self.current_sequences[team_id]

    def record_pass_in_sequence(self, team_id: int):
        """Record a pass during a potential counter"""
        if team_id in self.current_sequences:
            self.current_sequences[team_id]['passes'] += 1

    def set_counter_outcome(self, team_id: int, outcome: str):
        """Set the outcome of the last counter-attack"""
        team_counters = [c for c in self.counter_attacks if c.team_id == team_id]
        if team_counters:
            team_counters[-1].outcome = outcome

    def get_stats(self, team_id: int) -> Dict:
        """Get counter-attack statistics for a team"""
        team_counters = [c for c in self.counter_attacks if c.team_id == team_id]

        goals = sum(1 for c in team_counters if c.outcome == 'goal')
        shots = sum(1 for c in team_counters if c.outcome in ['goal', 'shot'])

        return {
            'total_counters': len(team_counters),
            'goals': goals,
            'shots': shots,
            'conversion_rate': goals / max(1, len(team_counters)) * 100,
            'avg_duration': sum(c.end_time - c.start_time for c in team_counters) / max(1, len(team_counters)),
            'avg_passes': sum(c.passes for c in team_counters) / max(1, len(team_counters))
        }


# ==================== DEFENSIVE PRESSURE ZONES ====================

@dataclass
class PressureZone:
    """A zone of defensive pressure"""
    zone_id: str
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    pressure_intensity: float = 0.0  # 0-1 scale
    defensive_actions: int = 0
    ball_recoveries: int = 0
    time_in_zone_seconds: float = 0.0


class DefensivePressureAnalyzer:
    """
    Analyzes defensive pressure zones and pressing patterns.
    """

    def __init__(self, grid_size: Tuple[int, int] = (6, 4)):
        """
        Args:
            grid_size: (x_divisions, y_divisions) for the zone grid
        """
        self.grid_size = grid_size
        self.home_zones: Dict[str, PressureZone] = {}
        self.away_zones: Dict[str, PressureZone] = {}
        self._init_zones()

        # Frame-by-frame pressure data
        self.pressure_frames: List[Dict] = []

    def _init_zones(self):
        """Initialize pressure zone grid"""
        x_div, y_div = self.grid_size
        for xi in range(x_div):
            for yi in range(y_div):
                zone_id = f"{xi}_{yi}"
                x_range = (xi / x_div, (xi + 1) / x_div)
                y_range = (yi / y_div, (yi + 1) / y_div)

                self.home_zones[zone_id] = PressureZone(
                    zone_id=zone_id,
                    x_range=x_range,
                    y_range=y_range
                )
                self.away_zones[zone_id] = PressureZone(
                    zone_id=zone_id,
                    x_range=x_range,
                    y_range=y_range
                )

    def _get_zone_id(self, x: float, y: float) -> str:
        """Get zone ID for a position"""
        x_div, y_div = self.grid_size
        xi = min(int(x * x_div), x_div - 1)
        yi = min(int(y * y_div), y_div - 1)
        return f"{xi}_{yi}"

    def calculate_frame_pressure(
        self,
        defending_team_positions: List[Tuple[float, float]],
        ball_position: Tuple[float, float],
        defending_team_id: int
    ) -> Dict[str, float]:
        """
        Calculate pressure intensity for each zone based on player positions.
        """
        zones = self.home_zones if defending_team_id == 0 else self.away_zones

        pressure_map = {}
        ball_x, ball_y = ball_position

        for zone_id, zone in zones.items():
            zone_center_x = (zone.x_range[0] + zone.x_range[1]) / 2
            zone_center_y = (zone.y_range[0] + zone.y_range[1]) / 2

            # Calculate player density in zone
            players_in_zone = 0
            for px, py in defending_team_positions:
                if zone.x_range[0] <= px < zone.x_range[1] and \
                   zone.y_range[0] <= py < zone.y_range[1]:
                    players_in_zone += 1

            # Distance from ball affects pressure importance
            dist_to_ball = np.sqrt((zone_center_x - ball_x)**2 + (zone_center_y - ball_y)**2)
            ball_proximity_factor = max(0, 1 - dist_to_ball * 2)

            # Pressure intensity (0-1)
            intensity = min(1.0, players_in_zone * 0.3) * (0.5 + 0.5 * ball_proximity_factor)
            pressure_map[zone_id] = intensity

        return pressure_map

    def record_frame_pressure(
        self,
        frame: int,
        defending_team_id: int,
        pressure_map: Dict[str, float],
        ball_position: Tuple[float, float]
    ):
        """Record pressure data for a frame"""
        self.pressure_frames.append({
            'frame': frame,
            'team_id': defending_team_id,
            'pressure': pressure_map.copy(),
            'ball': ball_position
        })

        # Update zone time
        zones = self.home_zones if defending_team_id == 0 else self.away_zones
        for zone_id, intensity in pressure_map.items():
            zones[zone_id].pressure_intensity = (
                zones[zone_id].pressure_intensity * 0.95 + intensity * 0.05
            )  # Exponential moving average
            zones[zone_id].time_in_zone_seconds += 1/30  # Assume 30fps

    def record_defensive_action(self, x: float, y: float, team_id: int, successful: bool):
        """Record a defensive action in a zone"""
        zone_id = self._get_zone_id(x, y)
        zones = self.home_zones if team_id == 0 else self.away_zones

        if zone_id in zones:
            zones[zone_id].defensive_actions += 1
            if successful:
                zones[zone_id].ball_recoveries += 1

    def get_pressure_heatmap(self, team_id: int) -> np.ndarray:
        """Get pressure intensity as 2D array for visualization"""
        zones = self.home_zones if team_id == 0 else self.away_zones

        x_div, y_div = self.grid_size
        heatmap = np.zeros((y_div, x_div))

        for zone_id, zone in zones.items():
            xi, yi = map(int, zone_id.split('_'))
            heatmap[yi, xi] = zone.pressure_intensity

        return heatmap

    def visualize_pressure(self, team_id: int, save_path: Optional[str] = None):
        """Create pressure zone visualization"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))

        heatmap = self.get_pressure_heatmap(team_id)

        # Draw pitch with pressure overlay
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.65)

        # Create custom colormap
        cmap = LinearSegmentedColormap.from_list('pressure',
                                                  ['#2e7d32', '#ffeb3b', '#f44336'])

        im = ax.imshow(heatmap, extent=[0, 1, 0, 0.65], origin='lower',
                       cmap=cmap, alpha=0.7, aspect='auto')

        # Pitch lines
        ax.plot([0, 1], [0, 0], 'w-', linewidth=2)
        ax.plot([0, 1], [0.65, 0.65], 'w-', linewidth=2)
        ax.plot([0, 0], [0, 0.65], 'w-', linewidth=2)
        ax.plot([1, 1], [0, 0.65], 'w-', linewidth=2)
        ax.plot([0.5, 0.5], [0, 0.65], 'w-', linewidth=1)

        plt.colorbar(im, ax=ax, label='Pressure Intensity')
        ax.set_title("Defensive Pressure Zones", fontsize=14)
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')

        return fig

    def get_weak_zones(self, team_id: int, threshold: float = 0.3) -> List[str]:
        """Identify zones with low defensive pressure"""
        zones = self.home_zones if team_id == 0 else self.away_zones
        return [
            zone_id for zone_id, zone in zones.items()
            if zone.pressure_intensity < threshold
        ]

    def get_summary(self, team_id: int) -> Dict:
        """Get summary of defensive pressure"""
        zones = self.home_zones if team_id == 0 else self.away_zones

        intensities = [z.pressure_intensity for z in zones.values()]
        actions = sum(z.defensive_actions for z in zones.values())
        recoveries = sum(z.ball_recoveries for z in zones.values())

        return {
            'avg_pressure': np.mean(intensities),
            'max_pressure_zone': max(zones.keys(), key=lambda z: zones[z].pressure_intensity),
            'min_pressure_zone': min(zones.keys(), key=lambda z: zones[z].pressure_intensity),
            'total_defensive_actions': actions,
            'total_recoveries': recoveries,
            'recovery_rate': recoveries / max(1, actions) * 100,
            'weak_zones': self.get_weak_zones(team_id)
        }
