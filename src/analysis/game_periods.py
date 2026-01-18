"""
Soccer Film Analysis - Auto Game Period Detection
Automatically detects kickoffs, halftime, and game end
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class GamePeriod(Enum):
    PRE_GAME = "pre_game"
    FIRST_HALF = "first_half"
    HALFTIME = "halftime"
    SECOND_HALF = "second_half"
    POST_GAME = "post_game"


@dataclass
class PeriodMarker:
    """Marks a game period transition"""
    frame: int
    timestamp_seconds: float
    period_type: str  # 'kickoff', 'halftime_start', 'second_half', 'game_end'
    confidence: float


class GamePeriodDetector:
    """
    Automatically detects game periods based on player and ball positions.

    Detection methods:
    1. Kickoff: Ball at center, players in their halves, then ball moves
    2. Halftime: Extended period with no ball movement near center
    3. Second half: Same as kickoff but with swapped sides
    4. Game end: Extended period with dispersed players
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.frame_data: List[Dict] = []
        self.detected_periods: List[PeriodMarker] = []

        # Detection parameters
        self.center_threshold = 0.1  # % of frame width for center detection
        self.stillness_frames = 30  # Frames ball must be still for kickoff
        self.halftime_duration = 300  # Minimum frames for halftime (10 sec)

    def add_frame(
        self,
        frame_num: int,
        detections,
        frame_width: int,
        frame_height: int
    ):
        """Add frame data for period detection"""
        ball_pos = None
        if detections.ball:
            ball_pos = (
                detections.ball.center[0] / frame_width,
                detections.ball.center[1] / frame_height
            )

        player_positions = []
        for player in detections.players + detections.goalkeepers:
            if player.team_id is not None:
                player_positions.append({
                    'team': player.team_id,
                    'x': player.center[0] / frame_width,
                    'y': player.center[1] / frame_height
                })

        self.frame_data.append({
            'frame': frame_num,
            'timestamp': frame_num / self.fps,
            'ball': ball_pos,
            'players': player_positions,
            'player_count': len(player_positions)
        })

    def detect_periods(self) -> List[PeriodMarker]:
        """
        Analyze all frame data and detect game periods.
        """
        if len(self.frame_data) < 100:
            logger.warning("Not enough frame data for period detection")
            return []

        self.detected_periods = []

        # Detect kickoffs
        kickoffs = self._detect_kickoffs()
        self.detected_periods.extend(kickoffs)

        # Detect halftime
        halftime = self._detect_halftime()
        if halftime:
            self.detected_periods.append(halftime)

        # Sort by frame
        self.detected_periods.sort(key=lambda p: p.frame)

        logger.info(f"Detected {len(self.detected_periods)} period markers")
        return self.detected_periods

    def _detect_kickoffs(self) -> List[PeriodMarker]:
        """Detect kickoff moments"""
        kickoffs = []

        # Find moments where:
        # 1. Ball is near center
        # 2. Ball was stationary
        # 3. Ball then moves away from center

        ball_positions = [(f['frame'], f['ball']) for f in self.frame_data if f['ball']]

        if len(ball_positions) < self.stillness_frames:
            return kickoffs

        for i in range(self.stillness_frames, len(ball_positions)):
            frame, ball_pos = ball_positions[i]

            # Check if ball is near center
            if abs(ball_pos[0] - 0.5) > self.center_threshold:
                continue

            # Check if ball was stationary before
            positions_before = [bp[1] for bp in ball_positions[i-self.stillness_frames:i]]
            if not self._is_stationary(positions_before):
                continue

            # Check if ball moves after
            if i + 10 < len(ball_positions):
                positions_after = [bp[1] for bp in ball_positions[i:i+10]]
                if self._has_significant_movement(positions_after):
                    # Check team positions (players should be in their halves)
                    frame_idx = next(
                        (idx for idx, f in enumerate(self.frame_data) if f['frame'] == frame),
                        None
                    )
                    if frame_idx and self._teams_in_position(self.frame_data[frame_idx]):
                        kickoff = PeriodMarker(
                            frame=frame,
                            timestamp_seconds=frame / self.fps,
                            period_type='kickoff' if len(kickoffs) == 0 else 'second_half',
                            confidence=0.8
                        )
                        kickoffs.append(kickoff)

                        # Skip ahead to avoid duplicate detection
                        i += int(self.fps * 60)  # Skip 1 minute

        return kickoffs

    def _detect_halftime(self) -> Optional[PeriodMarker]:
        """Detect halftime start"""
        # Find extended period with dispersed players or no ball
        no_ball_sequences = []
        current_start = None

        for i, frame_data in enumerate(self.frame_data):
            if frame_data['ball'] is None or frame_data['player_count'] < 15:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    length = i - current_start
                    if length > self.halftime_duration:
                        no_ball_sequences.append((current_start, i, length))
                    current_start = None

        # Find the longest sequence in the middle of the game
        if no_ball_sequences:
            # Filter to sequences in the middle third of the video
            total_frames = len(self.frame_data)
            middle_start = total_frames // 3
            middle_end = 2 * total_frames // 3

            middle_sequences = [
                s for s in no_ball_sequences
                if s[0] > middle_start and s[1] < middle_end
            ]

            if middle_sequences:
                # Use the longest one
                best = max(middle_sequences, key=lambda s: s[2])
                frame_num = self.frame_data[best[0]]['frame']
                return PeriodMarker(
                    frame=frame_num,
                    timestamp_seconds=frame_num / self.fps,
                    period_type='halftime_start',
                    confidence=0.7
                )

        return None

    def _is_stationary(self, positions: List[Tuple[float, float]]) -> bool:
        """Check if ball positions are stationary"""
        if not positions:
            return False

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        return np.std(xs) < 0.01 and np.std(ys) < 0.01

    def _has_significant_movement(self, positions: List[Tuple[float, float]]) -> bool:
        """Check if ball has significant movement"""
        if len(positions) < 2:
            return False

        total_dist = sum(
            np.sqrt((positions[i][0] - positions[i-1][0])**2 +
                   (positions[i][1] - positions[i-1][1])**2)
            for i in range(1, len(positions))
        )

        return total_dist > 0.05

    def _teams_in_position(self, frame_data: Dict) -> bool:
        """Check if teams are in their respective halves"""
        players = frame_data.get('players', [])

        team0_in_left = sum(1 for p in players if p['team'] == 0 and p['x'] < 0.5)
        team1_in_right = sum(1 for p in players if p['team'] == 1 and p['x'] > 0.5)

        team0_total = sum(1 for p in players if p['team'] == 0)
        team1_total = sum(1 for p in players if p['team'] == 1)

        if team0_total < 5 or team1_total < 5:
            return False

        # At least 70% of each team should be in their half
        return (team0_in_left / team0_total > 0.7 and
                team1_in_right / team1_total > 0.7)

    def get_current_period(self, frame_num: int) -> GamePeriod:
        """Get the current game period for a frame"""
        if not self.detected_periods:
            return GamePeriod.FIRST_HALF

        for period in reversed(self.detected_periods):
            if frame_num >= period.frame:
                if period.period_type == 'kickoff':
                    return GamePeriod.FIRST_HALF
                elif period.period_type == 'halftime_start':
                    return GamePeriod.HALFTIME
                elif period.period_type == 'second_half':
                    return GamePeriod.SECOND_HALF
                elif period.period_type == 'game_end':
                    return GamePeriod.POST_GAME

        return GamePeriod.PRE_GAME
