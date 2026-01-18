"""
Soccer Film Analysis - Detection Package
Player, ball, and pitch detection with tracking
"""

from .detector import (
    SoccerDetector,
    Detection,
    PlayerDetection,
    BallDetection,
    FrameDetections,
    PitchKeypoint,
    TeamClassifier,
    PossessionCalculator,
    PitchTransformer,
    draw_detections
)

from .pitch_detector import (
    PitchDetector,
    PitchBoundary
)

from .enhanced_detector import (
    EnhancedDetector,
    TrackedPerson
)

__all__ = [
    # Base detector
    'SoccerDetector',
    'Detection',
    'PlayerDetection',
    'BallDetection',
    'FrameDetections',
    'PitchKeypoint',
    'TeamClassifier',
    'PossessionCalculator',
    'PitchTransformer',
    'draw_detections',
    # Pitch detection
    'PitchDetector',
    'PitchBoundary',
    # Enhanced detector
    'EnhancedDetector',
    'TrackedPerson',
]
