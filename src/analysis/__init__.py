"""
Soccer Film Analysis - Analysis Module
"""

from .export import AnalysisExporter, format_timestamp
# from .hudl_import import HudlImporter, HudlEvent, compare_events

# Tactical analytics
from .tactical_analytics import (
    PitchZoneAnalyzer,
    PressingAnalyzer,
    GoalkeeperAnalyzer,
    SetPieceAnalyzer,
    CounterAttackDetector,
    DefensivePressureAnalyzer
)

# Advanced tactical
from .advanced_tactical import (
    TeamShapeAnalyzer,
    DuelAnalyzer,
    PassingLaneAnalyzer,
    ExpectedThreatModel,
    BirdsEyeViewTransformer,
    OpponentTendencyAnalyzer,
    FatigueDetector
)

# PDF reports
from .pdf_report import PDFReportGenerator

# Formation detection
from .formation_detection import (
    FormationDetector,
    FormationAnalyzer,
    FormationType,
    FormationSnapshot,
    FormationChange
)

# Expected goals
from .expected_goals import (
    ExpectedGoalsModel,
    xGTracker,
    xGZoneAnalyzer,
    ShotContext,
    ShotType,
    ShotSituation,
    ShotOutcome
)

# Space analysis
from .space_analysis import (
    SpaceCreationAnalyzer,
    ThirdManRunDetector,
    OffBallMovementAnalyzer,
    SpaceCalculator,
    RunType,
    SpaceType
)

__all__ = [
    "AnalysisExporter",
    "format_timestamp",
    # Tactical
    "PitchZoneAnalyzer",
    "PressingAnalyzer",
    "GoalkeeperAnalyzer",
    "SetPieceAnalyzer",
    "CounterAttackDetector",
    "DefensivePressureAnalyzer",
    # Advanced tactical
    "TeamShapeAnalyzer",
    "DuelAnalyzer",
    "PassingLaneAnalyzer",
    "ExpectedThreatModel",
    "BirdsEyeViewTransformer",
    "OpponentTendencyAnalyzer",
    "FatigueDetector",
    # PDF
    "PDFReportGenerator",
    # Formation
    "FormationDetector",
    "FormationAnalyzer",
    "FormationType",
    "FormationSnapshot",
    "FormationChange",
    # xG
    "ExpectedGoalsModel",
    "xGTracker",
    "xGZoneAnalyzer",
    "ShotContext",
    "ShotType",
    "ShotSituation",
    "ShotOutcome",
    # Space
    "SpaceCreationAnalyzer",
    "ThirdManRunDetector",
    "OffBallMovementAnalyzer",
    "SpaceCalculator",
    "RunType",
    "SpaceType",
]
