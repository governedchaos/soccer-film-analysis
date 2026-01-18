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

# Analysis pipeline
from .pipeline import (
    AnalysisPipeline,
    AnalysisPipelineConfig,
    AnalysisDepthLevel,
    FrameAnalysisResult,
    MatchAnalysisSummary
)

# Advanced analytics (heatmaps, pass networks, shots, speed)
from .advanced_analytics import (
    HeatmapGenerator,
    PassNetworkAnalyzer,
    ShotDetector,
    SpeedDistanceTracker,
    PossessionSequenceTracker,
    OffsideLine,
    PlayerPosition,
    PassEvent,
    ShotEvent,
    PossessionSequence
)

# Auto-save functionality
from .auto_save import (
    AutoSaveManager,
    AnalysisStateManager
)

# Game comparison
from .game_comparison import (
    GameStats,
    GameComparisonAnalyzer
)

# Game period detection
from .game_periods import (
    GamePeriod,
    PeriodMarker,
    GamePeriodDetector
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
    # Pipeline
    "AnalysisPipeline",
    "AnalysisPipelineConfig",
    "AnalysisDepthLevel",
    "FrameAnalysisResult",
    "MatchAnalysisSummary",
    # Advanced analytics
    "HeatmapGenerator",
    "PassNetworkAnalyzer",
    "ShotDetector",
    "SpeedDistanceTracker",
    "PossessionSequenceTracker",
    "OffsideLine",
    "PlayerPosition",
    "PassEvent",
    "ShotEvent",
    "PossessionSequence",
    # Auto-save
    "AutoSaveManager",
    "AnalysisStateManager",
    # Game comparison
    "GameStats",
    "GameComparisonAnalyzer",
    # Game periods
    "GamePeriod",
    "PeriodMarker",
    "GamePeriodDetector",
]
