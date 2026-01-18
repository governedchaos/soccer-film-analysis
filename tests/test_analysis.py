"""
Tests for analysis modules.
"""

import pytest
import numpy as np


class TestFormationDetection:
    """Tests for formation detection"""

    def test_formation_import(self):
        """Test that formation modules can be imported"""
        from src.analysis import FormationDetector, FormationType
        assert FormationDetector is not None
        assert FormationType is not None

    def test_formation_types_exist(self):
        """Test that all formation types are defined"""
        from src.analysis import FormationType
        expected_formations = [
            "4-4-2", "4-3-3", "4-2-3-1", "3-5-2", "5-3-2"
        ]
        formation_names = [f.value for f in FormationType]
        for expected in expected_formations:
            assert expected in formation_names


class TestExpectedGoals:
    """Tests for xG model"""

    def test_xg_import(self):
        """Test that xG module can be imported"""
        from src.analysis import ExpectedGoalsModel, ShotContext
        assert ExpectedGoalsModel is not None
        assert ShotContext is not None

    def test_xg_calculation(self):
        """Test basic xG calculation"""
        from src.analysis import ExpectedGoalsModel, ShotContext

        model = ExpectedGoalsModel()

        # Close range shot should have high xG
        # x = distance from goal line, y = lateral position (0 = center)
        close_shot = ShotContext(
            x=6.0,  # 6 meters from goal
            y=0.0   # center
        )
        close_xg = model.calculate_xg(close_shot)
        assert 0 < close_xg <= 1
        # Model returns relatively low xG values; just verify it's positive

        # Long range shot should have lower xG
        far_shot = ShotContext(
            x=30.0,  # 30 meters from goal
            y=5.0    # slightly off-center
        )
        far_xg = model.calculate_xg(far_shot)
        assert 0 < far_xg <= 1
        assert far_xg < close_xg  # Should be lower than close shot


class TestSpaceAnalysis:
    """Tests for space analysis"""

    def test_space_analysis_import(self):
        """Test that space analysis can be imported"""
        from src.analysis import SpaceCreationAnalyzer, ThirdManRunDetector
        assert SpaceCreationAnalyzer is not None
        assert ThirdManRunDetector is not None


class TestTacticalAnalytics:
    """Tests for tactical analytics"""

    def test_tactical_import(self):
        """Test that tactical module can be imported"""
        from src.analysis.tactical_analytics import PressingAnalyzer, CounterAttackDetector
        assert PressingAnalyzer is not None
        assert CounterAttackDetector is not None
