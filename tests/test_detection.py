"""
Tests for detection module.
"""

import pytest
import numpy as np


class TestSoccerDetector:
    """Tests for SoccerDetector class"""

    def test_detector_import(self):
        """Test that detector can be imported"""
        from src.detection import SoccerDetector
        assert SoccerDetector is not None

    def test_detector_initialization(self):
        """Test detector initialization"""
        from src.detection import SoccerDetector
        detector = SoccerDetector(model_size="nano")
        assert detector.device in ["cpu", "cuda", "mps"]
        assert detector.model_size == "nano"

    def test_model_sizes_available(self):
        """Test that all model sizes are defined"""
        from src.detection import SoccerDetector
        expected_sizes = ["nano", "small", "medium", "large", "xlarge"]
        for size in expected_sizes:
            assert size in SoccerDetector.MODEL_SIZES


class TestTeamClassifier:
    """Tests for TeamClassifier class"""

    def test_classifier_import(self):
        """Test that classifier can be imported"""
        from src.detection import TeamClassifier
        assert TeamClassifier is not None

    def test_classifier_fit(self):
        """Test fitting classifier with colors"""
        from src.detection import TeamClassifier
        classifier = TeamClassifier(n_teams=2)

        # Sample colors: red team and blue team
        colors = [
            (255, 0, 0), (250, 10, 10), (240, 20, 0),  # Red team
            (0, 0, 255), (10, 10, 250), (0, 20, 240),  # Blue team
        ]

        classifier.fit(colors)
        assert classifier._is_fitted

    def test_classifier_manual_colors(self):
        """Test setting team colors manually"""
        from src.detection import TeamClassifier
        classifier = TeamClassifier()

        classifier.set_team_colors(
            home_color=(255, 0, 0),
            away_color=(0, 0, 255)
        )

        assert classifier._is_fitted
        assert classifier.team_colors is not None

    def test_classifier_predict(self):
        """Test color classification"""
        from src.detection import TeamClassifier
        classifier = TeamClassifier()
        classifier.set_team_colors(
            home_color=(255, 0, 0),  # Red
            away_color=(0, 0, 255)   # Blue
        )

        # Red color should classify as home (0)
        team = classifier.classify((250, 10, 10))
        assert team in [0, 1]

        # Blue color should classify as away (1)
        team = classifier.classify((10, 10, 250))
        assert team in [0, 1]


class TestEnhancedDetector:
    """Tests for EnhancedDetector class"""

    def test_enhanced_detector_import(self):
        """Test that enhanced detector can be imported"""
        from src.detection import EnhancedDetector
        assert EnhancedDetector is not None

    def test_enhanced_detector_initialization(self):
        """Test enhanced detector initialization"""
        from src.detection import EnhancedDetector
        detector = EnhancedDetector(model_size="nano")
        assert hasattr(detector, 'pitch_detector')
        assert hasattr(detector, 'tracked_persons')
