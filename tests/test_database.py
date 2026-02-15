"""
Tests for database models and session management.

Uses in-memory SQLite to avoid requiring PostgreSQL for testing.
"""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import (
    Base,
    Game,
    Team,
    Player,
    TrackingData,
    Event,
    PlayerMetrics,
    TeamMetrics,
    Formation,
    AnalysisSession,
    TeamType,
    EventType,
    AnalysisStatus,
    DatabaseSession,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    """Create a database session for testing."""
    Session = sessionmaker(bind=engine)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture
def sample_game(session):
    """Create and return a sample game."""
    game = Game(
        video_path="/path/to/video.mp4",
        video_filename="video.mp4",
        video_duration_seconds=5400.0,
        video_fps=30.0,
        video_width=1920,
        video_height=1080,
        home_team_name="Home FC",
        away_team_name="Away United",
    )
    session.add(game)
    session.commit()
    return game


@pytest.fixture
def sample_team(session, sample_game):
    """Create and return a sample team."""
    team = Team(
        game_id=sample_game.id,
        team_type=TeamType.HOME,
        name="Home FC",
        primary_color={"r": 255, "g": 0, "b": 0},
        formation="4-4-2",
    )
    session.add(team)
    session.commit()
    return team


@pytest.fixture
def sample_player(session, sample_game, sample_team):
    """Create and return a sample player."""
    player = Player(
        game_id=sample_game.id,
        team_id=sample_team.id,
        tracking_id=7,
        jersey_number=10,
        name="Test Player",
        position="midfielder",
    )
    session.add(player)
    session.commit()
    return player


class TestEnums:
    """Test database enums."""

    def test_team_type_values(self):
        assert TeamType.HOME.value == "home"
        assert TeamType.AWAY.value == "away"
        assert TeamType.UNKNOWN.value == "unknown"

    def test_event_type_count(self):
        # All expected event types exist
        assert len(EventType) >= 18

    def test_event_type_values(self):
        assert EventType.GOAL.value == "goal"
        assert EventType.PENALTY.value == "penalty"
        assert EventType.RED_CARD.value == "red_card"

    def test_analysis_status_values(self):
        assert AnalysisStatus.PENDING.value == "pending"
        assert AnalysisStatus.PROCESSING.value == "processing"
        assert AnalysisStatus.COMPLETED.value == "completed"
        assert AnalysisStatus.FAILED.value == "failed"
        assert AnalysisStatus.CANCELLED.value == "cancelled"


class TestGameModel:
    """Test Game model."""

    def test_create_game(self, session, sample_game):
        assert sample_game.id is not None
        assert sample_game.video_filename == "video.mp4"
        assert sample_game.home_team_name == "Home FC"

    def test_game_repr(self, sample_game):
        r = repr(sample_game)
        assert "Home FC" in r
        assert "Away United" in r

    def test_game_created_at(self, sample_game):
        assert sample_game.created_at is not None

    def test_game_relationships(self, session, sample_game):
        assert sample_game.teams == []
        assert sample_game.players == []
        assert sample_game.events == []
        assert sample_game.analysis_sessions == []


class TestTeamModel:
    """Test Team model."""

    def test_create_team(self, session, sample_team):
        assert sample_team.id is not None
        assert sample_team.name == "Home FC"
        assert sample_team.team_type == TeamType.HOME

    def test_team_color_json(self, sample_team):
        assert sample_team.primary_color == {"r": 255, "g": 0, "b": 0}

    def test_team_repr(self, sample_team):
        r = repr(sample_team)
        assert "Home FC" in r

    def test_team_game_relationship(self, sample_team, sample_game):
        assert sample_team.game.id == sample_game.id


class TestPlayerModel:
    """Test Player model."""

    def test_create_player(self, sample_player):
        assert sample_player.id is not None
        assert sample_player.jersey_number == 10
        assert sample_player.name == "Test Player"

    def test_player_tracking_id(self, sample_player):
        assert sample_player.tracking_id == 7

    def test_player_repr(self, sample_player):
        r = repr(sample_player)
        assert "jersey=10" in r
        assert "tracking_id=7" in r

    def test_player_relationships(self, sample_player, sample_game, sample_team):
        assert sample_player.game.id == sample_game.id
        assert sample_player.team.id == sample_team.id

    def test_player_is_goalkeeper_default(self, sample_player):
        assert sample_player.is_goalkeeper is False


class TestTrackingDataModel:
    """Test TrackingData model."""

    def test_create_tracking_data(self, session, sample_game, sample_player):
        td = TrackingData(
            game_id=sample_game.id,
            player_id=sample_player.id,
            frame_number=100,
            timestamp_seconds=3.33,
            x=500.0,
            y=300.0,
            width=50.0,
            height=100.0,
            confidence=0.92,
            speed=5.5,
        )
        session.add(td)
        session.commit()
        assert td.id is not None
        assert td.frame_number == 100
        assert td.is_ball is False

    def test_tracking_data_ball(self, session, sample_game):
        td = TrackingData(
            game_id=sample_game.id,
            frame_number=50,
            x=400.0,
            y=200.0,
            is_ball=True,
            confidence=0.8,
        )
        session.add(td)
        session.commit()
        assert td.is_ball is True

    def test_tracking_data_repr(self, session, sample_game):
        td = TrackingData(
            game_id=sample_game.id,
            frame_number=10,
            x=100.0,
            y=200.0,
        )
        session.add(td)
        session.commit()
        r = repr(td)
        assert "frame=10" in r


class TestEventModel:
    """Test Event model."""

    def test_create_event(self, session, sample_game, sample_player):
        event = Event(
            game_id=sample_game.id,
            player_id=sample_player.id,
            event_type=EventType.GOAL,
            frame_number=4500,
            timestamp_seconds=150.0,
            game_minute=75,
            pitch_x=0.9,
            pitch_y=0.5,
            confidence=0.95,
        )
        session.add(event)
        session.commit()
        assert event.id is not None
        assert event.event_type == EventType.GOAL
        assert event.game_minute == 75

    def test_event_repr(self, session, sample_game):
        event = Event(
            game_id=sample_game.id,
            event_type=EventType.SHOT_ON_TARGET,
            game_minute=60,
        )
        session.add(event)
        session.commit()
        r = repr(event)
        assert "minute=60" in r

    def test_event_json_data(self, session, sample_game):
        event = Event(
            game_id=sample_game.id,
            event_type=EventType.GOAL,
            event_data={"xg": 0.85, "assist_player_id": 3},
        )
        session.add(event)
        session.commit()
        assert event.event_data["xg"] == 0.85

    def test_event_manual_and_verified(self, session, sample_game):
        event = Event(
            game_id=sample_game.id,
            event_type=EventType.FOUL,
            is_verified=True,
            is_manual=True,
        )
        session.add(event)
        session.commit()
        assert event.is_verified is True
        assert event.is_manual is True


class TestPlayerMetricsModel:
    """Test PlayerMetrics model."""

    def test_create_player_metrics(self, session, sample_game, sample_player):
        metrics = PlayerMetrics(
            game_id=sample_game.id,
            player_id=sample_player.id,
            total_distance_meters=10500.0,
            max_speed_ms=8.5,
            avg_speed_ms=3.2,
            sprint_count=15,
            passes_attempted=45,
            passes_completed=38,
            shots=3,
            goals=1,
            xg=0.95,
        )
        session.add(metrics)
        session.commit()
        assert metrics.id is not None
        assert metrics.total_distance_meters == 10500.0
        assert metrics.pass_accuracy == 0  # Default

    def test_player_metrics_repr(self, session, sample_game, sample_player):
        metrics = PlayerMetrics(
            game_id=sample_game.id,
            player_id=sample_player.id,
            total_distance_meters=8000.0,
        )
        session.add(metrics)
        session.commit()
        r = repr(metrics)
        assert "distance=8000.0m" in r


class TestTeamMetricsModel:
    """Test TeamMetrics model."""

    def test_create_team_metrics(self, session, sample_game, sample_team):
        metrics = TeamMetrics(
            game_id=sample_game.id,
            team_id=sample_team.id,
            possession_percentage=55.3,
            total_passes=420,
            passes_completed=370,
            pass_accuracy=88.1,
            shots=12,
            goals=2,
            xg=1.85,
            primary_formation="4-3-3",
        )
        session.add(metrics)
        session.commit()
        assert metrics.id is not None
        assert metrics.possession_percentage == 55.3

    def test_team_metrics_repr(self, session, sample_game, sample_team):
        metrics = TeamMetrics(
            game_id=sample_game.id,
            team_id=sample_team.id,
            possession_percentage=60.0,
        )
        session.add(metrics)
        session.commit()
        r = repr(metrics)
        assert "possession=60.0%" in r


class TestFormationModel:
    """Test Formation model."""

    def test_create_formation(self, session, sample_game, sample_team):
        formation = Formation(
            game_id=sample_game.id,
            team_id=sample_team.id,
            frame_number=900,
            timestamp_seconds=30.0,
            game_minute=30,
            formation_string="4-3-3",
            confidence=0.88,
            player_positions={"1": {"x": 0.5, "y": 0.1, "role": "GK"}},
        )
        session.add(formation)
        session.commit()
        assert formation.id is not None
        assert formation.formation_string == "4-3-3"

    def test_formation_repr(self, session, sample_game, sample_team):
        formation = Formation(
            game_id=sample_game.id,
            team_id=sample_team.id,
            formation_string="4-4-2",
            game_minute=45,
        )
        session.add(formation)
        session.commit()
        r = repr(formation)
        assert "4-4-2" in r
        assert "minute=45" in r


class TestAnalysisSessionModel:
    """Test AnalysisSession model."""

    def test_create_session(self, session, sample_game):
        analysis = AnalysisSession(
            game_id=sample_game.id,
            analysis_depth="standard",
            frame_sample_rate=5,
            status=AnalysisStatus.PROCESSING,
            progress_percentage=50.0,
        )
        session.add(analysis)
        session.commit()
        assert analysis.id is not None
        assert analysis.status == AnalysisStatus.PROCESSING

    def test_session_defaults(self, session, sample_game):
        analysis = AnalysisSession(game_id=sample_game.id)
        session.add(analysis)
        session.commit()
        assert analysis.status == AnalysisStatus.PENDING
        assert analysis.progress_percentage == 0
        assert analysis.current_frame == 0

    def test_session_repr(self, session, sample_game):
        analysis = AnalysisSession(
            game_id=sample_game.id,
            status=AnalysisStatus.COMPLETED,
            progress_percentage=100.0,
        )
        session.add(analysis)
        session.commit()
        r = repr(analysis)
        assert "100.0%" in r


class TestCascadeDelete:
    """Test that cascade deletes work properly."""

    def test_delete_game_cascades_teams(self, session, sample_game, sample_team):
        game_id = sample_game.id
        session.delete(sample_game)
        session.commit()
        assert session.query(Team).filter_by(game_id=game_id).count() == 0

    def test_delete_game_cascades_events(self, session, sample_game):
        event = Event(
            game_id=sample_game.id,
            event_type=EventType.CORNER,
        )
        session.add(event)
        session.commit()
        game_id = sample_game.id
        session.delete(sample_game)
        session.commit()
        assert session.query(Event).filter_by(game_id=game_id).count() == 0

    def test_delete_game_cascades_analysis_sessions(self, session, sample_game):
        analysis = AnalysisSession(game_id=sample_game.id)
        session.add(analysis)
        session.commit()
        game_id = sample_game.id
        session.delete(sample_game)
        session.commit()
        assert session.query(AnalysisSession).filter_by(game_id=game_id).count() == 0


class TestVersion:
    """Test version is set."""

    def test_version_exists(self):
        from src import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format(self):
        from src import __version__
        parts = __version__.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()
