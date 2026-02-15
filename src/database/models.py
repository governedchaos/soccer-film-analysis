"""
Soccer Film Analysis - Database Models
SQLAlchemy ORM models for storing analysis data
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    DateTime, ForeignKey, Text, JSON, Enum as SQLEnum, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import QueuePool
import enum

import logging

from config import settings

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()


# ============================================
# Enums
# ============================================

class TeamType(str, enum.Enum):
    """Team type enumeration"""
    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


class EventType(str, enum.Enum):
    """Game event types"""
    GOAL = "goal"
    SHOT_ON_TARGET = "shot_on_target"
    SHOT_OFF_TARGET = "shot_off_target"
    PASS_SUCCESSFUL = "pass_successful"
    PASS_UNSUCCESSFUL = "pass_unsuccessful"
    ASSIST = "assist"
    TURNOVER = "turnover"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    SAVE = "save"
    FOUL = "foul"
    CORNER = "corner"
    THROW_IN = "throw_in"
    FREE_KICK = "free_kick"
    PENALTY = "penalty"
    OFFSIDE = "offside"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"


class AnalysisStatus(str, enum.Enum):
    """Analysis session status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================
# Models
# ============================================

class Game(Base):
    """
    Stores game metadata
    """
    __tablename__ = "games"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Video file info
    video_path = Column(String(500), nullable=False)
    video_filename = Column(String(255), nullable=False)
    video_duration_seconds = Column(Float)
    video_fps = Column(Float)
    video_width = Column(Integer)
    video_height = Column(Integer)
    
    # Game info
    game_date = Column(DateTime)
    home_team_name = Column(String(100))
    away_team_name = Column(String(100))
    venue = Column(String(200))
    competition = Column(String(100))
    
    # Analysis metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    analysis_sessions = relationship("AnalysisSession", back_populates="game", cascade="all, delete-orphan")
    teams = relationship("Team", back_populates="game", cascade="all, delete-orphan")
    players = relationship("Player", back_populates="game", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="game", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Game(id={self.id}, home={self.home_team_name}, away={self.away_team_name})>"


class Team(Base):
    """
    Stores team information for a game
    """
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    
    team_type = Column(SQLEnum(TeamType), nullable=False)
    name = Column(String(100))
    
    # Jersey colors (RGB values stored as JSON)
    primary_color = Column(JSON)  # {"r": 255, "g": 0, "b": 0}
    secondary_color = Column(JSON)
    goalkeeper_color = Column(JSON)
    
    # Formation info
    formation = Column(String(20))  # e.g., "4-4-2", "4-3-3"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    game = relationship("Game", back_populates="teams")
    players = relationship("Player", back_populates="team")
    metrics = relationship("TeamMetrics", back_populates="team", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Team(id={self.id}, name={self.name}, type={self.team_type})>"


class Player(Base):
    """
    Stores player information
    """
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="SET NULL"))
    
    # Tracking ID assigned during analysis
    tracking_id = Column(Integer)
    
    # Player info
    jersey_number = Column(Integer)
    name = Column(String(100))
    position = Column(String(50))  # e.g., "goalkeeper", "defender", "midfielder", "forward"
    is_goalkeeper = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    game = relationship("Game", back_populates="players")
    team = relationship("Team", back_populates="players")
    tracking_data = relationship("TrackingData", back_populates="player", cascade="all, delete-orphan")
    metrics = relationship("PlayerMetrics", back_populates="player", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="player")
    
    # Indexes for faster queries
    __table_args__ = (
        Index("idx_player_game_tracking", "game_id", "tracking_id"),
    )
    
    def __repr__(self):
        return f"<Player(id={self.id}, jersey={self.jersey_number}, tracking_id={self.tracking_id})>"


class TrackingData(Base):
    """
    Frame-by-frame tracking data for players and ball
    """
    __tablename__ = "tracking_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"))
    
    # Frame info
    frame_number = Column(Integer, nullable=False)
    timestamp_seconds = Column(Float)
    
    # Position in video frame (pixels)
    x = Column(Float)
    y = Column(Float)
    width = Column(Float)
    height = Column(Float)
    
    # Position on pitch (normalized 0-1 or meters)
    pitch_x = Column(Float)
    pitch_y = Column(Float)
    
    # Detection confidence
    confidence = Column(Float)
    
    # Is this the ball?
    is_ball = Column(Boolean, default=False)
    
    # Speed at this frame (m/s)
    speed = Column(Float)
    
    # Indexes for faster queries
    __table_args__ = (
        Index("idx_tracking_game_frame", "game_id", "frame_number"),
        Index("idx_tracking_player_frame", "player_id", "frame_number"),
    )
    
    # Relationships
    player = relationship("Player", back_populates="tracking_data")
    
    def __repr__(self):
        return f"<TrackingData(frame={self.frame_number}, x={self.x}, y={self.y})>"


class Event(Base):
    """
    Detected game events
    """
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="SET NULL"))
    
    event_type = Column(SQLEnum(EventType), nullable=False)
    
    # Timing
    frame_number = Column(Integer)
    timestamp_seconds = Column(Float)
    game_minute = Column(Integer)  # Game time in minutes
    
    # Position where event occurred (normalized pitch coordinates)
    pitch_x = Column(Float)
    pitch_y = Column(Float)
    
    # Additional data (e.g., xG for shots, pass receiver ID)
    event_data = Column(JSON)
    
    # Manual verification
    is_verified = Column(Boolean, default=False)
    is_manual = Column(Boolean, default=False)  # Manually tagged by user
    
    confidence = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    game = relationship("Game", back_populates="events")
    player = relationship("Player", back_populates="events")
    
    # Indexes
    __table_args__ = (
        Index("idx_event_game_type", "game_id", "event_type"),
        Index("idx_event_timestamp", "game_id", "timestamp_seconds"),
    )
    
    def __repr__(self):
        return f"<Event(type={self.event_type}, minute={self.game_minute})>"


class PlayerMetrics(Base):
    """
    Aggregated player statistics for a game
    """
    __tablename__ = "player_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False)
    analysis_session_id = Column(Integer, ForeignKey("analysis_sessions.id", ondelete="CASCADE"))
    
    # Movement metrics
    total_distance_meters = Column(Float, default=0)
    max_speed_ms = Column(Float, default=0)  # meters per second
    avg_speed_ms = Column(Float, default=0)
    sprint_count = Column(Integer, default=0)
    
    # Ball involvement
    passes_attempted = Column(Integer, default=0)
    passes_completed = Column(Integer, default=0)
    pass_accuracy = Column(Float, default=0)
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    
    # Defensive metrics
    tackles = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    turnovers = Column(Integer, default=0)
    
    # Advanced metrics
    xg = Column(Float, default=0)  # Expected goals
    possession_percentage = Column(Float, default=0)
    
    # Position heatmap data (stored as JSON grid)
    heatmap_data = Column(JSON)
    
    # Overall performance rating (0-10)
    performance_rating = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    player = relationship("Player", back_populates="metrics")
    analysis_session = relationship("AnalysisSession", back_populates="player_metrics")
    
    def __repr__(self):
        return f"<PlayerMetrics(player={self.player_id}, distance={self.total_distance_meters}m)>"


class TeamMetrics(Base):
    """
    Aggregated team statistics for a game
    """
    __tablename__ = "team_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    analysis_session_id = Column(Integer, ForeignKey("analysis_sessions.id", ondelete="CASCADE"))
    
    # Possession
    possession_percentage = Column(Float, default=0)
    possession_in_final_third = Column(Float, default=0)
    
    # Passing
    total_passes = Column(Integer, default=0)
    passes_completed = Column(Integer, default=0)
    pass_accuracy = Column(Float, default=0)
    
    # Shooting
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    goals = Column(Integer, default=0)
    xg = Column(Float, default=0)
    
    # Defensive
    tackles = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    
    # Formation analysis
    primary_formation = Column(String(20))
    formation_changes = Column(JSON)  # List of formation changes with timestamps
    
    # Distance and movement
    total_team_distance = Column(Float, default=0)
    avg_team_speed = Column(Float, default=0)
    
    # Pressing metrics
    pressing_intensity = Column(Float)  # PPDA or similar
    high_press_count = Column(Integer, default=0)
    
    # Heatmap data
    team_heatmap = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="metrics")
    analysis_session = relationship("AnalysisSession", back_populates="team_metrics")
    
    def __repr__(self):
        return f"<TeamMetrics(team={self.team_id}, possession={self.possession_percentage}%)>"


class Formation(Base):
    """
    Formation snapshots at different points in the game
    """
    __tablename__ = "formations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    analysis_session_id = Column(Integer, ForeignKey("analysis_sessions.id", ondelete="CASCADE"))
    
    # Timing
    frame_number = Column(Integer)
    timestamp_seconds = Column(Float)
    game_minute = Column(Integer)
    
    # Formation string (e.g., "4-4-2")
    formation_string = Column(String(20))
    
    # Player positions in this formation (JSON)
    # {"player_id": {"x": 0.5, "y": 0.3, "role": "CB"}, ...}
    player_positions = Column(JSON)
    
    confidence = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis_session = relationship("AnalysisSession", back_populates="formations")
    
    def __repr__(self):
        return f"<Formation(formation={self.formation_string}, minute={self.game_minute})>"


class AnalysisSession(Base):
    """
    Metadata for each analysis run
    """
    __tablename__ = "analysis_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    
    # Analysis parameters
    analysis_depth = Column(String(20))  # quick, standard, deep
    frame_sample_rate = Column(Integer)
    
    # Timing
    start_frame = Column(Integer, default=0)
    end_frame = Column(Integer)
    
    # Status
    status = Column(SQLEnum(AnalysisStatus), default=AnalysisStatus.PENDING)
    progress_percentage = Column(Float, default=0)
    current_frame = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    processing_time_seconds = Column(Float)
    
    # Error info
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    game = relationship("Game", back_populates="analysis_sessions")
    player_metrics = relationship("PlayerMetrics", back_populates="analysis_session", cascade="all, delete-orphan")
    team_metrics = relationship("TeamMetrics", back_populates="analysis_session", cascade="all, delete-orphan")
    formations = relationship("Formation", back_populates="analysis_session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<AnalysisSession(id={self.id}, status={self.status}, progress={self.progress_percentage}%)>"


# ============================================
# Database Connection & Session Management
# ============================================

def get_engine():
    """Create and return database engine with connection pooling"""
    connection_string = settings.db_connection_string
    
    # SQLite doesn't support connection pooling
    if connection_string.startswith("sqlite"):
        return create_engine(
            connection_string,
            connect_args={"check_same_thread": False},  # Allow multi-threaded access
            echo=settings.log_level == "DEBUG"
        )
    
    # PostgreSQL with connection pooling
    return create_engine(
        connection_string,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Verify connections before using
        echo=settings.log_level == "DEBUG"
    )


def get_session_factory():
    """Create session factory"""
    engine = get_engine()
    return sessionmaker(bind=engine)


def init_database():
    """Initialize database - create all tables"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database initialized: %s", settings.db_name)


def drop_database():
    """Drop all tables - USE WITH CAUTION"""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.warning("Database tables dropped: %s", settings.db_name)


# Context manager for sessions
class DatabaseSession:
    """Context manager for database sessions"""
    
    def __init__(self):
        self._session_factory = get_session_factory()
        self._session = None
    
    def __enter__(self):
        self._session = self._session_factory()
        return self._session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._session.rollback()
        else:
            self._session.commit()
        self._session.close()
        return False


def get_db_session():
    """Get a database session context manager"""
    return DatabaseSession()
