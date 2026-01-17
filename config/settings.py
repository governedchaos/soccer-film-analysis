"""
Soccer Film Analysis - Application Settings
Uses Pydantic for type-safe configuration management
"""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class AnalysisDepth(str, Enum):
    """Analysis depth levels"""
    QUICK = "quick"       # Basic detection only (~5-10 min)
    STANDARD = "standard" # Detection + core events (~20-30 min)
    DEEP = "deep"         # Full analytics (~45-60 min)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    All settings can be overridden via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==========================================
    # Database Configuration (PostgreSQL)
    # ==========================================
    db_host: str = Field(default="localhost", description="PostgreSQL host")
    db_port: int = Field(default=5432, description="PostgreSQL port")
    db_name: str = Field(default="soccer_analysis", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="", description="Database password")
    database_url: Optional[str] = Field(default=None, description="Full connection string (overrides individual settings)")
    
    @property
    def db_connection_string(self) -> str:
        """Get the PostgreSQL database connection string"""
        # Use explicit DATABASE_URL if provided (e.g., for cloud deployments)
        if self.database_url:
            return self.database_url
        
        # Build connection string from individual settings
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    # ==========================================
    # API Keys
    # ==========================================
    roboflow_api_key: str = Field(default="", description="Roboflow API key for detection models")
    
    # ==========================================
    # Logging
    # ==========================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", 
        description="Application logging level"
    )
    
    # ==========================================
    # Processing Settings
    # ==========================================
    default_analysis_depth: AnalysisDepth = Field(
        default=AnalysisDepth.STANDARD,
        description="Default analysis depth"
    )
    max_processing_threads: int = Field(
        default=4, 
        ge=1, 
        le=16,
        description="Maximum threads for video processing"
    )
    enable_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    
    # Frame sampling rates for different analysis depths
    # (process every Nth frame)
    frame_sample_quick: int = Field(default=10, description="Frame sample rate for quick analysis")
    frame_sample_standard: int = Field(default=5, description="Frame sample rate for standard analysis")
    frame_sample_deep: int = Field(default=2, description="Frame sample rate for deep analysis")
    
    # ==========================================
    # Paths
    # ==========================================
    video_input_dir: str = Field(default="data/videos", description="Video input directory")
    output_dir: str = Field(default="data/outputs", description="Output directory")
    models_dir: str = Field(default="data/models", description="Models directory")
    logs_dir: str = Field(default="logs", description="Logs directory")
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory"""
        return Path(__file__).parent.parent
    
    def get_video_dir(self) -> Path:
        """Get absolute path to video directory"""
        path = self.project_root / self.video_input_dir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_output_dir(self) -> Path:
        """Get absolute path to output directory"""
        path = self.project_root / self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_models_dir(self) -> Path:
        """Get absolute path to models directory"""
        path = self.project_root / self.models_dir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_logs_dir(self) -> Path:
        """Get absolute path to logs directory"""
        path = self.project_root / self.logs_dir
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # ==========================================
    # Detection Thresholds
    # ==========================================
    player_confidence_threshold: float = Field(
        default=0.3, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence for player detection"
    )
    ball_confidence_threshold: float = Field(
        default=0.25, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence for ball detection"
    )
    pitch_confidence_threshold: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence for pitch keypoint detection"
    )
    referee_confidence_threshold: float = Field(
        default=0.35, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence for referee detection"
    )
    
    # ==========================================
    # GUI Settings
    # ==========================================
    window_width: int = Field(default=1600, ge=800, description="Window width")
    window_height: int = Field(default=900, ge=600, description="Window height")
    theme: Literal["light", "dark"] = Field(default="dark", description="UI theme")
    
    # ==========================================
    # Model Configuration
    # ==========================================
    # Roboflow model IDs for soccer detection
    player_detection_model: str = Field(
        default="football-players-detection-3zvbc/10",
        description="Roboflow model ID for player detection"
    )
    pitch_detection_model: str = Field(
        default="football-field-detection-f07vi/14",
        description="Roboflow model ID for pitch detection"
    )
    ball_detection_model: str = Field(
        default="football-ball-detection-rejhg/2",
        description="Roboflow model ID for ball detection"
    )
    
    def get_frame_sample_rate(self, depth: Optional[AnalysisDepth] = None) -> int:
        """Get the frame sampling rate based on analysis depth"""
        depth = depth or self.default_analysis_depth
        rates = {
            AnalysisDepth.QUICK: self.frame_sample_quick,
            AnalysisDepth.STANDARD: self.frame_sample_standard,
            AnalysisDepth.DEEP: self.frame_sample_deep,
        }
        return rates.get(depth, self.frame_sample_standard)
    
    def get_device(self) -> str:
        """Get the compute device (cuda, mps, or cpu)"""
        if not self.enable_gpu:
            return "cpu"
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
        except ImportError:
            pass
        
        return "cpu"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global settings
    settings = Settings()
    return settings
