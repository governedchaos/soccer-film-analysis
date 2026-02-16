"""
Soccer Film Analysis - Application Settings
Uses Pydantic for type-safe configuration management
"""

import os
import json
import logging
from pathlib import Path
from typing import Literal, Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


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
    # API Keys (Optional - local models work without API keys)
    # ==========================================
    roboflow_api_key: str = Field(default="", description="Roboflow API key (optional - only needed for Roboflow models)")
    
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

    # GPU Memory Management
    gpu_memory_limit_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Max GPU memory to use in GB (0 = no limit)"
    )
    gpu_memory_fraction: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Fraction of available GPU memory to use (0.1-1.0)"
    )
    gpu_cache_clear_interval: int = Field(
        default=100,
        ge=10,
        description="Clear GPU cache every N frames during processing"
    )
    auto_adjust_batch_size: bool = Field(
        default=True,
        description="Automatically adjust batch size based on available GPU memory"
    )

    # Frame sampling rates for different analysis depths
    # (process every Nth frame)
    frame_sample_quick: int = Field(default=10, description="Frame sample rate for quick analysis")
    frame_sample_standard: int = Field(default=5, description="Frame sample rate for standard analysis")
    frame_sample_deep: int = Field(default=2, description="Frame sample rate for deep analysis")

    # Inference resolution: downscale frames before YOLO inference for speed.
    # Detection bboxes are mapped back to original resolution automatically.
    # 0 = no downscaling (use original resolution). Recommended: 640 for CPU, 960 for GPU.
    inference_resolution: int = Field(
        default=0,
        ge=0,
        description="Max width for YOLO inference (0 = original resolution)"
    )

    # Auto-select YOLO model size based on hardware
    yolo_model_size_cpu: str = Field(
        default="nano",
        description="YOLO model size when running on CPU (nano is 2x faster than small)"
    )

    # Simplified annotations for faster playback
    simplified_annotations: bool = Field(
        default=False,
        description="Use dots instead of rectangles+text for faster rendering"
    )
    
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
    # Local YOLO model settings (no API key required)
    yolo_model_size: str = Field(
        default="small",
        description="YOLO model size: nano, small, medium, large, xlarge"
    )
    use_local_models: bool = Field(
        default=True,
        description="Use local YOLO models (True) or Roboflow API (False)"
    )

    # Legacy Roboflow model IDs (only used if use_local_models=False)
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

    def get_effective_inference_resolution(self) -> int:
        """Get the effective inference resolution, auto-selecting if not set."""
        if self.inference_resolution > 0:
            return self.inference_resolution
        # Auto-select based on device
        device = self.get_device()
        if device == "cpu":
            return 640  # Much faster on CPU
        elif device == "mps":
            return 960  # Good balance for Apple Silicon
        else:
            return 0  # CUDA can handle full res

    def get_effective_yolo_model_size(self) -> str:
        """Get the YOLO model size, auto-selecting based on hardware."""
        device = self.get_device()
        if device == "cpu":
            return self.yolo_model_size_cpu
        return self.yolo_model_size
    
    def get_device(self) -> str:
        """Get the compute device (cuda, mps, or cpu)"""
        if not self.enable_gpu:
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
        except ImportError:
            pass

        return "cpu"

    def get_device_info(self) -> dict:
        """Get detailed device information for diagnostics"""
        info = {
            "device": "cpu",
            "gpu_enabled_in_settings": self.enable_gpu,
            "cuda_available": False,
            "cuda_version": None,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "mps_available": False,
            "torch_version": None,
            "recommendation": None
        }

        try:
            import torch
            info["torch_version"] = torch.__version__

            # Check CUDA
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_version"] = torch.version.cuda
                info["cuda_device_count"] = torch.cuda.device_count()
                if info["cuda_device_count"] > 0:
                    info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["device"] = "cuda"

            # Check MPS (Apple Silicon)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info["mps_available"] = True
                info["device"] = "mps"

            # Recommendations
            if not info["cuda_available"] and not info["mps_available"]:
                info["recommendation"] = (
                    "No GPU detected. For NVIDIA GPUs, install PyTorch with CUDA: "
                    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                )
            elif not self.enable_gpu:
                info["recommendation"] = "GPU available but disabled in settings. Set ENABLE_GPU=true in .env"

        except ImportError:
            info["recommendation"] = "PyTorch not installed. Run: pip install torch torchvision"

        return info


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


class GPUMemoryManager:
    """
    Utility class for managing GPU memory during video processing.

    Provides methods to:
    - Monitor GPU memory usage
    - Clear GPU cache
    - Estimate optimal batch sizes
    - Set memory limits
    """

    def __init__(self):
        self._torch_available = False
        self._cuda_available = False
        self._mps_available = False
        self._last_clear_frame = 0

        try:
            import torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
            self._mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            pass

    def get_memory_info(self) -> dict:
        """
        Get current GPU memory usage information.

        Returns:
            Dict with 'allocated_gb', 'cached_gb', 'total_gb', 'free_gb'
        """
        info = {
            'allocated_gb': 0.0,
            'cached_gb': 0.0,
            'total_gb': 0.0,
            'free_gb': 0.0,
            'device': 'cpu'
        }

        if not self._torch_available:
            return info

        import torch

        if self._cuda_available:
            info['device'] = 'cuda'
            info['allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            info['cached_gb'] = torch.cuda.memory_reserved() / 1e9
            total_memory = torch.cuda.get_device_properties(0).total_memory
            info['total_gb'] = total_memory / 1e9
            info['free_gb'] = info['total_gb'] - info['cached_gb']
        elif self._mps_available:
            info['device'] = 'mps'
            # MPS doesn't expose memory info directly
            # Estimate based on system memory
            info['estimated'] = True

        return info

    def clear_cache(self, force: bool = False):
        """
        Clear GPU memory cache.

        Args:
            force: Force cache clearing even if not needed
        """
        if not self._torch_available:
            return

        import torch

        if self._cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self._mps_available:
            # MPS has limited cache control
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

    def should_clear_cache(self, current_frame: int) -> bool:
        """
        Check if cache should be cleared based on frame count.

        Args:
            current_frame: Current frame number being processed

        Returns:
            True if cache should be cleared
        """
        interval = settings.gpu_cache_clear_interval
        if current_frame - self._last_clear_frame >= interval:
            self._last_clear_frame = current_frame
            return True
        return False

    def maybe_clear_cache(self, current_frame: int):
        """
        Clear cache if interval has been reached.

        Args:
            current_frame: Current frame number
        """
        if self.should_clear_cache(current_frame):
            self.clear_cache()

    def estimate_batch_size(self, frame_size_mb: float = 6.0) -> int:
        """
        Estimate optimal batch size based on available GPU memory.

        Args:
            frame_size_mb: Approximate memory per frame in MB (default ~6MB for 1080p)

        Returns:
            Recommended batch size
        """
        if not self._cuda_available:
            return 1 if not self._mps_available else 8

        memory_info = self.get_memory_info()
        free_gb = memory_info['free_gb']

        # Use configured fraction of available memory
        usable_gb = free_gb * settings.gpu_memory_fraction

        # Apply hard limit if set
        if settings.gpu_memory_limit_gb > 0:
            usable_gb = min(usable_gb, settings.gpu_memory_limit_gb)

        # Reserve some memory for model weights and overhead (~2GB)
        usable_gb = max(0, usable_gb - 2.0)

        # Calculate batch size (convert GB to MB)
        usable_mb = usable_gb * 1024
        batch_size = max(1, int(usable_mb / frame_size_mb))

        # Cap at reasonable maximum
        return min(batch_size, 32)

    def set_memory_limit(self, limit_gb: float):
        """
        Set GPU memory limit (CUDA only).

        Args:
            limit_gb: Memory limit in GB
        """
        if not self._cuda_available or not self._torch_available:
            return

        import torch

        if limit_gb > 0:
            limit_bytes = int(limit_gb * 1e9)
            torch.cuda.set_per_process_memory_fraction(
                limit_bytes / torch.cuda.get_device_properties(0).total_memory
            )


# Global GPU memory manager instance
gpu_memory_manager = GPUMemoryManager()


def get_gpu_memory_manager() -> GPUMemoryManager:
    """Get the global GPU memory manager instance"""
    return gpu_memory_manager


class ConfigPersistence:
    """
    Handles saving and loading user configuration between sessions.

    Configuration is stored in a JSON file in the user's config directory.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the configuration persistence manager.

        Args:
            config_dir: Directory to store config files (default: project_root/config/user)
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "user"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.config_dir / "user_config.json"
        self.recent_files_path = self.config_dir / "recent_files.json"
        self.team_presets_path = self.config_dir / "team_presets.json"

    def save_user_config(self, config: Dict[str, Any]) -> bool:
        """
        Save user configuration to file.

        Args:
            config: Configuration dictionary to save

        Returns:
            True if saved successfully
        """
        try:
            # Add metadata
            config_with_meta = {
                "_saved_at": datetime.now().isoformat(),
                "_version": "1.0",
                **config
            }
            with open(self.config_file, 'w') as f:
                json.dump(config_with_meta, f, indent=2)
            return True
        except Exception as e:
            logger.error("Failed to save config: %s", e)
            return False

    def load_user_config(self) -> Dict[str, Any]:
        """
        Load user configuration from file.

        Returns:
            Configuration dictionary (empty dict if file doesn't exist)
        """
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            # Remove metadata
            config.pop("_saved_at", None)
            config.pop("_version", None)
            return config
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return {}

    def save_recent_files(self, recent_files: list) -> bool:
        """
        Save recent file list.

        Args:
            recent_files: List of recent file paths (most recent first)

        Returns:
            True if saved successfully
        """
        try:
            # Limit to 10 recent files
            recent_files = recent_files[:10]
            with open(self.recent_files_path, 'w') as f:
                json.dump({"recent_files": recent_files}, f, indent=2)
            return True
        except Exception:
            return False

    def load_recent_files(self) -> list:
        """
        Load recent file list.

        Returns:
            List of recent file paths
        """
        if not self.recent_files_path.exists():
            return []

        try:
            with open(self.recent_files_path, 'r') as f:
                data = json.load(f)
            return data.get("recent_files", [])
        except Exception:
            return []

    def add_recent_file(self, file_path: str) -> list:
        """
        Add a file to the recent files list.

        Args:
            file_path: Path to add

        Returns:
            Updated recent files list
        """
        recent = self.load_recent_files()

        # Remove if already exists
        if file_path in recent:
            recent.remove(file_path)

        # Add to front
        recent.insert(0, file_path)

        # Save and return
        self.save_recent_files(recent)
        return recent

    def save_team_preset(self, preset_name: str, team_config: Dict[str, Any]) -> bool:
        """
        Save a team color/name preset for reuse.

        Args:
            preset_name: Name for this preset
            team_config: Team configuration dict

        Returns:
            True if saved successfully
        """
        try:
            # Load existing presets
            presets = self.load_team_presets()

            # Add/update preset
            presets[preset_name] = {
                "saved_at": datetime.now().isoformat(),
                "config": team_config
            }

            with open(self.team_presets_path, 'w') as f:
                json.dump(presets, f, indent=2)
            return True
        except Exception:
            return False

    def load_team_presets(self) -> Dict[str, Any]:
        """
        Load all team presets.

        Returns:
            Dict of preset_name -> preset_config
        """
        if not self.team_presets_path.exists():
            return {}

        try:
            with open(self.team_presets_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def get_team_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific team preset by name.

        Args:
            preset_name: Name of the preset

        Returns:
            Preset config or None if not found
        """
        presets = self.load_team_presets()
        preset = presets.get(preset_name)
        if preset:
            return preset.get("config")
        return None

    def delete_team_preset(self, preset_name: str) -> bool:
        """
        Delete a team preset.

        Args:
            preset_name: Name of preset to delete

        Returns:
            True if deleted
        """
        presets = self.load_team_presets()
        if preset_name in presets:
            del presets[preset_name]
            try:
                with open(self.team_presets_path, 'w') as f:
                    json.dump(presets, f, indent=2)
                return True
            except Exception:
                pass
        return False


# Global config persistence instance
config_persistence = ConfigPersistence()


def get_config_persistence() -> ConfigPersistence:
    """Get the global config persistence instance"""
    return config_persistence
