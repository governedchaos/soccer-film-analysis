"""
Configuration module for Soccer Film Analysis
"""

from .settings import (
    Settings,
    settings,
    get_settings,
    reload_settings,
    AnalysisDepth,
    GPUMemoryManager,
    gpu_memory_manager,
    get_gpu_memory_manager,
    ConfigPersistence,
    config_persistence,
    get_config_persistence
)

__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "reload_settings",
    "AnalysisDepth",
    "GPUMemoryManager",
    "gpu_memory_manager",
    "get_gpu_memory_manager",
    "ConfigPersistence",
    "config_persistence",
    "get_config_persistence"
]
