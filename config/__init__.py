"""
Configuration module for Soccer Film Analysis
"""

from .settings import Settings, settings, get_settings, reload_settings, AnalysisDepth

__all__ = [
    "Settings",
    "settings", 
    "get_settings",
    "reload_settings",
    "AnalysisDepth"
]
