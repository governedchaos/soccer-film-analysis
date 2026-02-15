"""
Tests for configuration module — Settings, GPUMemoryManager, ConfigPersistence.
"""

import json
import pytest
from pathlib import Path

from config.settings import (
    Settings,
    AnalysisDepth,
    GPUMemoryManager,
    ConfigPersistence,
    get_settings,
    reload_settings,
    get_gpu_memory_manager,
    get_config_persistence,
)


class TestAnalysisDepth:
    """Test AnalysisDepth enum."""

    def test_enum_values(self):
        assert AnalysisDepth.QUICK.value == "quick"
        assert AnalysisDepth.STANDARD.value == "standard"
        assert AnalysisDepth.DEEP.value == "deep"

    def test_enum_count(self):
        assert len(AnalysisDepth) == 3


class TestSettings:
    """Test Settings class defaults and methods."""

    def test_default_db_settings(self):
        s = Settings()
        assert s.db_host == "localhost"
        assert s.db_port == 5432
        assert s.db_name == "soccer_analysis"

    def test_db_connection_string(self):
        s = Settings()
        conn = s.db_connection_string
        assert "postgresql://" in conn
        assert "soccer_analysis" in conn

    def test_db_connection_string_override(self):
        s = Settings(database_url="sqlite:///test.db")
        assert s.db_connection_string == "sqlite:///test.db"

    def test_default_processing_settings(self):
        s = Settings()
        assert s.default_analysis_depth == AnalysisDepth.STANDARD
        assert 1 <= s.max_processing_threads <= 16
        assert s.enable_gpu is True

    def test_gpu_memory_settings(self):
        s = Settings()
        assert s.gpu_memory_limit_gb >= 0.0
        assert 0.1 <= s.gpu_memory_fraction <= 1.0
        assert s.gpu_cache_clear_interval >= 10

    def test_detection_thresholds_in_range(self):
        s = Settings()
        for threshold in [
            s.player_confidence_threshold,
            s.ball_confidence_threshold,
            s.pitch_confidence_threshold,
            s.referee_confidence_threshold,
        ]:
            assert 0.0 <= threshold <= 1.0

    def test_gui_settings(self):
        s = Settings()
        assert s.window_width >= 800
        assert s.window_height >= 600
        assert s.theme in ["light", "dark"]

    def test_frame_sample_rate_quick(self):
        s = Settings()
        rate = s.get_frame_sample_rate(AnalysisDepth.QUICK)
        assert rate == s.frame_sample_quick

    def test_frame_sample_rate_standard(self):
        s = Settings()
        rate = s.get_frame_sample_rate(AnalysisDepth.STANDARD)
        assert rate == s.frame_sample_standard

    def test_frame_sample_rate_deep(self):
        s = Settings()
        rate = s.get_frame_sample_rate(AnalysisDepth.DEEP)
        assert rate == s.frame_sample_deep

    def test_frame_sample_rate_default(self):
        s = Settings()
        rate = s.get_frame_sample_rate()
        assert rate == s.frame_sample_standard  # default is STANDARD

    def test_get_device(self):
        s = Settings()
        device = s.get_device()
        assert device in ["cpu", "cuda", "mps"]

    def test_get_device_gpu_disabled(self):
        s = Settings(enable_gpu=False)
        assert s.get_device() == "cpu"

    def test_get_device_info_keys(self):
        s = Settings()
        info = s.get_device_info()
        expected_keys = [
            "device", "gpu_enabled_in_settings", "cuda_available",
            "cuda_version", "cuda_device_count", "cuda_device_name",
            "mps_available", "torch_version", "recommendation",
        ]
        for key in expected_keys:
            assert key in info

    def test_project_root(self):
        s = Settings()
        root = s.project_root
        assert root.exists()
        assert (root / "src").exists()

    def test_model_config_extra_ignore(self):
        """Settings ignores unknown env variables."""
        s = Settings()
        assert s.model_config["extra"] == "ignore"


class TestSettingsModuleFunctions:
    """Test module-level settings functions."""

    def test_get_settings(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_reload_settings(self):
        s1 = get_settings()
        s2 = reload_settings()
        assert isinstance(s2, Settings)
        # Both should be valid Settings instances
        assert s2.db_name == "soccer_analysis"


class TestGPUMemoryManager:
    """Test GPUMemoryManager class."""

    def test_initialization(self):
        mgr = GPUMemoryManager()
        assert mgr._last_clear_frame == 0

    def test_get_memory_info(self):
        mgr = GPUMemoryManager()
        info = mgr.get_memory_info()
        assert "allocated_gb" in info
        assert "cached_gb" in info
        assert "total_gb" in info
        assert "free_gb" in info
        assert "device" in info
        assert info["device"] in ["cpu", "cuda", "mps"]

    def test_clear_cache_no_error(self):
        mgr = GPUMemoryManager()
        # Should not raise even without GPU
        mgr.clear_cache()
        mgr.clear_cache(force=True)

    def test_should_clear_cache_interval(self):
        mgr = GPUMemoryManager()
        # _last_clear_frame starts at 0, so frame 0 means 0-0 < interval → False
        assert mgr.should_clear_cache(0) is False
        assert mgr.should_clear_cache(1) is False
        assert mgr.should_clear_cache(50) is False

    def test_should_clear_cache_after_interval(self):
        mgr = GPUMemoryManager()
        from config.settings import settings
        interval = settings.gpu_cache_clear_interval
        # Once we reach the interval, should trigger
        assert mgr.should_clear_cache(interval) is True
        # Then immediately after should not trigger again
        assert mgr.should_clear_cache(interval + 1) is False

    def test_maybe_clear_cache_no_error(self):
        mgr = GPUMemoryManager()
        mgr.maybe_clear_cache(0)
        mgr.maybe_clear_cache(500)

    def test_estimate_batch_size(self):
        mgr = GPUMemoryManager()
        batch = mgr.estimate_batch_size()
        assert isinstance(batch, int)
        assert batch >= 1

    def test_estimate_batch_size_custom_frame_size(self):
        mgr = GPUMemoryManager()
        batch = mgr.estimate_batch_size(frame_size_mb=12.0)
        assert isinstance(batch, int)
        assert batch >= 1

    def test_set_memory_limit_no_error(self):
        mgr = GPUMemoryManager()
        # Should not raise even without CUDA
        mgr.set_memory_limit(4.0)
        mgr.set_memory_limit(0.0)

    def test_get_gpu_memory_manager(self):
        mgr = get_gpu_memory_manager()
        assert isinstance(mgr, GPUMemoryManager)


class TestConfigPersistence:
    """Test ConfigPersistence class."""

    @pytest.fixture
    def config_dir(self, tmp_path):
        """Create a temporary config directory."""
        return tmp_path / "config"

    @pytest.fixture
    def persistence(self, config_dir):
        return ConfigPersistence(config_dir=config_dir)

    def test_initialization_creates_dir(self, config_dir):
        ConfigPersistence(config_dir=config_dir)
        assert config_dir.exists()

    def test_save_and_load_user_config(self, persistence):
        config = {"theme": "dark", "window_width": 1600}
        assert persistence.save_user_config(config) is True
        loaded = persistence.load_user_config()
        assert loaded["theme"] == "dark"
        assert loaded["window_width"] == 1600

    def test_load_user_config_empty(self, persistence):
        loaded = persistence.load_user_config()
        assert loaded == {}

    def test_save_user_config_metadata_stripped(self, persistence):
        persistence.save_user_config({"key": "value"})
        loaded = persistence.load_user_config()
        assert "_saved_at" not in loaded
        assert "_version" not in loaded

    def test_save_and_load_recent_files(self, persistence):
        files = ["/path/to/a.mp4", "/path/to/b.mp4"]
        assert persistence.save_recent_files(files) is True
        loaded = persistence.load_recent_files()
        assert loaded == files

    def test_load_recent_files_empty(self, persistence):
        loaded = persistence.load_recent_files()
        assert loaded == []

    def test_recent_files_limit(self, persistence):
        files = [f"/path/to/video_{i}.mp4" for i in range(20)]
        persistence.save_recent_files(files)
        loaded = persistence.load_recent_files()
        assert len(loaded) == 10  # Limited to 10

    def test_add_recent_file(self, persistence):
        persistence.add_recent_file("/path/a.mp4")
        persistence.add_recent_file("/path/b.mp4")
        recent = persistence.load_recent_files()
        assert recent[0] == "/path/b.mp4"  # Most recent first
        assert recent[1] == "/path/a.mp4"

    def test_add_recent_file_deduplicates(self, persistence):
        persistence.add_recent_file("/path/a.mp4")
        persistence.add_recent_file("/path/b.mp4")
        persistence.add_recent_file("/path/a.mp4")  # Re-add
        recent = persistence.load_recent_files()
        assert recent[0] == "/path/a.mp4"
        assert len(recent) == 2

    def test_save_and_load_team_preset(self, persistence):
        config = {"primary_color": [255, 0, 0], "name": "Red Team"}
        assert persistence.save_team_preset("reds", config) is True
        loaded = persistence.get_team_preset("reds")
        assert loaded["name"] == "Red Team"

    def test_get_nonexistent_preset(self, persistence):
        assert persistence.get_team_preset("nonexistent") is None

    def test_load_team_presets_empty(self, persistence):
        presets = persistence.load_team_presets()
        assert presets == {}

    def test_delete_team_preset(self, persistence):
        config = {"name": "Blues"}
        persistence.save_team_preset("blues", config)
        assert persistence.delete_team_preset("blues") is True
        assert persistence.get_team_preset("blues") is None

    def test_delete_nonexistent_preset(self, persistence):
        assert persistence.delete_team_preset("nonexistent") is False

    def test_get_config_persistence(self):
        p = get_config_persistence()
        assert isinstance(p, ConfigPersistence)
