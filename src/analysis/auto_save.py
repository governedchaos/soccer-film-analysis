"""
Soccer Film Analysis - Auto Save
Periodic saves to prevent data loss
"""

import json
import pickle
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from loguru import logger


class AutoSaveManager:
    """
    Manages automatic saving of analysis state.
    """

    def __init__(
        self,
        save_dir: Optional[Path] = None,
        interval_seconds: int = 60,
        max_saves: int = 5
    ):
        """
        Args:
            save_dir: Directory for auto-save files
            interval_seconds: Seconds between saves
            max_saves: Maximum number of auto-save files to keep
        """
        from config import settings
        self.save_dir = save_dir or (settings.get_output_dir() / "autosave")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval_seconds
        self.max_saves = max_saves

        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._get_state_callback: Optional[Callable[[], Dict]] = None
        self._last_save_time: Optional[datetime] = None

    def start(self, get_state_callback: Callable[[], Dict]):
        """
        Start auto-save timer.

        Args:
            get_state_callback: Function that returns current state to save
        """
        self._get_state_callback = get_state_callback
        self._running = True
        self._schedule_save()
        logger.info(f"Auto-save started (every {self.interval}s)")

    def stop(self):
        """Stop auto-save timer"""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
        logger.info("Auto-save stopped")

    def _schedule_save(self):
        """Schedule next save"""
        if not self._running:
            return

        self._timer = threading.Timer(self.interval, self._perform_save)
        self._timer.daemon = True
        self._timer.start()

    def _perform_save(self):
        """Perform the save operation"""
        try:
            if self._get_state_callback:
                state = self._get_state_callback()
                self.save_state(state)
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
        finally:
            self._schedule_save()

    def save_state(self, state: Dict[str, Any]) -> Path:
        """
        Save current state to file.

        Args:
            state: State dictionary to save

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"autosave_{timestamp}.json"
        filepath = self.save_dir / filename

        # Add metadata
        state['_autosave_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }

        # Save as JSON for readability
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            filepath = filepath.with_suffix('.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

        self._last_save_time = datetime.now()
        logger.debug(f"Auto-saved to: {filepath}")

        # Cleanup old saves
        self._cleanup_old_saves()

        return filepath

    def _cleanup_old_saves(self):
        """Remove old auto-save files beyond max_saves"""
        saves = sorted(self.save_dir.glob("autosave_*"), reverse=True)

        for old_save in saves[self.max_saves:]:
            try:
                old_save.unlink()
                logger.debug(f"Removed old autosave: {old_save}")
            except Exception as e:
                logger.warning(f"Failed to remove old autosave: {e}")

    def load_latest(self) -> Optional[Dict]:
        """
        Load the most recent auto-save.

        Returns:
            State dictionary or None if no saves exist
        """
        saves = sorted(self.save_dir.glob("autosave_*"), reverse=True)

        if not saves:
            return None

        latest = saves[0]
        logger.info(f"Loading auto-save: {latest}")

        try:
            if latest.suffix == '.json':
                with open(latest, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(latest, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load auto-save: {e}")
            return None

    def get_available_saves(self) -> list:
        """Get list of available auto-saves"""
        saves = sorted(self.save_dir.glob("autosave_*"), reverse=True)
        return [
            {
                'path': str(s),
                'filename': s.name,
                'modified': datetime.fromtimestamp(s.stat().st_mtime).isoformat()
            }
            for s in saves
        ]

    def get_last_save_time(self) -> Optional[datetime]:
        """Get time of last save"""
        return self._last_save_time


class AnalysisStateManager:
    """
    Manages saving and loading of complete analysis state.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        from config import settings
        self.output_dir = output_dir or settings.get_output_dir()

    def save_analysis(
        self,
        video_path: str,
        frame_detections: Dict,
        events: list,
        game_config: Dict,
        corrections: Dict,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save complete analysis state.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"analysis_save_{timestamp}.pkl"
        filepath = self.output_dir / filename

        state = {
            'version': '1.0',
            'saved_at': datetime.now().isoformat(),
            'video_path': video_path,
            'frame_count': len(frame_detections),
            'event_count': len(events),
            'game_config': game_config,
            'corrections': corrections,
            # Note: frame_detections might be large, consider compression
            'frame_detections': self._serialize_detections(frame_detections),
            'events': events
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Analysis saved to: {filepath}")
        return filepath

    def load_analysis(self, filepath: Path) -> Dict:
        """
        Load saved analysis state.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        logger.info(f"Analysis loaded from: {filepath}")
        return state

    def _serialize_detections(self, detections: Dict) -> Dict:
        """Serialize detections for saving"""
        # Convert to simple dicts for serialization
        serialized = {}
        for frame_num, det in detections.items():
            serialized[frame_num] = {
                'frame_number': det.frame_number,
                'timestamp_seconds': det.timestamp_seconds,
                'players': [
                    {
                        'bbox': p.bbox,
                        'confidence': p.confidence,
                        'team_id': p.team_id,
                        'tracker_id': p.tracker_id
                    }
                    for p in det.players
                ],
                'ball': {
                    'bbox': det.ball.bbox,
                    'confidence': det.ball.confidence
                } if det.ball else None,
                'referees': [
                    {'bbox': r.bbox, 'tracker_id': r.tracker_id}
                    for r in det.referees
                ]
            }
        return serialized
