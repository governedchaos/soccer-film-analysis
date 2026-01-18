"""
Video Caching & Preprocessing Pipeline
Background worker for pre-processing videos to speed up analysis
"""

import cv2
import numpy as np
import hashlib
import json
import pickle
import threading
import queue
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CacheStatus(Enum):
    """Status of cached video data."""
    NOT_CACHED = "not_cached"
    PROCESSING = "processing"
    CACHED = "cached"
    FAILED = "failed"
    OUTDATED = "outdated"


@dataclass
class VideoMetadata:
    """Metadata for a cached video."""
    video_path: str
    video_hash: str
    duration_seconds: float
    frame_count: int
    fps: float
    width: int
    height: int
    file_size: int
    created_at: str
    processed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoMetadata':
        return cls(**data)


@dataclass
class FrameCache:
    """Cached frame data."""
    frame_number: int
    timestamp: float
    thumbnail: Optional[np.ndarray] = None
    detections: Optional[List[Dict[str, Any]]] = None
    features: Optional[Dict[str, Any]] = None


@dataclass
class PreprocessingTask:
    """Task for the preprocessing queue."""
    video_path: str
    priority: int = 0
    extract_thumbnails: bool = True
    thumbnail_interval: int = 30  # Every N frames
    run_detection: bool = False
    detection_model: Optional[Any] = None
    callback: Optional[Callable[[str, float], None]] = None


@dataclass
class CacheEntry:
    """Complete cache entry for a video."""
    metadata: VideoMetadata
    status: CacheStatus
    thumbnails: Dict[int, str] = field(default_factory=dict)  # frame_num -> path
    detections: Dict[int, List[Dict]] = field(default_factory=dict)
    keyframes: List[int] = field(default_factory=list)
    scene_changes: List[int] = field(default_factory=list)
    progress: float = 0.0
    error_message: Optional[str] = None


class VideoCacheManager:
    """
    Manages video caching and preprocessing.

    Features:
    - Background preprocessing of videos
    - Thumbnail extraction at configurable intervals
    - Detection result caching
    - Scene change detection
    - Keyframe extraction
    """

    def __init__(
        self,
        cache_dir: str = "video_cache",
        max_workers: int = 2,
        thumbnail_size: Tuple[int, int] = (320, 180),
        max_cache_size_gb: float = 10.0
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.thumbnail_dir = self.cache_dir / "thumbnails"
        self.thumbnail_dir.mkdir(exist_ok=True)

        self.detection_dir = self.cache_dir / "detections"
        self.detection_dir.mkdir(exist_ok=True)

        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        self.thumbnail_size = thumbnail_size
        self.max_cache_size_gb = max_cache_size_gb
        self.max_workers = max_workers

        # Thread pool for background processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Future] = {}

        # Processing queue with priority
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()

        # Cache index
        self.cache_index: Dict[str, CacheEntry] = {}
        self._load_cache_index()

        # Lock for thread safety
        self._lock = threading.Lock()

        # Start background worker
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def _get_video_hash(self, video_path: str) -> str:
        """Generate hash for video file based on path, size, and modification time."""
        path = Path(video_path)
        stat = path.stat()
        hash_input = f"{video_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _load_cache_index(self):
        """Load cache index from disk."""
        index_path = self.cache_dir / "cache_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    for video_hash, entry_data in data.items():
                        metadata = VideoMetadata.from_dict(entry_data['metadata'])
                        self.cache_index[video_hash] = CacheEntry(
                            metadata=metadata,
                            status=CacheStatus(entry_data['status']),
                            thumbnails=entry_data.get('thumbnails', {}),
                            detections={int(k): v for k, v in entry_data.get('detections', {}).items()},
                            keyframes=entry_data.get('keyframes', []),
                            scene_changes=entry_data.get('scene_changes', []),
                            progress=entry_data.get('progress', 0.0)
                        )
            except Exception as e:
                logger.error(f"Failed to load cache index: {e}")

    def _save_cache_index(self):
        """Save cache index to disk."""
        index_path = self.cache_dir / "cache_index.json"
        try:
            data = {}
            for video_hash, entry in self.cache_index.items():
                data[video_hash] = {
                    'metadata': entry.metadata.to_dict(),
                    'status': entry.status.value,
                    'thumbnails': entry.thumbnails,
                    'detections': {str(k): v for k, v in entry.detections.items()},
                    'keyframes': entry.keyframes,
                    'scene_changes': entry.scene_changes,
                    'progress': entry.progress
                }
            with open(index_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def get_cache_status(self, video_path: str) -> CacheStatus:
        """Get caching status for a video."""
        video_hash = self._get_video_hash(video_path)

        with self._lock:
            if video_hash not in self.cache_index:
                return CacheStatus.NOT_CACHED

            entry = self.cache_index[video_hash]

            # Check if video has changed
            current_hash = self._get_video_hash(video_path)
            if current_hash != video_hash:
                return CacheStatus.OUTDATED

            return entry.status

    def get_cache_progress(self, video_path: str) -> float:
        """Get preprocessing progress (0.0 to 1.0)."""
        video_hash = self._get_video_hash(video_path)

        with self._lock:
            if video_hash in self.cache_index:
                return self.cache_index[video_hash].progress
            return 0.0

    def queue_preprocessing(
        self,
        video_path: str,
        priority: int = 0,
        extract_thumbnails: bool = True,
        thumbnail_interval: int = 30,
        run_detection: bool = False,
        detection_model: Optional[Any] = None,
        callback: Optional[Callable[[str, float], None]] = None
    ) -> bool:
        """
        Queue a video for background preprocessing.

        Args:
            video_path: Path to video file
            priority: Higher priority = processed first (default 0)
            extract_thumbnails: Whether to extract thumbnails
            thumbnail_interval: Extract thumbnail every N frames
            run_detection: Whether to run object detection
            detection_model: YOLO model for detection
            callback: Progress callback(video_path, progress)

        Returns:
            True if queued successfully
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        task = PreprocessingTask(
            video_path=video_path,
            priority=-priority,  # Negative for priority queue (lower = higher priority)
            extract_thumbnails=extract_thumbnails,
            thumbnail_interval=thumbnail_interval,
            run_detection=run_detection,
            detection_model=detection_model,
            callback=callback
        )

        self.task_queue.put((task.priority, task))
        return True

    def _process_queue(self):
        """Background worker to process preprocessing tasks."""
        while self._running:
            try:
                priority, task = self.task_queue.get(timeout=1.0)
                self._preprocess_video(task)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in preprocessing worker: {e}")

    def _preprocess_video(self, task: PreprocessingTask):
        """Preprocess a single video."""
        video_path = task.video_path
        video_hash = self._get_video_hash(video_path)

        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            file_size = Path(video_path).stat().st_size

            # Create metadata
            metadata = VideoMetadata(
                video_path=video_path,
                video_hash=video_hash,
                duration_seconds=duration,
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
                file_size=file_size,
                created_at=datetime.now().isoformat()
            )

            # Initialize cache entry
            with self._lock:
                self.cache_index[video_hash] = CacheEntry(
                    metadata=metadata,
                    status=CacheStatus.PROCESSING,
                    progress=0.0
                )

            thumbnails: Dict[int, str] = {}
            detections: Dict[int, List[Dict]] = {}
            keyframes: List[int] = []
            scene_changes: List[int] = []

            prev_frame = None
            frame_num = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Update progress
                progress = frame_num / max(frame_count, 1)
                with self._lock:
                    self.cache_index[video_hash].progress = progress

                if task.callback:
                    task.callback(video_path, progress)

                # Extract thumbnails at intervals
                if task.extract_thumbnails and frame_num % task.thumbnail_interval == 0:
                    thumbnail = cv2.resize(frame, self.thumbnail_size)
                    thumb_path = self.thumbnail_dir / f"{video_hash}_{frame_num}.jpg"
                    cv2.imwrite(str(thumb_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    thumbnails[frame_num] = str(thumb_path)

                # Detect scene changes
                if prev_frame is not None:
                    diff = cv2.absdiff(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    )
                    mean_diff = np.mean(diff)
                    if mean_diff > 30:  # Scene change threshold
                        scene_changes.append(frame_num)

                # Run detection if requested
                if task.run_detection and task.detection_model is not None:
                    if frame_num % task.thumbnail_interval == 0:  # Same interval as thumbnails
                        results = task.detection_model(frame)
                        frame_detections = []
                        for r in results:
                            for box in r.boxes:
                                frame_detections.append({
                                    'class': int(box.cls[0]),
                                    'confidence': float(box.conf[0]),
                                    'bbox': box.xyxy[0].tolist()
                                })
                        detections[frame_num] = frame_detections

                prev_frame = frame.copy()
                frame_num += 1

            cap.release()

            # Identify keyframes (scene changes + regular intervals)
            keyframes = sorted(set(scene_changes + list(thumbnails.keys())))

            # Update cache entry
            with self._lock:
                entry = self.cache_index[video_hash]
                entry.status = CacheStatus.CACHED
                entry.thumbnails = thumbnails
                entry.detections = detections
                entry.keyframes = keyframes
                entry.scene_changes = scene_changes
                entry.progress = 1.0
                entry.metadata.processed_at = datetime.now().isoformat()
                self._save_cache_index()

            # Save detections to disk
            if detections:
                detection_path = self.detection_dir / f"{video_hash}_detections.pkl"
                with open(detection_path, 'wb') as f:
                    pickle.dump(detections, f)

            logger.info(f"Preprocessing complete: {video_path}")

            if task.callback:
                task.callback(video_path, 1.0)

        except Exception as e:
            logger.error(f"Preprocessing failed for {video_path}: {e}")
            with self._lock:
                if video_hash in self.cache_index:
                    self.cache_index[video_hash].status = CacheStatus.FAILED
                    self.cache_index[video_hash].error_message = str(e)
                    self._save_cache_index()

    def get_thumbnail(self, video_path: str, frame_number: int) -> Optional[np.ndarray]:
        """Get cached thumbnail for a frame."""
        video_hash = self._get_video_hash(video_path)

        with self._lock:
            if video_hash not in self.cache_index:
                return None

            entry = self.cache_index[video_hash]

            # Find nearest thumbnail
            available_frames = sorted(entry.thumbnails.keys())
            if not available_frames:
                return None

            # Binary search for nearest frame
            nearest = min(available_frames, key=lambda x: abs(x - frame_number))
            thumb_path = entry.thumbnails.get(nearest)

            if thumb_path and Path(thumb_path).exists():
                return cv2.imread(thumb_path)

        return None

    def get_cached_detections(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Dict[int, List[Dict]]:
        """Get cached detection results for frame range."""
        video_hash = self._get_video_hash(video_path)

        with self._lock:
            if video_hash not in self.cache_index:
                return {}

            entry = self.cache_index[video_hash]

            if not entry.detections:
                # Try loading from disk
                detection_path = self.detection_dir / f"{video_hash}_detections.pkl"
                if detection_path.exists():
                    with open(detection_path, 'rb') as f:
                        entry.detections = pickle.load(f)

            # Filter by frame range
            result = {}
            for frame_num, dets in entry.detections.items():
                if frame_num >= start_frame:
                    if end_frame is None or frame_num <= end_frame:
                        result[frame_num] = dets

            return result

    def get_scene_changes(self, video_path: str) -> List[int]:
        """Get detected scene change frames."""
        video_hash = self._get_video_hash(video_path)

        with self._lock:
            if video_hash in self.cache_index:
                return self.cache_index[video_hash].scene_changes.copy()
        return []

    def get_keyframes(self, video_path: str) -> List[int]:
        """Get keyframe numbers."""
        video_hash = self._get_video_hash(video_path)

        with self._lock:
            if video_hash in self.cache_index:
                return self.cache_index[video_hash].keyframes.copy()
        return []

    def get_video_metadata(self, video_path: str) -> Optional[VideoMetadata]:
        """Get cached video metadata."""
        video_hash = self._get_video_hash(video_path)

        with self._lock:
            if video_hash in self.cache_index:
                return self.cache_index[video_hash].metadata
        return None

    def clear_cache(self, video_path: Optional[str] = None):
        """Clear cache for a specific video or all videos."""
        with self._lock:
            if video_path:
                video_hash = self._get_video_hash(video_path)
                if video_hash in self.cache_index:
                    entry = self.cache_index[video_hash]

                    # Delete thumbnail files
                    for thumb_path in entry.thumbnails.values():
                        try:
                            Path(thumb_path).unlink(missing_ok=True)
                        except Exception:
                            pass

                    # Delete detection file
                    detection_path = self.detection_dir / f"{video_hash}_detections.pkl"
                    detection_path.unlink(missing_ok=True)

                    del self.cache_index[video_hash]
            else:
                # Clear all
                import shutil
                shutil.rmtree(self.thumbnail_dir, ignore_errors=True)
                shutil.rmtree(self.detection_dir, ignore_errors=True)
                self.thumbnail_dir.mkdir(exist_ok=True)
                self.detection_dir.mkdir(exist_ok=True)
                self.cache_index.clear()

            self._save_cache_index()

    def get_cache_size(self) -> float:
        """Get total cache size in GB."""
        total_size = 0
        for path in self.cache_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 ** 3)

    def cleanup_old_cache(self, max_age_days: int = 30):
        """Remove cache entries older than specified days."""
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)

        with self._lock:
            to_remove = []
            for video_hash, entry in self.cache_index.items():
                processed_at = entry.metadata.processed_at
                if processed_at:
                    entry_time = datetime.fromisoformat(processed_at).timestamp()
                    if entry_time < cutoff:
                        to_remove.append(video_hash)

            for video_hash in to_remove:
                entry = self.cache_index[video_hash]
                for thumb_path in entry.thumbnails.values():
                    try:
                        Path(thumb_path).unlink(missing_ok=True)
                    except Exception:
                        pass

                detection_path = self.detection_dir / f"{video_hash}_detections.pkl"
                detection_path.unlink(missing_ok=True)

                del self.cache_index[video_hash]

            self._save_cache_index()

    def shutdown(self):
        """Shutdown the cache manager."""
        self._running = False
        self._worker_thread.join(timeout=5.0)
        self.executor.shutdown(wait=False)
        self._save_cache_index()


class FrameExtractor:
    """Utility for efficient frame extraction from cached videos."""

    def __init__(self, cache_manager: VideoCacheManager):
        self.cache_manager = cache_manager
        self._video_caps: Dict[str, cv2.VideoCapture] = {}

    def get_frame(self, video_path: str, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame, using cache when possible.
        Falls back to direct extraction if not cached.
        """
        # Try cached thumbnail first (faster but lower quality)
        thumbnail = self.cache_manager.get_thumbnail(video_path, frame_number)
        if thumbnail is not None:
            return thumbnail

        # Extract directly from video
        return self._extract_frame_direct(video_path, frame_number)

    def _extract_frame_direct(self, video_path: str, frame_number: int) -> Optional[np.ndarray]:
        """Extract frame directly from video file."""
        if video_path not in self._video_caps:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            self._video_caps[video_path] = cap

        cap = self._video_caps[video_path]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        return frame if ret else None

    def get_frame_range(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        step: int = 1
    ) -> List[Tuple[int, np.ndarray]]:
        """Get a range of frames."""
        frames = []

        if video_path not in self._video_caps:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames
            self._video_caps[video_path] = cap

        cap = self._video_caps[video_path]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_num in range(start_frame, end_frame + 1, step):
            if step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret, frame = cap.read()
            if ret:
                frames.append((frame_num, frame))

            if step == 1:
                # Sequential read is faster
                for _ in range(step - 1):
                    cap.read()

        return frames

    def close(self):
        """Close all video captures."""
        for cap in self._video_caps.values():
            cap.release()
        self._video_caps.clear()
