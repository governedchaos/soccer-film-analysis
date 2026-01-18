"""
Soccer Film Analysis - Video Processor
Main orchestrator for video analysis pipeline
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Generator, Tuple, List, Dict, Any
from dataclasses import dataclass
import threading
import queue
from loguru import logger

from config import settings, AnalysisDepth
from src.detection.detector import (
    SoccerDetector, TeamClassifier, PitchTransformer, PossessionCalculator,
    FrameDetections, PlayerDetection, draw_detections
)
from src.detection.enhanced_detector import EnhancedDetector
from src.database.models import (
    Game, Team, Player, TrackingData, Event, PlayerMetrics,
    TeamMetrics, AnalysisSession, TeamType, AnalysisStatus,
    get_db_session
)
from src.analysis.pipeline import (
    AnalysisPipeline,
    AnalysisPipelineConfig,
    AnalysisDepthLevel,
    FrameAnalysisResult,
    MatchAnalysisSummary
)


@dataclass
class VideoInfo:
    """Video metadata"""
    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    
    @classmethod
    def from_capture(cls, path: Path, cap: cv2.VideoCapture) -> "VideoInfo":
        return cls(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration_seconds=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        )


class BatchFrameReader:
    """
    Efficiently reads frames in batches for GPU processing.

    Pre-loads a batch of frames from the video so they can be
    processed together in a single GPU inference call.
    """

    def __init__(
        self,
        cap: cv2.VideoCapture,
        batch_size: int = 8,
        frame_sample_rate: int = 1
    ):
        """
        Initialize the batch frame reader.

        Args:
            cap: OpenCV video capture object
            batch_size: Number of frames to load per batch
            frame_sample_rate: Only read every Nth frame
        """
        self.cap = cap
        self.batch_size = batch_size
        self.frame_sample_rate = frame_sample_rate

    def read_batch(
        self,
        start_frame: int,
        end_frame: int
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Read a batch of frames starting from start_frame.

        Args:
            start_frame: Frame number to start from
            end_frame: Maximum frame number (exclusive)

        Returns:
            Tuple of (frames list, frame_numbers list)
        """
        frames: List[np.ndarray] = []
        frame_numbers: List[int] = []

        current_frame = start_frame

        while len(frames) < self.batch_size and current_frame < end_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = self.cap.read()

            if ret:
                frames.append(frame)
                frame_numbers.append(current_frame)
            else:
                logger.warning(f"Failed to read frame {current_frame}")

            current_frame += self.frame_sample_rate

        return frames, frame_numbers

    def iterate_batches(
        self,
        start_frame: int,
        end_frame: int
    ) -> Generator[Tuple[List[np.ndarray], List[int]], None, None]:
        """
        Generator that yields batches of frames.

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (exclusive)

        Yields:
            Tuple of (frames list, frame_numbers list)
        """
        current_frame = start_frame

        while current_frame < end_frame:
            frames, frame_numbers = self.read_batch(current_frame, end_frame)

            if not frames:
                break

            yield frames, frame_numbers

            # Move to next batch
            if frame_numbers:
                current_frame = frame_numbers[-1] + self.frame_sample_rate


@dataclass
class AnalysisProgress:
    """Progress information for analysis"""
    current_frame: int
    total_frames: int
    percentage: float
    elapsed_seconds: float
    estimated_remaining_seconds: float
    frames_per_second: float
    status: str
    current_phase: str
    
    def __str__(self):
        return (
            f"{self.percentage:.1f}% complete | "
            f"Frame {self.current_frame}/{self.total_frames} | "
            f"{self.frames_per_second:.1f} FPS | "
            f"ETA: {self.estimated_remaining_seconds/60:.1f} min"
        )


class VideoProcessor:
    """
    Main video processing class that orchestrates the entire analysis pipeline.
    
    Workflow:
    1. Load video and extract metadata
    2. Optionally run calibration phase (team colors, pitch detection)
    3. Process frames with detection and tracking
    4. Analyze events and compute metrics
    5. Store results in database
    6. Generate visualizations and reports
    """
    
    def __init__(
        self,
        detector: Optional[SoccerDetector] = None,
        team_classifier: Optional[TeamClassifier] = None,
        pitch_transformer: Optional[PitchTransformer] = None,
        possession_calculator: Optional[PossessionCalculator] = None
    ):
        """
        Initialize the video processor.

        Args:
            detector: SoccerDetector instance (created if not provided)
            team_classifier: TeamClassifier instance
            pitch_transformer: PitchTransformer instance
            possession_calculator: PossessionCalculator instance
        """
        # Use EnhancedDetector by default for better referee/ball detection
        self.detector = detector or EnhancedDetector()
        self.team_classifier = team_classifier or TeamClassifier()
        self.pitch_transformer = pitch_transformer or PitchTransformer()
        self.possession_calculator = possession_calculator or PossessionCalculator()
        
        # Video info
        self.video_info: Optional[VideoInfo] = None
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Analysis state
        self._is_processing = False
        self._should_stop = False
        self._current_frame = 0
        
        # Results storage
        self.frame_detections: Dict[int, FrameDetections] = {}

        # Analysis pipeline (tactical analysis)
        self.analysis_pipeline: Optional[AnalysisPipeline] = None
        self.frame_analysis_results: Dict[int, FrameAnalysisResult] = {}

        # Detection statistics tracking
        self._detection_stats = {
            "frames_processed": 0,
            "frames_with_ball": 0,
            "frames_with_players": 0,
            "total_player_detections": 0,
            "total_referee_detections": 0,
            "home_team_detections": 0,
            "away_team_detections": 0,
            "ball_detection_rate": 0.0,
            "avg_players_per_frame": 0.0,
        }

        # Callbacks
        self._progress_callback: Optional[Callable[[AnalysisProgress], None]] = None
        self._frame_callback: Optional[Callable[[np.ndarray, FrameDetections], None]] = None

        # Log device info at startup
        device_info = settings.get_device_info()
        logger.info(f"VideoProcessor initialized")
        logger.info(f"Compute device: {device_info['device']} | PyTorch: {device_info['torch_version']}")
        if device_info['cuda_available']:
            logger.info(f"CUDA: {device_info['cuda_version']} | GPU: {device_info['cuda_device_name']}")
        elif device_info['mps_available']:
            logger.info("Apple MPS acceleration enabled")
        elif device_info['recommendation']:
            logger.warning(device_info['recommendation'])
    
    def load_video(self, video_path: str | Path) -> VideoInfo:
        """
        Load a video file and extract metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInfo with video metadata
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        
        # Release previous capture if any
        if self._cap is not None:
            self._cap.release()
        
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")
        
        self.video_info = VideoInfo.from_capture(path, self._cap)
        
        logger.info(
            f"Loaded video: {path.name} | "
            f"{self.video_info.width}x{self.video_info.height} | "
            f"{self.video_info.fps:.1f} FPS | "
            f"{self.video_info.total_frames} frames | "
            f"{self.video_info.duration_seconds/60:.1f} min"
        )
        
        # Reset state
        self.detector.reset_tracker()
        self.team_classifier.reset_cache()
        self.frame_detections.clear()
        self._current_frame = 0

        return self.video_info
    
    def calibrate_teams(
        self,
        sample_frames: int = 30,
        home_color: Optional[Tuple[int, int, int]] = None,
        away_color: Optional[Tuple[int, int, int]] = None
    ) -> bool:
        """
        Calibrate team classification.
        
        Either provide manual colors or let the system learn from sample frames.
        
        Args:
            sample_frames: Number of frames to sample for automatic calibration
            home_color: Manual RGB color for home team
            away_color: Manual RGB color for away team
            
        Returns:
            True if calibration successful
        """
        if self._cap is None:
            raise RuntimeError("No video loaded. Call load_video() first.")
        
        # Manual color override
        if home_color and away_color:
            self.team_classifier.set_team_colors(home_color, away_color)
            return True
        
        logger.info(f"Auto-calibrating team colors from {sample_frames} frames...")
        
        # Sample frames evenly across video
        frame_indices = np.linspace(0, self.video_info.total_frames - 1, sample_frames, dtype=int)
        all_colors = []
        
        for idx in frame_indices:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
            
            if not ret:
                continue
            
            # Detect without tracking for calibration
            detections = self.detector.detect_frame(
                frame, idx, self.video_info.fps,
                detect_pitch=False, track_objects=False
            )
            
            # Collect player colors (exclude goalkeepers and referees)
            for player in detections.players:
                if player.dominant_color is not None:
                    all_colors.append(player.dominant_color)
        
        if len(all_colors) < 10:
            logger.warning(f"Not enough player colors detected ({len(all_colors)})")
            return False
        
        # Fit team classifier
        self.team_classifier.fit(all_colors)
        
        # Reset video position
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return True
    
    def process_video(
        self,
        analysis_depth: Optional[AnalysisDepth] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        progress_callback: Optional[Callable[[AnalysisProgress], None]] = None,
        frame_callback: Optional[Callable[[np.ndarray, FrameDetections], None]] = None,
        save_to_db: bool = True,
        game_id: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process the loaded video through the analysis pipeline.

        Args:
            analysis_depth: Analysis depth level (quick/standard/deep)
            start_frame: Frame to start processing from
            end_frame: Frame to stop processing at (None = end of video)
            progress_callback: Function called with progress updates
            frame_callback: Function called with each processed frame and detections
            save_to_db: Whether to save results to database
            game_id: Existing game ID to update (creates new if None)
            batch_size: Frames per batch for GPU processing (auto-detected if None)

        Returns:
            Dict with analysis results summary
        """
        if self._cap is None:
            raise RuntimeError("No video loaded. Call load_video() first.")

        if self._is_processing:
            raise RuntimeError("Analysis already in progress")

        self._is_processing = True
        self._should_stop = False
        self._progress_callback = progress_callback
        self._frame_callback = frame_callback

        # Set parameters
        depth = analysis_depth or settings.default_analysis_depth
        frame_sample_rate = settings.get_frame_sample_rate(depth)
        end_frame = end_frame or self.video_info.total_frames

        # Determine if we should use batch processing (GPU only)
        device_info = settings.get_device_info()
        use_batch_processing = device_info['cuda_available'] or device_info['mps_available']

        # Set batch size based on available memory and device
        if batch_size is None:
            if device_info['cuda_available']:
                batch_size = 16  # Good default for most GPUs
            elif device_info['mps_available']:
                batch_size = 8   # Apple Silicon - more conservative
            else:
                batch_size = 1   # CPU - no benefit from batching

        logger.info(
            f"Starting {depth.value} analysis | "
            f"Frames {start_frame}-{end_frame} | "
            f"Sample rate: 1/{frame_sample_rate} | "
            f"Batch size: {batch_size if use_batch_processing else 'N/A (CPU)'}"
        )

        # Initialize analysis pipeline based on depth
        pipeline_depth = {
            AnalysisDepth.QUICK: AnalysisDepthLevel.MINIMAL,
            AnalysisDepth.STANDARD: AnalysisDepthLevel.STANDARD,
            AnalysisDepth.DEEP: AnalysisDepthLevel.COMPREHENSIVE,
        }.get(depth, AnalysisDepthLevel.STANDARD)

        pipeline_config = AnalysisPipelineConfig.from_depth(pipeline_depth)
        self.analysis_pipeline = AnalysisPipeline(pipeline_config)
        self.frame_analysis_results.clear()

        start_time = datetime.now()
        processed_frames = 0
        total_frames_to_process = (end_frame - start_frame) // frame_sample_rate

        # Create database records if saving
        session_id = None
        if save_to_db:
            session_id = self._create_analysis_session(depth, start_frame, end_frame, game_id)

        try:
            self._current_frame = start_frame

            if use_batch_processing and batch_size > 1:
                # GPU batch processing path
                processed_frames = self._process_video_batched(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_sample_rate=frame_sample_rate,
                    batch_size=batch_size,
                    start_time=start_time,
                    total_frames_to_process=total_frames_to_process,
                    save_to_db=save_to_db,
                    session_id=session_id
                )
            else:
                # CPU single-frame processing path
                processed_frames = self._process_video_sequential(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_sample_rate=frame_sample_rate,
                    start_time=start_time,
                    total_frames_to_process=total_frames_to_process,
                    save_to_db=save_to_db,
                    session_id=session_id
                )

            # Analysis complete
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Update session status
            if save_to_db and session_id:
                self._complete_analysis_session(session_id, elapsed)
            
            logger.info(
                f"Analysis complete | "
                f"{processed_frames} frames processed | "
                f"{elapsed/60:.1f} minutes"
            )

            # Log detection statistics
            self.log_detection_stats()

            # Get analysis pipeline summary
            analysis_summary = None
            if self.analysis_pipeline:
                analysis_summary = self.analysis_pipeline.get_summary()
                self._log_analysis_summary(analysis_summary)

            return {
                "status": "completed",
                "processed_frames": processed_frames,
                "elapsed_seconds": elapsed,
                "session_id": session_id,
                "detection_stats": self.get_detection_stats(),
                "analysis_summary": analysis_summary
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            if save_to_db and session_id:
                self._fail_analysis_session(session_id, str(e))
            raise
        
        finally:
            self._is_processing = False
    
    def stop_processing(self):
        """Stop the current analysis"""
        self._should_stop = True
        logger.info("Stop requested")

    def _process_video_sequential(
        self,
        start_frame: int,
        end_frame: int,
        frame_sample_rate: int,
        start_time: datetime,
        total_frames_to_process: int,
        save_to_db: bool,
        session_id: Optional[int]
    ) -> int:
        """
        Process video frames one at a time (CPU mode).

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            frame_sample_rate: Process every Nth frame
            start_time: When processing started
            total_frames_to_process: Total frames to process
            save_to_db: Whether to save to database
            session_id: Database session ID

        Returns:
            Number of frames processed
        """
        processed_frames = 0

        for frame_num in range(start_frame, end_frame, frame_sample_rate):
            if self._should_stop:
                logger.info("Analysis stopped by user")
                break

            # Read frame
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self._cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_num}")
                continue

            # Process single frame
            detections = self._process_single_frame(frame, frame_num)

            # Save to database
            if save_to_db and session_id:
                self._save_frame_data(session_id, detections)

            # Frame callback
            if self._frame_callback:
                annotated_frame = draw_detections(frame, detections)
                self._frame_callback(annotated_frame, detections)

            processed_frames += 1
            self._current_frame = frame_num

            # Progress update
            self._report_progress(
                frame_num, start_frame, end_frame,
                processed_frames, total_frames_to_process, start_time
            )

        return processed_frames

    def _process_video_batched(
        self,
        start_frame: int,
        end_frame: int,
        frame_sample_rate: int,
        batch_size: int,
        start_time: datetime,
        total_frames_to_process: int,
        save_to_db: bool,
        session_id: Optional[int]
    ) -> int:
        """
        Process video frames in batches for GPU efficiency.

        Uses batch YOLO inference for better GPU utilization.

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            frame_sample_rate: Process every Nth frame
            batch_size: Number of frames per batch
            start_time: When processing started
            total_frames_to_process: Total frames to process
            save_to_db: Whether to save to database
            session_id: Database session ID

        Returns:
            Number of frames processed
        """
        processed_frames = 0
        batch_reader = BatchFrameReader(self._cap, batch_size, frame_sample_rate)

        logger.info(f"Using GPU batch processing (batch_size={batch_size})")

        for frames, frame_numbers in batch_reader.iterate_batches(start_frame, end_frame):
            if self._should_stop:
                logger.info("Analysis stopped by user")
                break

            # Run batch detection
            try:
                detections_list = self.detector.detect_batch(
                    frames, frame_numbers, self.video_info.fps,
                    detect_pitch=False, track_objects=True
                )
            except Exception as e:
                logger.error(f"Batch detection failed: {e}")
                # Fallback to sequential processing for this batch
                detections_list = []
                for frame, frame_num in zip(frames, frame_numbers):
                    det = self.detector.detect_frame(
                        frame, frame_num, self.video_info.fps,
                        detect_pitch=False, track_objects=True
                    )
                    detections_list.append(det)

            # Process each frame's results (team classification, etc.)
            for frame, frame_num, detections in zip(frames, frame_numbers, detections_list):
                # Classify teams
                self.team_classifier.classify_players(detections.players)
                self.team_classifier.classify_players(detections.goalkeepers)

                # Calculate possession
                self.possession_calculator.calculate_possession(detections)

                # Update detection statistics
                self._update_detection_stats(detections)

                # Run tactical analysis pipeline
                if self.analysis_pipeline:
                    try:
                        analysis_result = self.analysis_pipeline.process_frame(
                            detections, fps=self.video_info.fps
                        )
                        self.frame_analysis_results[frame_num] = analysis_result
                    except Exception as analysis_error:
                        logger.debug(f"Analysis pipeline error on frame {frame_num}: {analysis_error}")

                # Store detections
                self.frame_detections[frame_num] = detections

                # Save to database
                if save_to_db and session_id:
                    self._save_frame_data(session_id, detections)

                # Frame callback
                if self._frame_callback:
                    annotated_frame = draw_detections(frame, detections)
                    self._frame_callback(annotated_frame, detections)

                processed_frames += 1
                self._current_frame = frame_num

            # Progress update (once per batch)
            if frame_numbers:
                self._report_progress(
                    frame_numbers[-1], start_frame, end_frame,
                    processed_frames, total_frames_to_process, start_time
                )

        return processed_frames

    def _process_single_frame(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> FrameDetections:
        """
        Process a single frame through detection and analysis.

        Args:
            frame: BGR image
            frame_num: Frame number

        Returns:
            FrameDetections for this frame
        """
        try:
            detections = self.detector.detect_frame(
                frame, frame_num, self.video_info.fps,
                detect_pitch=False, track_objects=True
            )

            # Classify teams
            self.team_classifier.classify_players(detections.players)
            self.team_classifier.classify_players(detections.goalkeepers)

            # Calculate possession
            self.possession_calculator.calculate_possession(detections)

            # Update detection statistics
            self._update_detection_stats(detections)

            # Run tactical analysis pipeline
            if self.analysis_pipeline:
                try:
                    analysis_result = self.analysis_pipeline.process_frame(
                        detections, fps=self.video_info.fps
                    )
                    self.frame_analysis_results[frame_num] = analysis_result
                except Exception as analysis_error:
                    logger.debug(f"Analysis pipeline error on frame {frame_num}: {analysis_error}")

            # Store detections
            self.frame_detections[frame_num] = detections
            return detections

        except Exception as detection_error:
            logger.warning(f"Detection failed on frame {frame_num}: {detection_error}")
            # Create empty detections to continue
            detections = FrameDetections(
                frame_number=frame_num,
                timestamp_seconds=frame_num / self.video_info.fps
            )
            self.frame_detections[frame_num] = detections
            return detections

    def _report_progress(
        self,
        frame_num: int,
        start_frame: int,
        end_frame: int,
        processed_frames: int,
        total_frames_to_process: int,
        start_time: datetime
    ):
        """Report progress to callback if registered."""
        if self._progress_callback:
            elapsed = (datetime.now() - start_time).total_seconds()
            fps = processed_frames / max(elapsed, 0.001)
            remaining_frames = total_frames_to_process - processed_frames
            eta = remaining_frames / max(fps, 0.001)

            progress = AnalysisProgress(
                current_frame=frame_num,
                total_frames=end_frame,
                percentage=(frame_num - start_frame) / max(end_frame - start_frame, 1) * 100,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=eta,
                frames_per_second=fps,
                status="processing",
                current_phase="detection"
            )
            self._progress_callback(progress)

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a specific frame from the video"""
        if self._cap is None:
            return None
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()
        return frame if ret else None
    
    def get_annotated_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a frame with detection annotations overlaid"""
        frame = self.get_frame(frame_number)
        if frame is None:
            return None

        detections = self.frame_detections.get(frame_number)
        if detections:
            return draw_detections(frame, detections)

        return frame

    def _update_detection_stats(self, detections: FrameDetections):
        """Update running detection statistics"""
        self._detection_stats["frames_processed"] += 1

        # Ball detection
        if detections.ball is not None:
            self._detection_stats["frames_with_ball"] += 1

        # Player detections
        num_players = len(detections.players)
        num_refs = len(detections.referees)
        num_gk = len(detections.goalkeepers)

        if num_players > 0 or num_gk > 0:
            self._detection_stats["frames_with_players"] += 1

        self._detection_stats["total_player_detections"] += num_players + num_gk
        self._detection_stats["total_referee_detections"] += num_refs

        # Team counts
        for player in detections.players + detections.goalkeepers:
            if player.team_id == 0:
                self._detection_stats["home_team_detections"] += 1
            elif player.team_id == 1:
                self._detection_stats["away_team_detections"] += 1

        # Update rates
        frames = self._detection_stats["frames_processed"]
        if frames > 0:
            self._detection_stats["ball_detection_rate"] = (
                self._detection_stats["frames_with_ball"] / frames * 100
            )
            self._detection_stats["avg_players_per_frame"] = (
                self._detection_stats["total_player_detections"] / frames
            )

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get current detection statistics including cache performance"""
        stats = self._detection_stats.copy()
        # Add cache stats
        stats["color_cache"] = self.detector.get_color_cache_stats()
        stats["team_cache"] = self.team_classifier.get_cache_stats()
        return stats

    def log_detection_stats(self):
        """Log detection statistics summary including cache performance"""
        stats = self._detection_stats
        logger.info("=" * 50)
        logger.info("DETECTION STATISTICS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Frames processed: {stats['frames_processed']}")
        logger.info(f"Ball detection rate: {stats['ball_detection_rate']:.1f}%")
        logger.info(f"Avg players per frame: {stats['avg_players_per_frame']:.1f}")
        logger.info(f"Total player detections: {stats['total_player_detections']}")
        logger.info(f"  - Home team: {stats['home_team_detections']}")
        logger.info(f"  - Away team: {stats['away_team_detections']}")
        logger.info(f"Referee detections: {stats['total_referee_detections']}")

        # Log cache performance stats
        logger.info("-" * 50)
        logger.info("CACHE PERFORMANCE")
        color_stats = self.detector.get_color_cache_stats()
        team_stats = self.team_classifier.get_cache_stats()
        logger.info(f"Color cache: {color_stats['cache_hits']} hits, {color_stats['cache_misses']} misses "
                   f"({color_stats['hit_rate_percent']:.1f}% hit rate)")
        logger.info(f"Team cache:  {team_stats['cache_hits']} hits, {team_stats['cache_misses']} misses "
                   f"({team_stats['hit_rate_percent']:.1f}% hit rate)")
        logger.info(f"Unique players tracked: {color_stats['cache_size']}")
        logger.info("=" * 50)

    def reset_detection_stats(self):
        """Reset detection statistics"""
        self._detection_stats = {
            "frames_processed": 0,
            "frames_with_ball": 0,
            "frames_with_players": 0,
            "total_player_detections": 0,
            "total_referee_detections": 0,
            "home_team_detections": 0,
            "away_team_detections": 0,
            "ball_detection_rate": 0.0,
            "avg_players_per_frame": 0.0,
        }

    def _log_analysis_summary(self, summary: MatchAnalysisSummary):
        """Log tactical analysis summary"""
        logger.info("=" * 50)
        logger.info("TACTICAL ANALYSIS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Possession: Home {summary.possession_home:.1f}% - Away {summary.possession_away:.1f}%")

        if summary.home_primary_formation:
            logger.info(f"Home formation: {summary.home_primary_formation.value} ({summary.formation_changes_home} changes)")
        if summary.away_primary_formation:
            logger.info(f"Away formation: {summary.away_primary_formation.value} ({summary.formation_changes_away} changes)")

        if summary.home_avg_compactness > 0 or summary.away_avg_compactness > 0:
            logger.info(f"Team compactness: Home {summary.home_avg_compactness:.2f} - Away {summary.away_avg_compactness:.2f}")
            logger.info(f"Team width: Home {summary.home_avg_width:.1f}m - Away {summary.away_avg_width:.1f}m")

        if summary.home_xg > 0 or summary.away_xg > 0:
            logger.info(f"xG: Home {summary.home_xg:.2f} ({summary.home_shots} shots) - "
                       f"Away {summary.away_xg:.2f} ({summary.away_shots} shots)")

        if summary.home_counter_attacks > 0 or summary.away_counter_attacks > 0:
            logger.info(f"Counter attacks: Home {summary.home_counter_attacks} - Away {summary.away_counter_attacks}")

        logger.info(f"Runs detected: {summary.total_runs_detected}")
        logger.info("=" * 50)

    def get_analysis_summary(self) -> Optional[MatchAnalysisSummary]:
        """Get the tactical analysis summary"""
        if self.analysis_pipeline:
            return self.analysis_pipeline.get_summary()
        return None

    def get_frame_analysis(self, frame_number: int) -> Optional[FrameAnalysisResult]:
        """Get tactical analysis for a specific frame"""
        return self.frame_analysis_results.get(frame_number)

    def iterate_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields (frame_number, frame) tuples.
        
        Args:
            start_frame: Starting frame
            end_frame: Ending frame (None = end of video)
            step: Frame step size
        """
        if self._cap is None:
            return
        
        end = end_frame or self.video_info.total_frames
        
        for frame_num in range(start_frame, end, step):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self._cap.read()
            if ret:
                yield frame_num, frame
    
    def _create_analysis_session(
        self,
        depth: AnalysisDepth,
        start_frame: int,
        end_frame: int,
        game_id: Optional[int] = None
    ) -> int:
        """Create analysis session in database"""
        with get_db_session() as session:
            # Create or get game record
            if game_id is None:
                game = Game(
                    video_path=str(self.video_info.path),
                    video_filename=self.video_info.path.name,
                    video_duration_seconds=self.video_info.duration_seconds,
                    video_fps=self.video_info.fps,
                    video_width=self.video_info.width,
                    video_height=self.video_info.height
                )
                session.add(game)
                session.flush()
                game_id = game.id
            
            # Create analysis session
            analysis = AnalysisSession(
                game_id=game_id,
                analysis_depth=depth.value,
                frame_sample_rate=settings.get_frame_sample_rate(depth),
                start_frame=start_frame,
                end_frame=end_frame,
                status=AnalysisStatus.PROCESSING,
                started_at=datetime.utcnow()
            )
            session.add(analysis)
            session.flush()
            
            return analysis.id
    
    def _save_frame_data(self, session_id: int, detections: FrameDetections):
        """Save frame detection data to database (batched for performance)"""
        # Implementation would batch writes for better performance
        # Simplified here for clarity
        pass
    
    def _complete_analysis_session(self, session_id: int, elapsed_seconds: float):
        """Mark analysis session as complete"""
        with get_db_session() as session:
            analysis = session.query(AnalysisSession).get(session_id)
            if analysis:
                analysis.status = AnalysisStatus.COMPLETED
                analysis.completed_at = datetime.utcnow()
                analysis.processing_time_seconds = elapsed_seconds
                analysis.progress_percentage = 100.0
    
    def _fail_analysis_session(self, session_id: int, error: str):
        """Mark analysis session as failed"""
        with get_db_session() as session:
            analysis = session.query(AnalysisSession).get(session_id)
            if analysis:
                analysis.status = AnalysisStatus.FAILED
                analysis.error_message = error
    
    def release(self):
        """Release video capture resources"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.debug("Video capture released")
    
    def __del__(self):
        self.release()


# ============================================
# Threaded Video Processor
# ============================================

class ThreadedVideoProcessor(VideoProcessor):
    """
    Video processor that runs analysis in a background thread.
    Useful for keeping GUI responsive during long analyses.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread: Optional[threading.Thread] = None
        self._result_queue: queue.Queue = queue.Queue()
    
    def process_video_async(
        self,
        analysis_depth: Optional[AnalysisDepth] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        progress_callback: Optional[Callable[[AnalysisProgress], None]] = None,
        frame_callback: Optional[Callable[[np.ndarray, FrameDetections], None]] = None,
        completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        save_to_db: bool = True,
        game_id: Optional[int] = None
    ):
        """
        Start video processing in a background thread.
        
        Same parameters as process_video(), plus:
        - completion_callback: Called when processing completes
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Processing already in progress")
        
        def run_analysis():
            try:
                result = self.process_video(
                    analysis_depth=analysis_depth,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    progress_callback=progress_callback,
                    frame_callback=frame_callback,
                    save_to_db=save_to_db,
                    game_id=game_id
                )
                self._result_queue.put(("success", result))
                if completion_callback:
                    completion_callback(result)
            except Exception as e:
                self._result_queue.put(("error", str(e)))
                if completion_callback:
                    completion_callback({"status": "failed", "error": str(e)})
        
        self._thread = threading.Thread(target=run_analysis, daemon=True)
        self._thread.start()
        logger.info("Background analysis started")
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for background processing to complete"""
        if self._thread is None:
            return {"status": "no_analysis"}
        
        self._thread.join(timeout)
        
        if not self._result_queue.empty():
            status, result = self._result_queue.get()
            if status == "error":
                raise RuntimeError(result)
            return result
        
        return {"status": "timeout"}
