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
    SoccerDetector, TeamClassifier, PitchTransformer,
    FrameDetections, PlayerDetection, draw_detections
)
from src.database.models import (
    Game, Team, Player, TrackingData, Event, PlayerMetrics, 
    TeamMetrics, AnalysisSession, TeamType, AnalysisStatus,
    get_db_session
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
        pitch_transformer: Optional[PitchTransformer] = None
    ):
        """
        Initialize the video processor.
        
        Args:
            detector: SoccerDetector instance (created if not provided)
            team_classifier: TeamClassifier instance
            pitch_transformer: PitchTransformer instance
        """
        self.detector = detector or SoccerDetector()
        self.team_classifier = team_classifier or TeamClassifier()
        self.pitch_transformer = pitch_transformer or PitchTransformer()
        
        # Video info
        self.video_info: Optional[VideoInfo] = None
        self._cap: Optional[cv2.VideoCapture] = None
        
        # Analysis state
        self._is_processing = False
        self._should_stop = False
        self._current_frame = 0
        
        # Results storage
        self.frame_detections: Dict[int, FrameDetections] = {}
        
        # Callbacks
        self._progress_callback: Optional[Callable[[AnalysisProgress], None]] = None
        self._frame_callback: Optional[Callable[[np.ndarray, FrameDetections], None]] = None
        
        logger.info("VideoProcessor initialized")
    
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
        game_id: Optional[int] = None
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
        
        logger.info(
            f"Starting {depth.value} analysis | "
            f"Frames {start_frame}-{end_frame} | "
            f"Sample rate: 1/{frame_sample_rate}"
        )
        
        start_time = datetime.now()
        processed_frames = 0
        total_frames_to_process = (end_frame - start_frame) // frame_sample_rate
        
        # Create database records if saving
        session_id = None
        if save_to_db:
            session_id = self._create_analysis_session(depth, start_frame, end_frame, game_id)
        
        try:
            # Position video at start frame
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._current_frame = start_frame
            
            # Main processing loop
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
                
                # Run detection
                detections = self.detector.detect_frame(
                    frame, frame_num, self.video_info.fps,
                    detect_pitch=True, track_objects=True
                )
                
                # Classify teams
                self.team_classifier.classify_players(detections.players)
                self.team_classifier.classify_players(detections.goalkeepers)
                
                # Store detections
                self.frame_detections[frame_num] = detections
                
                # Save to database
                if save_to_db and session_id:
                    self._save_frame_data(session_id, detections)
                
                # Callbacks
                if self._frame_callback:
                    annotated_frame = draw_detections(frame, detections)
                    self._frame_callback(annotated_frame, detections)
                
                processed_frames += 1
                self._current_frame = frame_num
                
                # Progress update
                if self._progress_callback:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    fps = processed_frames / max(elapsed, 0.001)
                    remaining_frames = total_frames_to_process - processed_frames
                    eta = remaining_frames / max(fps, 0.001)
                    
                    progress = AnalysisProgress(
                        current_frame=frame_num,
                        total_frames=end_frame,
                        percentage=(frame_num - start_frame) / (end_frame - start_frame) * 100,
                        elapsed_seconds=elapsed,
                        estimated_remaining_seconds=eta,
                        frames_per_second=fps,
                        status="processing",
                        current_phase="detection"
                    )
                    self._progress_callback(progress)
            
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
            
            return {
                "status": "completed",
                "processed_frames": processed_frames,
                "elapsed_seconds": elapsed,
                "session_id": session_id
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
