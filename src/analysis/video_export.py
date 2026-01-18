"""
Soccer Film Analysis - Video Export
Export video with detection overlays baked in
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Callable, Tuple
from datetime import datetime
from loguru import logger

from src.detection.detector import FrameDetections, draw_detections


class VideoExporter:
    """
    Exports analyzed video with detection overlays.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        from config import settings
        self.output_dir = output_dir or settings.get_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_video(
        self,
        input_video_path: Path,
        detections_by_frame: Dict[int, FrameDetections],
        output_filename: Optional[str] = None,
        codec: str = "mp4v",
        include_stats_overlay: bool = True,
        include_minimap: bool = False,
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Path:
        """
        Export video with detection overlays.

        Args:
            input_video_path: Path to original video
            detections_by_frame: Dict mapping frame numbers to detections
            output_filename: Output filename (auto-generated if None)
            codec: Video codec (mp4v, XVID, etc.)
            include_stats_overlay: Include stats overlay on video
            include_minimap: Include tactical minimap
            team_colors: Custom team colors {0: (B,G,R), 1: (B,G,R)}
            progress_callback: Called with progress percentage

        Returns:
            Path to exported video
        """
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = output_filename or f"analyzed_{input_video_path.stem}_{timestamp}.mp4"
        output_path = self.output_dir / output_filename

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        logger.info(f"Exporting video to: {output_path}")

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for this frame
            detections = detections_by_frame.get(frame_num)

            if detections:
                # Draw detection overlays
                frame = draw_detections(frame, detections, team_colors)

                if include_stats_overlay:
                    frame = self._draw_stats_overlay(frame, detections, frame_num, fps)

                if include_minimap:
                    frame = self._draw_minimap(frame, detections)

            out.write(frame)
            frame_num += 1

            if progress_callback and frame_num % 100 == 0:
                progress_callback(frame_num / total_frames * 100)

        cap.release()
        out.release()

        logger.info(f"Video exported: {output_path} ({frame_num} frames)")
        return output_path

    def _draw_stats_overlay(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        frame_num: int,
        fps: float
    ) -> np.ndarray:
        """Draw stats overlay on frame"""
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Stats text
        time_str = f"{int(frame_num/fps//60):02d}:{int(frame_num/fps%60):02d}"
        stats = [
            f"Time: {time_str}",
            f"Frame: {frame_num}",
            f"Players: {len(detections.players)}",
            f"Ball: {'Detected' if detections.ball else 'Not found'}",
            f"Refs: {len(detections.referees)}"
        ]

        y = 30
        for stat in stats:
            cv2.putText(frame, stat, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20

        return frame

    def _draw_minimap(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        minimap_size: Tuple[int, int] = (200, 130)
    ) -> np.ndarray:
        """Draw tactical minimap in corner"""
        h, w = frame.shape[:2]
        mw, mh = minimap_size

        # Create minimap background (green pitch)
        minimap = np.zeros((mh, mw, 3), dtype=np.uint8)
        minimap[:] = (34, 139, 34)  # Forest green

        # Draw pitch lines
        cv2.rectangle(minimap, (5, 5), (mw-5, mh-5), (255, 255, 255), 1)
        cv2.line(minimap, (mw//2, 5), (mw//2, mh-5), (255, 255, 255), 1)
        cv2.circle(minimap, (mw//2, mh//2), 15, (255, 255, 255), 1)

        # Draw player positions (scaled)
        for player in detections.players:
            px = int((player.center[0] / w) * mw)
            py = int((player.center[1] / h) * mh)
            color = (0, 255, 255) if player.team_id == 0 else (0, 0, 255)
            cv2.circle(minimap, (px, py), 3, color, -1)

        # Draw ball
        if detections.ball:
            bx = int((detections.ball.center[0] / w) * mw)
            by = int((detections.ball.center[1] / h) * mh)
            cv2.circle(minimap, (bx, by), 4, (255, 255, 255), -1)

        # Place minimap on frame
        x_offset = w - mw - 10
        y_offset = h - mh - 10
        frame[y_offset:y_offset+mh, x_offset:x_offset+mw] = minimap

        return frame

    def export_clips(
        self,
        input_video_path: Path,
        events: List[Dict],
        detections_by_frame: Dict[int, FrameDetections],
        clip_padding_seconds: float = 3.0,
        output_prefix: str = "clip"
    ) -> List[Path]:
        """
        Export individual clips for each event.

        Args:
            input_video_path: Path to original video
            events: List of events with 'frame' key
            detections_by_frame: Frame detections
            clip_padding_seconds: Seconds before/after event
            output_prefix: Prefix for clip filenames

        Returns:
            List of paths to exported clips
        """
        cap = cv2.VideoCapture(str(input_video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        padding_frames = int(clip_padding_seconds * fps)
        exported_clips = []

        for i, event in enumerate(events):
            event_frame = event.get('frame', 0)
            event_type = event.get('event_type', 'event')

            start_frame = max(0, event_frame - padding_frames)
            end_frame = min(total_frames, event_frame + padding_frames)

            # Output path
            clip_filename = f"{output_prefix}_{i+1:03d}_{event_type}.mp4"
            clip_path = self.output_dir / clip_filename

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                detections = detections_by_frame.get(frame_num)
                if detections:
                    frame = draw_detections(frame, detections)

                # Highlight event frame
                if frame_num == event_frame:
                    cv2.putText(frame, f">>> {event_type.upper()} <<<",
                               (width//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                               1.0, (0, 255, 255), 2)

                out.write(frame)

            out.release()
            exported_clips.append(clip_path)
            logger.info(f"Exported clip: {clip_path}")

        cap.release()
        return exported_clips
