"""
Soccer Film Analysis - Jersey Number OCR
Detects and reads jersey numbers from player detections
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# Try to import EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")


@dataclass
class JerseyNumberResult:
    """Result of jersey number detection"""
    number: Optional[int] = None
    confidence: float = 0.0
    raw_text: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2 within player crop


@dataclass
class PlayerRoster:
    """Player roster information"""
    number: int
    name: str
    position: str = ""
    team_id: int = 0  # 0=home, 1=away


class JerseyNumberReader:
    """
    Reads jersey numbers from player detection crops using OCR.
    """

    def __init__(self, languages: List[str] = None):
        """
        Initialize the jersey number reader.

        Args:
            languages: OCR languages (default: ['en'])
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not installed. Run: pip install easyocr")

        self.languages = languages or ['en']
        self._reader = None

        # Number recognition settings
        self.min_confidence = 0.3
        self.valid_numbers = set(range(1, 100))  # Valid jersey numbers 1-99

        # Image preprocessing settings
        self.target_height = 100  # Resize jersey crop to this height

        # Roster mapping
        self.home_roster: Dict[int, PlayerRoster] = {}
        self.away_roster: Dict[int, PlayerRoster] = {}

        # Cache for detected numbers (tracker_id -> number)
        self._number_cache: Dict[int, int] = {}
        self._number_confidence: Dict[int, float] = {}

        logger.info("JerseyNumberReader initialized")

    @property
    def reader(self):
        """Lazy load EasyOCR reader"""
        if self._reader is None:
            logger.info("Loading EasyOCR model (this may take a moment)...")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=True,  # Use GPU if available
                verbose=False
            )
            logger.info("EasyOCR model loaded")
        return self._reader

    def load_roster(self, roster_data: List[Dict], team_id: int):
        """
        Load a team roster for player identification.

        Args:
            roster_data: List of dicts with 'number', 'name', 'position'
            team_id: 0 for home, 1 for away
        """
        roster_dict = self.home_roster if team_id == 0 else self.away_roster
        roster_dict.clear()

        for player in roster_data:
            number = player.get('number')
            if number is not None:
                roster_dict[number] = PlayerRoster(
                    number=number,
                    name=player.get('name', ''),
                    position=player.get('position', ''),
                    team_id=team_id
                )

        logger.info(f"Loaded roster with {len(roster_dict)} players for team {team_id}")

    def load_roster_from_csv(self, csv_path: str, team_id: int):
        """
        Load roster from CSV file.

        Expected columns: number, name, position (position optional)
        """
        import csv

        roster_data = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try different column name variations
                number = None
                for col in ['number', 'Number', '#', 'Jersey', 'jersey']:
                    if col in row and row[col]:
                        try:
                            number = int(row[col])
                            break
                        except ValueError:
                            continue

                name = row.get('name') or row.get('Name') or row.get('Player') or ''
                position = row.get('position') or row.get('Position') or row.get('Pos') or ''

                if number is not None:
                    roster_data.append({
                        'number': number,
                        'name': name,
                        'position': position
                    })

        self.load_roster(roster_data, team_id)

    def read_jersey_number(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        tracker_id: Optional[int] = None
    ) -> JerseyNumberResult:
        """
        Read jersey number from a player detection.

        Args:
            frame: Full video frame (BGR)
            bbox: Player bounding box (x1, y1, x2, y2)
            tracker_id: Optional tracker ID for caching

        Returns:
            JerseyNumberResult with detected number
        """
        # Check cache first
        if tracker_id is not None and tracker_id in self._number_cache:
            cached_conf = self._number_confidence.get(tracker_id, 0)
            if cached_conf > 0.7:  # High confidence cached result
                return JerseyNumberResult(
                    number=self._number_cache[tracker_id],
                    confidence=cached_conf,
                    raw_text=str(self._number_cache[tracker_id])
                )

        # Extract jersey region (upper body)
        jersey_crop = self._extract_jersey_region(frame, bbox)
        if jersey_crop is None or jersey_crop.size == 0:
            return JerseyNumberResult()

        # Preprocess for OCR
        processed = self._preprocess_jersey_image(jersey_crop)

        # Run OCR
        try:
            results = self.reader.readtext(
                processed,
                allowlist='0123456789',
                paragraph=False,
                min_size=10
            )
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return JerseyNumberResult()

        # Parse results
        best_result = self._parse_ocr_results(results)

        # Update cache if we have a tracker ID
        if tracker_id is not None and best_result.number is not None:
            # Update cache with weighted average of confidence
            if tracker_id in self._number_cache:
                old_num = self._number_cache[tracker_id]
                old_conf = self._number_confidence[tracker_id]

                if best_result.number == old_num:
                    # Same number - increase confidence
                    new_conf = min(0.99, old_conf + best_result.confidence * 0.1)
                    self._number_confidence[tracker_id] = new_conf
                elif best_result.confidence > old_conf:
                    # Different number with higher confidence - replace
                    self._number_cache[tracker_id] = best_result.number
                    self._number_confidence[tracker_id] = best_result.confidence
            else:
                self._number_cache[tracker_id] = best_result.number
                self._number_confidence[tracker_id] = best_result.confidence

        return best_result

    def _extract_jersey_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        """Extract the jersey (upper body) region from player bbox"""
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        width = x2 - x1

        # Jersey is typically in the upper 40-70% of the body
        jersey_y1 = y1 + int(height * 0.15)  # Skip head
        jersey_y2 = y1 + int(height * 0.55)  # Upper body only

        # Add some horizontal margin
        margin = int(width * 0.1)
        jersey_x1 = max(0, x1 + margin)
        jersey_x2 = min(frame.shape[1], x2 - margin)

        # Validate coordinates
        jersey_y1 = max(0, min(jersey_y1, frame.shape[0] - 1))
        jersey_y2 = max(0, min(jersey_y2, frame.shape[0] - 1))

        if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
            return None

        return frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2].copy()

    def _preprocess_jersey_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess jersey crop for better OCR"""
        # Resize to consistent height
        h, w = image.shape[:2]
        if h < 20:
            return image

        scale = self.target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(image, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        # Adaptive threshold for better number visibility
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        return binary

    def _parse_ocr_results(self, results: List) -> JerseyNumberResult:
        """Parse EasyOCR results and find the best jersey number"""
        best_number = None
        best_confidence = 0.0
        best_text = ""
        best_bbox = None

        for bbox, text, confidence in results:
            # Clean the text
            clean_text = ''.join(c for c in text if c.isdigit())

            if not clean_text:
                continue

            # Try to parse as number
            try:
                number = int(clean_text)
            except ValueError:
                continue

            # Check if it's a valid jersey number
            if number not in self.valid_numbers:
                # Try to fix common OCR errors
                if number > 99 and len(clean_text) > 2:
                    # Might have detected extra digits - try first 2
                    try:
                        number = int(clean_text[:2])
                    except ValueError:
                        continue

            if number in self.valid_numbers and confidence > best_confidence:
                best_number = number
                best_confidence = confidence
                best_text = text
                # Convert bbox format
                if bbox:
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    best_bbox = (
                        int(min(x_coords)), int(min(y_coords)),
                        int(max(x_coords)), int(max(y_coords))
                    )

        return JerseyNumberResult(
            number=best_number,
            confidence=best_confidence,
            raw_text=best_text,
            bbox=best_bbox
        )

    def get_player_name(self, number: int, team_id: int) -> Optional[str]:
        """Look up player name from roster"""
        roster = self.home_roster if team_id == 0 else self.away_roster
        player = roster.get(number)
        return player.name if player else None

    def get_cached_number(self, tracker_id: int) -> Optional[int]:
        """Get cached jersey number for a tracker ID"""
        return self._number_cache.get(tracker_id)

    def clear_cache(self):
        """Clear the number cache"""
        self._number_cache.clear()
        self._number_confidence.clear()

    def get_detection_stats(self) -> Dict:
        """Get statistics about number detection"""
        return {
            "total_players_identified": len(self._number_cache),
            "numbers_detected": list(self._number_cache.values()),
            "average_confidence": sum(self._number_confidence.values()) / max(1, len(self._number_confidence))
        }


class BatchJerseyReader:
    """
    Efficiently reads jersey numbers from multiple players in batches.
    """

    def __init__(self, reader: Optional[JerseyNumberReader] = None):
        self.reader = reader or JerseyNumberReader()
        self._batch_results: Dict[int, JerseyNumberResult] = {}

    def process_frame(
        self,
        frame: np.ndarray,
        player_detections: List,
        sample_rate: int = 5
    ) -> Dict[int, JerseyNumberResult]:
        """
        Process multiple players in a frame.

        Args:
            frame: Video frame
            player_detections: List of PlayerDetection objects
            sample_rate: Only process every Nth detection (for performance)

        Returns:
            Dict mapping tracker_id to JerseyNumberResult
        """
        results = {}

        for i, player in enumerate(player_detections):
            # Skip some detections for performance
            if i % sample_rate != 0:
                # Use cached value if available
                if player.tracker_id and player.tracker_id in self._batch_results:
                    results[player.tracker_id] = self._batch_results[player.tracker_id]
                continue

            if player.tracker_id is None:
                continue

            result = self.reader.read_jersey_number(
                frame,
                player.bbox,
                player.tracker_id
            )

            if result.number is not None:
                results[player.tracker_id] = result
                self._batch_results[player.tracker_id] = result

        return results
