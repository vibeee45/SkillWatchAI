import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import numpy as np

from camera.frame_output import FramePacket

logger = logging.getLogger("skillwatch.frame_pipeline")


@dataclass
class QualityResult:
    """Image-quality checks for one sampled frame."""
    valid: bool
    black: bool
    very_dark: bool
    blurred: bool
    frozen: bool
    mean_brightness: float
    blur_score: float
    difference_score: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "black": self.black,
            "very_dark": self.very_dark,
            "blurred": self.blurred,
            "frozen": self.frozen,
            "mean_brightness": round(self.mean_brightness, 2),
            "blur_score": round(self.blur_score, 2),
            "difference_score": round(self.difference_score, 4),
            "reasons": self.reasons,
        }


@dataclass
class AIFrame:
    """Standard frame object passed from Phase 3 into future AI modules."""
    frame: np.ndarray
    camera_id: Optional[str]
    room: Optional[str]
    source_type: Optional[str]
    frame_id: str
    timestamp: str
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int
    sample_index: int
    quality: QualityResult
    capture_fps: float
    processing_latency_ms: float

    def as_metadata(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "room": self.room,
            "source_type": self.source_type,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "inference_width": self.inference_width,
            "inference_height": self.inference_height,
            "sample_index": self.sample_index,
            "quality": self.quality.as_dict(),
            "capture_fps": round(self.capture_fps, 2),
            "processing_latency_ms": round(self.processing_latency_ms, 3),
        }


@dataclass
class PipelineStats:
    received_frames: int = 0
    sampled_frames: int = 0
    skipped_frames: int = 0
    processed_frames: int = 0
    last_frame_id: Optional[str] = None
    last_timestamp: Optional[str] = None
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    processing_fps: float = 0.0
    quality_failures: int = 0
    last_quality: Optional[dict[str, Any]] = None
    _latency_sum_ms: float = 0.0
    _window_start: float = field(default_factory=time.monotonic)
    _window_processed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "received_frames": self.received_frames,
            "sampled_frames": self.sampled_frames,
            "skipped_frames": self.skipped_frames,
            "processed_frames": self.processed_frames,
            "last_frame_id": self.last_frame_id,
            "last_timestamp": self.last_timestamp,
            "last_latency_ms": round(self.last_latency_ms, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "processing_fps": round(self.processing_fps, 2),
            "quality_failures": self.quality_failures,
            "last_quality": self.last_quality,
        }


class AIProcessor(ABC):
    """Common interface for all future AI modules in Phase 4+."""

    @abstractmethod
    def process(self, frame: AIFrame) -> dict[str, Any]:
        """Consume a standardized AIFrame and return structured results."""
        raise NotImplementedError


class NoOpAIProcessor(AIProcessor):
    """Safe placeholder until person/face/infrastructure models are added."""

    def process(self, frame: AIFrame) -> dict[str, Any]:
        return {
            "processor": "noop",
            "frame_id": frame.frame_id,
            "quality_valid": frame.quality.valid,
        }


class FramePipeline:
    """Phase 3 capture-to-AI preparation pipeline.

    Responsibilities:
    - frame sampling
    - resize for inference
    - timestamp/frame/camera metadata
    - image-quality checks
    - FPS and latency metrics
    - standardized AI input
    """

    def __init__(
        self,
        sample_every_n: int = 3,
        target_width: int = 640,
        target_height: int = 360,
        resize_mode: str = "fit",
        black_threshold: float = 5.0,
        dark_threshold: float = 35.0,
        blur_threshold: float = 50.0,
        freeze_difference_threshold: float = 0.5,
        freeze_after_samples: int = 2,
        ai_processor: Optional[AIProcessor] = None,
    ):
        if sample_every_n < 1:
            raise ValueError("sample_every_n must be >= 1")
        if target_width < 1 or target_height < 1:
            raise ValueError("target dimensions must be positive")

        self.sample_every_n = sample_every_n
        self.target_width = target_width
        self.target_height = target_height
        self.resize_mode = resize_mode
        self.black_threshold = black_threshold
        self.dark_threshold = dark_threshold
        self.blur_threshold = blur_threshold
        self.freeze_difference_threshold = freeze_difference_threshold
        self.freeze_after_samples = max(1, freeze_after_samples)
        self.ai_processor = ai_processor or NoOpAIProcessor()
        self.stats = PipelineStats()
        self.last_ai_result: Optional[dict[str, Any]] = None
        self._last_gray: Optional[np.ndarray] = None
        self._same_frame_count = 0

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_mode == "stretch":
            return cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)

        # Fit inside target dimensions while preserving aspect ratio, then pad.
        h, w = frame.shape[:2]
        scale = min(self.target_width / max(w, 1), self.target_height / max(h, 1))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        y = (self.target_height - nh) // 2
        x = (self.target_width - nw) // 2
        canvas[y:y + nh, x:x + nw] = resized
        return canvas

    def _quality(self, frame: np.ndarray) -> QualityResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mean_brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        black = mean_brightness <= self.black_threshold
        very_dark = mean_brightness <= self.dark_threshold
        blurred = blur_score < self.blur_threshold

        difference_score = 255.0
        if self._last_gray is not None and self._last_gray.shape == gray.shape:
            difference_score = float(np.mean(cv2.absdiff(gray, self._last_gray)))
        if difference_score <= self.freeze_difference_threshold:
            self._same_frame_count += 1
        else:
            self._same_frame_count = 0
        frozen = self._same_frame_count >= self.freeze_after_samples
        self._last_gray = gray.copy()

        reasons = []
        if black:
            reasons.append("black_frame")
        elif very_dark:
            reasons.append("very_dark_frame")
        if blurred:
            reasons.append("blurred_frame")
        if frozen:
            reasons.append("frozen_frame")

        return QualityResult(
            valid=not reasons,
            black=black,
            very_dark=very_dark,
            blurred=blurred,
            frozen=frozen,
            mean_brightness=mean_brightness,
            blur_score=blur_score,
            difference_score=difference_score,
            reasons=reasons,
        )

    def process(self, packet: FramePacket) -> Optional[AIFrame]:
        self.stats.received_frames += 1
        if packet.frame_number % self.sample_every_n != 0:
            self.stats.skipped_frames += 1
            return None

        self.stats.sampled_frames += 1
        started = time.perf_counter()
        # Run quality checks on the captured frame before padding/resizing so
        # aspect-ratio padding cannot distort the quality metrics.
        quality = self._quality(packet.frame)
        prepared = self._resize(packet.frame)
        timestamp = packet.timestamp or datetime.now(timezone.utc).isoformat()
        frame_id = f"{packet.camera_id or 'UNKNOWN'}-{packet.frame_number:08d}"
        latency_ms = (time.perf_counter() - started) * 1000.0

        ai_frame = AIFrame(
            frame=prepared,
            camera_id=packet.camera_id,
            room=packet.room,
            source_type=packet.source_type,
            frame_id=frame_id,
            timestamp=timestamp,
            original_width=packet.width,
            original_height=packet.height,
            inference_width=prepared.shape[1],
            inference_height=prepared.shape[0],
            sample_index=self.stats.sampled_frames,
            quality=quality,
            capture_fps=packet.fps,
            processing_latency_ms=latency_ms,
        )

        self.stats.processed_frames += 1
        self.stats.last_frame_id = frame_id
        self.stats.last_timestamp = timestamp
        self.stats.last_latency_ms = latency_ms
        self.stats._latency_sum_ms += latency_ms
        self.stats.avg_latency_ms = self.stats._latency_sum_ms / self.stats.processed_frames
        self.stats._window_processed += 1
        elapsed = time.monotonic() - self.stats._window_start
        if elapsed >= 1.0:
            self.stats.processing_fps = self.stats._window_processed / elapsed
            logger.info(
                "phase3 fps=%.2f processed=%d avg_latency_ms=%.3f quality_failures=%d",
                self.stats.processing_fps,
                self.stats.processed_frames,
                self.stats.avg_latency_ms,
                self.stats.quality_failures,
            )
            self.stats._window_processed = 0
            self.stats._window_start = time.monotonic()
        self.stats.last_quality = quality.as_dict()
        if not quality.valid:
            self.stats.quality_failures += 1

        logger.debug(
            "frame_id=%s camera_id=%s latency_ms=%.3f quality_valid=%s",
            frame_id, packet.camera_id, latency_ms, quality.valid,
        )
        return ai_frame

    def run_ai(self, ai_frame: AIFrame) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.ai_processor.process(ai_frame)
        self.last_ai_result = result
        ai_latency_ms = (time.perf_counter() - started) * 1000.0
        result["ai_latency_ms"] = round(ai_latency_ms, 3)
        return result

    def get_status(self) -> dict[str, Any]:
        return {
            "sample_every_n": self.sample_every_n,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "resize_mode": self.resize_mode,
            "stats": self.stats.as_dict(),
            "last_ai_result": self.last_ai_result,
        }

    def reset(self) -> None:
        self.stats = PipelineStats()
        self.last_ai_result: Optional[dict[str, Any]] = None
        self._last_gray = None
        self._same_frame_count = 0
