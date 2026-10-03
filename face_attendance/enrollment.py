from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .face_detector import FaceDetector
from .recognizer import SFaceRecognizer


@dataclass
class EnrollmentSample:
    feature: np.ndarray
    confidence: float
    brightness: float
    blur_score: float


class EnrollmentService:
    """Captures consented demo samples from the active camera and creates one protected representation."""

    def __init__(self, detector: FaceDetector, recognizer: SFaceRecognizer):
        self.detector = detector
        self.recognizer = recognizer

    def process_frame(self, frame: np.ndarray) -> EnrollmentSample:
        detections = self.detector.detect(frame)
        if len(detections) != 1:
            raise ValueError(f"Enrollment requires exactly one visible face; found {len(detections)}.")
        detection = detections[0]
        if not detection.quality_valid:
            reasons = ", ".join(detection.quality_reasons) or "low_quality"
            raise ValueError(f"Face quality is too low: {reasons}.")
        if detection.confidence < self.detector.confidence_threshold:
            raise ValueError("Face confidence is below the enrollment threshold.")

        row = np.array(
            [
                detection.x,
                detection.y,
                detection.width,
                detection.height,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                detection.confidence,
            ],
            dtype=np.float32,
        )
        # YuNet's SFace alignment needs the five landmarks. Re-run detector to obtain the full row.
        raw = self.detector.detector
        if raw is None:
            raise RuntimeError("YuNet detector is not loaded.")
        h, w = frame.shape[:2]
        raw.setInputSize((w, h))
        _, raw_faces = raw.detect(frame)
        if raw_faces is None or len(raw_faces) == 0:
            raise ValueError("Face disappeared before enrollment embedding was generated.")
        best = min(
            raw_faces,
            key=lambda r: abs(float(r[0]) - detection.x) + abs(float(r[1]) - detection.y),
        )
        feature = self.recognizer.extract(frame, np.asarray(best, dtype=np.float32))
        return EnrollmentSample(feature, detection.confidence, detection.brightness, detection.blur_score)

    @staticmethod
    def quality_score(samples: list[EnrollmentSample]) -> float:
        if not samples:
            return 0.0
        scores = []
        for sample in samples:
            brightness_score = min(1.0, sample.brightness / 80.0)
            blur_score = min(1.0, sample.blur_score / 120.0)
            confidence_score = max(0.0, min(1.0, sample.confidence))
            scores.append(0.45 * confidence_score + 0.3 * brightness_score + 0.25 * blur_score)
        return float(sum(scores) / len(scores))
