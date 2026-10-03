from __future__ import annotations

from typing import Any

from processing.frame_pipeline import AIFrame, AIProcessor

from .face_detector import FaceDetector


class FaceDetectionProcessor(AIProcessor):
    """Phase 4 AI processor: local face detection only."""

    def __init__(self, detector: FaceDetector):
        self.detector = detector

    def process(self, frame: AIFrame) -> dict[str, Any]:
        detections = self.detector.detect(frame.frame)
        return {
            "processor": "face_detection",
            "model": "YuNet",
            "frame_id": frame.frame_id,
            "camera_id": frame.camera_id,
            "room": frame.room,
            "face_count": len(detections),
            "detections": [d.as_dict() for d in detections],
            "inference_latency_ms": round(self.detector.last_inference_ms, 3),
            "model_loaded": self.detector.loaded,
            "error": self.detector.last_error,
        }
