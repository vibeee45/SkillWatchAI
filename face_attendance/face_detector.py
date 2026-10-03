from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class FaceDetection:
    """One local face detection. No identity/recognition is performed."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    quality_valid: bool = True
    quality_reasons: list[str] = field(default_factory=list)
    brightness: float = 0.0
    blur_score: float = 0.0
    track_id: Optional[int] = None

    @property
    def bbox(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": round(float(self.confidence), 4),
            "quality_valid": self.quality_valid,
            "quality_reasons": self.quality_reasons,
            "brightness": round(self.brightness, 2),
            "blur_score": round(self.blur_score, 2),
            "track_id": self.track_id,
        }


@dataclass
class FaceTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    missed: int = 0


class FaceDetector:
    """Local YuNet face detector with quality checks and lightweight IoU tracking.

    This module detects faces only. It does not perform face recognition,
    identity matching, embeddings, or attendance decisions.
    """

    def __init__(
        self,
        model_path: str | Path = "models/face_detection_yunet_2023mar.onnx",
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        min_face_size: int = 20,
        face_brightness_min: float = 20.0,
        face_blur_threshold: float = 20.0,
        track_iou_threshold: float = 0.25,
        max_missed_frames: int = 8,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.min_face_size = int(min_face_size)
        self.face_brightness_min = float(face_brightness_min)
        self.face_blur_threshold = float(face_blur_threshold)
        self.track_iou_threshold = float(track_iou_threshold)
        self.max_missed_frames = int(max_missed_frames)

        self.detector = None
        self.loaded = False
        self.last_error: Optional[str] = None
        self.inference_count = 0
        self.total_faces = 0
        self.last_inference_ms = 0.0
        self.last_detections: list[FaceDetection] = []
        self._next_track_id = 1
        self._tracks: list[FaceTrack] = []

    def load(self) -> bool:
        if not self.model_path.exists():
            self.loaded = False
            self.last_error = (
                f"YuNet model not found: {self.model_path}. "
                "Run: python tools/download_yunet_model.py"
            )
            return False
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=(320, 320),
                score_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
            self.loaded = True
            self.last_error = None
            return True
        except Exception as exc:
            self.detector = None
            self.loaded = False
            self.last_error = str(exc)
            return False

    def _quality(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[bool, list[str], float, float]:
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            return False, ["invalid_bbox"], 0.0, 0.0

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        reasons: list[str] = []

        if min(w, h) < self.min_face_size:
            reasons.append("face_too_small")
        if brightness < self.face_brightness_min:
            reasons.append("face_too_dark")
        if blur_score < self.face_blur_threshold:
            reasons.append("face_blurred")

        return not reasons, reasons, brightness, blur_score

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    def _track(self, detections: list[FaceDetection]) -> None:
        unmatched_tracks = set(range(len(self._tracks)))
        matched_det = set()

        # Greedy highest-IoU matching is sufficient for a lightweight prototype.
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                pairs.append((self._iou(track.bbox, tuple(det.bbox)), ti, di))
        pairs.sort(reverse=True)

        for iou, ti, di in pairs:
            if iou < self.track_iou_threshold or ti not in unmatched_tracks or di in matched_det:
                continue
            det = detections[di]
            track = self._tracks[ti]
            track.bbox = tuple(det.bbox)
            track.confidence = det.confidence
            track.missed = 0
            det.track_id = track.track_id
            unmatched_tracks.remove(ti)
            matched_det.add(di)

        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            track = FaceTrack(self._next_track_id, tuple(det.bbox), det.confidence)
            self._next_track_id += 1
            self._tracks.append(track)
            det.track_id = track.track_id

        for ti in sorted(unmatched_tracks, reverse=True):
            self._tracks[ti].missed += 1
            if self._tracks[ti].missed > self.max_missed_frames:
                self._tracks.pop(ti)

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if not self.loaded and not self.load():
            self.last_detections = []
            return []
        if frame is None or frame.size == 0:
            self.last_detections = []
            return []

        started = time.perf_counter()
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, raw_faces = self.detector.detect(frame)
        detections: list[FaceDetection] = []

        if raw_faces is not None:
            for row in raw_faces:
                x, y, w, h = [float(v) for v in row[:4]]
                confidence = float(row[14])
                bbox = (int(round(x)), int(round(y)), int(round(w)), int(round(h)))
                valid, reasons, brightness, blur_score = self._quality(frame, bbox)
                detections.append(
                    FaceDetection(
                        x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3],
                        confidence=confidence,
                        quality_valid=valid,
                        quality_reasons=reasons,
                        brightness=brightness,
                        blur_score=blur_score,
                    )
                )

        self._track(detections)
        self.last_detections = detections
        self.inference_count += 1
        self.total_faces += len(detections)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        return detections

    def status(self) -> dict[str, Any]:
        return {
            "model": "YuNet",
            "model_path": str(self.model_path),
            "loaded": self.loaded,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "min_face_size": self.min_face_size,
            "inference_count": self.inference_count,
            "total_faces": self.total_faces,
            "last_inference_ms": round(self.last_inference_ms, 3),
            "active_tracks": len(self._tracks),
            "detections": [d.as_dict() for d in self.last_detections],
            "last_error": self.last_error,
        }

    def reset(self) -> None:
        self.last_detections = []
        self.inference_count = 0
        self.total_faces = 0
        self.last_inference_ms = 0.0
        self._tracks = []
        self._next_track_id = 1
