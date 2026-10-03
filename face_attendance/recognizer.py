from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class SFaceRecognizer:
    """Local OpenCV SFace wrapper for protected face representations."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = None
        self.loaded = False
        self.last_error: str | None = None
        self.last_inference_ms = 0.0

    def load(self) -> bool:
        if not self.model_path.exists():
            self.loaded = False
            self.last_error = (
                f"SFace model not found: {self.model_path}. "
                "Run: python tools/download_sface_model.py"
            )
            return False
        try:
            self.model = cv2.FaceRecognizerSF.create(
                model=str(self.model_path),
                config="",
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
            self.loaded = True
            self.last_error = None
            return True
        except Exception as exc:
            self.model = None
            self.loaded = False
            self.last_error = str(exc)
            return False

    def extract(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        if not self.loaded and not self.load():
            raise RuntimeError(self.last_error or "SFace model could not be loaded.")
        started = time.perf_counter()
        aligned = self.model.alignCrop(frame, face_row)
        feature = self.model.feature(aligned)
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        return vector

    @staticmethod
    def aggregate(features: list[np.ndarray]) -> np.ndarray:
        if not features:
            raise ValueError("At least one face representation is required.")
        matrix = np.vstack([np.asarray(v, dtype=np.float32).reshape(1, -1) for v in features])
        vector = matrix.mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)

    def status(self) -> dict[str, Any]:
        return {
            "model": "SFace",
            "model_path": str(self.model_path),
            "loaded": self.loaded,
            "last_inference_ms": round(self.last_inference_ms, 3),
            "last_error": self.last_error,
        }
