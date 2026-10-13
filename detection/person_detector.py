from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class PersonDetection:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    track_id: int | None = None
    zone: str | None = None

    @property
    def bbox(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": round(float(self.confidence), 4),
            "track_id": self.track_id,
            "zone": self.zone,
        }


class LightweightPersonDetector:
    """
    CPU-friendly YOLOv8n person detector running through OpenCV DNN.

    YOLOv8n is used only for PERSON detection.
    Identity remains completely separate and is supplied by
    the Phase 6 YuNet + SFace recognition pipeline.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        min_area: int = 500,
    ):
        self.model_path = Path(model_path)
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.min_area = int(min_area)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO person detector model not found: {self.model_path}"
            )

        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))

        # CPU-only inference.
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.inference_count = 0
        self.total_people = 0
        self.last_inference_ms = 0.0
        self.last_detections: list[PersonDetection] = []

    def _letterbox(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, float, int, int]:
        """
        Resize while preserving aspect ratio and pad to square.
        Returns:
            image,
            scale,
            pad_x,
            pad_y
        """
        height, width = frame.shape[:2]

        scale = min(
            self.input_size / max(width, 1),
            self.input_size / max(height, 1),
        )

        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8,
        )

        pad_x = (self.input_size - new_width) // 2
        pad_y = (self.input_size - new_height) // 2

        canvas[
            pad_y : pad_y + new_height,
            pad_x : pad_x + new_width,
        ] = resized

        return canvas, scale, pad_x, pad_y

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        if frame is None or frame.size == 0:
            self.last_detections = []
            return []

        started = time.perf_counter()

        input_image, scale, pad_x, pad_y = self._letterbox(frame)

        blob = cv2.dnn.blobFromImage(
            input_image,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        output = self.net.forward()

        output = np.asarray(output)

        # Typical YOLOv8 ONNX output:
        # (1, 84, 8400)
        #
        # Convert to:
        # (8400, 84)
        if output.ndim == 3:
            output = output[0]

        if output.ndim != 2:
            raise RuntimeError(
                f"Unexpected YOLO output shape: {output.shape}"
            )

        # YOLOv8 normally returns [attributes, predictions].
        if output.shape[0] < output.shape[1]:
            output = output.T

        boxes: list[list[int]] = []
        scores: list[float] = []

        for row in output:
            if row.shape[0] < 6:
                continue

            cx = float(row[0])
            cy = float(row[1])
            width = float(row[2])
            height = float(row[3])

            # YOLOv8:
            # [x, y, w, h, class_0, class_1, ...]
            #
            # Older YOLO exports:
            # [x, y, w, h, objectness, class_0, ...]
            if row.shape[0] == 84:
                class_scores = row[4:]
                confidence = float(class_scores[self.PERSON_CLASS_ID])
            else:
                objectness = float(row[4])
                class_scores = row[5:]

                if self.PERSON_CLASS_ID >= len(class_scores):
                    continue

                confidence = (
                    objectness
                    * float(class_scores[self.PERSON_CLASS_ID])
                )

            if confidence < self.confidence_threshold:
                continue

            # Coordinates are in the 640x640 letterboxed image.
            x1 = (cx - width / 2.0 - pad_x) / scale
            y1 = (cy - height / 2.0 - pad_y) / scale
            x2 = (cx + width / 2.0 - pad_x) / scale
            y2 = (cy + height / 2.0 - pad_y) / scale

            frame_height, frame_width = frame.shape[:2]

            x1 = max(0.0, min(float(frame_width - 1), x1))
            y1 = max(0.0, min(float(frame_height - 1), y1))
            x2 = max(0.0, min(float(frame_width), x2))
            y2 = max(0.0, min(float(frame_height), y2))

            box_width = int(round(x2 - x1))
            box_height = int(round(y2 - y1))

            if box_width <= 0 or box_height <= 0:
                continue

            if box_width * box_height < self.min_area:
                continue

            boxes.append(
                [
                    int(round(x1)),
                    int(round(y1)),
                    box_width,
                    box_height,
                ]
            )
            scores.append(confidence)

        if boxes:
            keep = cv2.dnn.NMSBoxes(
                boxes,
                scores,
                self.confidence_threshold,
                self.nms_threshold,
            )

            indices = [
                int(i)
                for i in np.array(keep).reshape(-1)
            ] if len(keep) else []

            detections = [
                PersonDetection(
                    *boxes[i],
                    confidence=scores[i],
                )
                for i in indices
            ]
        else:
            detections = []

        self.last_detections = detections
        self.inference_count += 1
        self.total_people += len(detections)
        self.last_inference_ms = (
            time.perf_counter() - started
        ) * 1000.0

        return detections

    def status(self) -> dict[str, Any]:
        return {
            "model": "YOLOv8n ONNX person detector",
            "device": "CPU",
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "input_size": self.input_size,
            "inference_count": self.inference_count,
            "total_detected": self.total_people,
            "last_inference_ms": round(
                self.last_inference_ms,
                3,
            ),
            "last_detections": [
                d.as_dict()
                for d in self.last_detections
            ],
        }

    def reset(self) -> None:
        self.inference_count = 0
        self.total_people = 0
        self.last_inference_ms = 0.0
        self.last_detections = []