from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:  # Prefer ONNX Runtime to avoid intermittent OpenCV 5 DNN graph-engine failures.
    import onnxruntime as ort
except ImportError:  # OpenCV fallback keeps existing installations runnable.
    ort = None


COCO_EQUIPMENT_CLASSES: dict[int, str] = {
    13: "bench",       # Classroom benches / benches
    56: "chair",       # Chairs
    57: "couch",       # Lounge chairs / seating
    60: "table",       # Dining tables / classroom desks
    62: "tv_monitor",  # TVs / computer monitors / projector screens
    63: "laptop",      # Laptops / notebooks
    66: "keyboard",    # Keyboards
}


@dataclass
class EquipmentDetection:
    class_id: int
    class_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    zone_id: str | None = None

    @property
    def bbox(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": self.bbox,
            "confidence": round(float(self.confidence), 4),
            "zone_id": self.zone_id,
        }


class InfrastructureDetector:
    """YOLOv8 COCO equipment subset with diagnostics and resilient inference.

    The standard YOLOv8 COCO ONNX output is (1, 84, 8400): four box values
    followed by 80 class scores. There is no separate objectness score.
    ONNX Runtime is preferred; OpenCV DNN is retained as a fallback.
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.15,
        nms_threshold: float = 0.45,
        min_area: int = 100,
        classes: dict[int, str] | None = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Infrastructure detector model not found: {self.model_path}")
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.min_area = int(min_area)
        self.classes = dict(classes or COCO_EQUIPMENT_CLASSES)
        self._session = None
        self._input_name: str | None = None
        self._net = None
        self.engine = "opencv-dnn"
        self.inference_count = 0
        self.failure_count = 0
        self.last_inference_ms = 0.0
        self.last_detections: list[EquipmentDetection] = []
        self.last_error: str | None = None
        self.last_output_shape: list[int] | None = None
        self.last_class_max_scores: dict[str, float] = {}
        self.last_candidate_count = 0
        self.last_frame_stats: dict[str, Any] = {}
        self.last_all_class_max_score = 0.0
        self.last_top_class_scores: list[dict[str, Any]] = []
        self._load_engine()

    def _load_engine(self) -> None:
        """Load preferred CPU engine, then fall back to OpenCV DNN."""
        self._session = None
        if ort is not None:
            try:
                options = ort.SessionOptions()
                options.log_severity_level = 3
                self._session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                self.engine = "onnxruntime-cpu"
                return
            except Exception as exc:
                self.last_error = f"ONNX Runtime load failed; using OpenCV DNN: {exc}"
        self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.engine = "opencv-dnn"

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        scale = min(self.input_size / max(w, 1), self.input_size / max(h, 1))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        px, py = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, scale, px, py

    def _forward(self, canvas: np.ndarray) -> np.ndarray:
        # YOLOv8 ONNX expects RGB float32 NCHW values in [0, 1].
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        if self._session is not None:
            return np.asarray(self._session.run(None, {self._input_name: tensor})[0])
        blob = cv2.dnn.blobFromImage(
            canvas, scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size), swapRB=True, crop=False,
        )
        try:
            self._net.setInput(blob)
            return np.asarray(self._net.forward())
        except Exception:
            # OpenCV 5 can intermittently fail in its graph engine. Recreate the
            # network once before surfacing an error; do not fake a zero detection.
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._net.setInput(blob)
            return np.asarray(self._net.forward())

    @staticmethod
    def _normalize_output(output: np.ndarray) -> np.ndarray:
        arr = np.asarray(output)
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise RuntimeError(f"Only batch size 1 is supported; output shape={arr.shape}")
            arr = arr[0]
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected YOLO output shape: {arr.shape}")
        # YOLOv8 COCO is (84, N); row-oriented is (N, 84).
        if arr.shape[0] in (84, 85) and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        if arr.shape[1] < 5:
            raise RuntimeError(f"YOLO output has too few columns: {arr.shape}")
        if not np.isfinite(arr).all():
            raise RuntimeError("YOLO output contains NaN or infinity")
        return arr

    def detect(self, frame: np.ndarray) -> list[EquipmentDetection]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.last_detections = []
            self.last_error = "Empty or invalid frame supplied to infrastructure detector"
            self.failure_count += 1
            raise ValueError(self.last_error)
        if frame.ndim != 3 or frame.shape[2] != 3:
            self.last_error = f"Expected a BGR image with 3 channels; got shape={frame.shape}"
            self.failure_count += 1
            raise ValueError(self.last_error)

        started = time.perf_counter()
        try:
            # Frame telemetry helps distinguish a real empty room from a stale,
            # black, washed-out, or otherwise unsuitable camera frame.
            frame_f = frame.astype(np.float32)
            self.last_frame_stats = {
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "min": int(np.min(frame)),
                "max": int(np.max(frame)),
                "mean": round(float(np.mean(frame_f)), 3),
                "std": round(float(np.std(frame_f)), 3),
                "dark_pixel_pct": round(float(np.mean(np.max(frame, axis=2) < 16) * 100.0), 2),
                "bright_pixel_pct": round(float(np.mean(np.min(frame, axis=2) > 245) * 100.0), 2),
            }
            canvas, scale, pad_x, pad_y = self._letterbox(frame)
            output = self._normalize_output(self._forward(canvas))
            self.last_output_shape = list(np.asarray(output).shape)
            class_scores = output[:, 4:]
            all_max = np.max(class_scores, axis=0)
            self.last_all_class_max_score = round(float(np.max(all_max)), 6) if all_max.size else 0.0
            top_ids = np.argsort(all_max)[::-1][:10]
            # COCO names are diagnostic only; these are not used for filtering.
            coco_names = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
                "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"
            ]
            self.last_top_class_scores = [
                {"class_id": int(cid), "class_name": coco_names[int(cid)] if int(cid) < len(coco_names) else str(int(cid)),
                 "max_score": round(float(all_max[int(cid)]), 6)}
                for cid in top_ids
            ]
            self.last_class_max_scores = {
                name: round(float(np.max(class_scores[:, class_id])), 6)
                if class_id < class_scores.shape[1] else 0.0
                for class_id, name in self.classes.items()
            }

            boxes_by_class: dict[int, list[list[int]]] = {cid: [] for cid in self.classes}
            scores_by_class: dict[int, list[float]] = {cid: [] for cid in self.classes}
            fh, fw = frame.shape[:2]
            for row in output:
                cx, cy, bw, bh = map(float, row[:4])
                if bw <= 0 or bh <= 0:
                    continue
                for class_id in self.classes:
                    if class_id >= class_scores.shape[1]:
                        continue
                    score = float(row[4 + class_id])
                    if score < self.confidence_threshold:
                        continue
                    x1 = max(0.0, min(float(fw), (cx - bw / 2 - pad_x) / scale))
                    y1 = max(0.0, min(float(fh), (cy - bh / 2 - pad_y) / scale))
                    x2 = max(0.0, min(float(fw), (cx + bw / 2 - pad_x) / scale))
                    y2 = max(0.0, min(float(fh), (cy + bh / 2 - pad_y) / scale))
                    box_w, box_h = int(round(x2 - x1)), int(round(y2 - y1))
                    if box_w <= 0 or box_h <= 0 or box_w * box_h < self.min_area:
                        continue
                    boxes_by_class[class_id].append([int(round(x1)), int(round(y1)), box_w, box_h])
                    scores_by_class[class_id].append(score)

            detections: list[EquipmentDetection] = []
            for class_id, boxes in boxes_by_class.items():
                scores = scores_by_class[class_id]
                if not boxes:
                    continue
                keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold)
                indices = np.asarray(keep).reshape(-1).tolist() if len(keep) else []
                for index in indices:
                    x, y, width, height = boxes[int(index)]
                    detections.append(EquipmentDetection(
                        class_id, self.classes[class_id], x, y, width, height, scores[int(index)]
                    ))

            self.last_detections = detections
            self.last_candidate_count = sum(len(v) for v in boxes_by_class.values())
            self.inference_count += 1
            self.last_inference_ms = (time.perf_counter() - started) * 1000.0
            self.last_error = None
            return detections
        except Exception as exc:
            self.last_detections = []
            self.last_inference_ms = (time.perf_counter() - started) * 1000.0
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.failure_count += 1
            raise

    def status(self) -> dict[str, Any]:
        return {
            "model": "YOLOv8n ONNX / COCO equipment subset",
            "model_path": str(self.model_path),
            "device": "CPU",
            "engine": self.engine,
            "loaded": self._session is not None or self._net is not None,
            "supported_classes": [{"class_id": key, "class_name": value} for key, value in self.classes.items()],
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "inference_count": self.inference_count,
            "failure_count": self.failure_count,
            "last_inference_ms": round(self.last_inference_ms, 3),
            "last_output_shape": self.last_output_shape,
            "last_class_max_scores": self.last_class_max_scores,
            "last_candidate_count": self.last_candidate_count,
            "last_frame_stats": self.last_frame_stats,
            "last_all_class_max_score": self.last_all_class_max_score,
            "last_top_class_scores": self.last_top_class_scores,
            "last_error": self.last_error,
            "last_detections": [d.as_dict() for d in self.last_detections],
        }

    def reset(self) -> None:
        self.inference_count = 0
        self.failure_count = 0
        self.last_inference_ms = 0.0
        self.last_detections = []
        self.last_error = None
        self.last_output_shape = None
        self.last_class_max_scores = {}
        self.last_candidate_count = 0
