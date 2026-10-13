from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class FaceDetection:
    """
    One local YuNet face detection.

    The complete YuNet model row is preserved in `model_row`
    because SFace requires the facial landmark coordinates
    contained in that row for alignCrop().
    """

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

    # Complete 15-value YuNet output row.
    # Used internally by SFace for landmark-based alignment.
    model_row: Optional[np.ndarray] = None

    @property
    def bbox(self) -> list[int]:
        return [
            self.x,
            self.y,
            self.width,
            self.height,
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": round(float(self.confidence), 4),
            "quality_valid": bool(self.quality_valid),
            "quality_reasons": list(self.quality_reasons),
            "brightness": round(float(self.brightness), 2),
            "blur_score": round(float(self.blur_score), 2),
            "track_id": self.track_id,
        }


@dataclass
class FaceTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    missed: int = 0


class FaceDetector:
    """
    Local YuNet face detector with:

    - confidence filtering
    - face quality checks
    - brightness check
    - blur check
    - IoU-based face tracking
    - complete YuNet landmark preservation for SFace

    This module only detects/tracks faces.
    Recognition is handled separately by FaceRecognitionService.
    """

    def __init__(
        self,
        model_path: str | Path = "models/face_detection_yunet_2023mar.onnx",
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        min_face_size: int = 20,
        face_brightness_min: float = 20.0,
        face_blur_threshold: float = 8.0,
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

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    def load(self) -> bool:
        """
        Load YuNet ONNX model through OpenCV.
        """

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

    # ---------------------------------------------------------
    # QUALITY CHECK
    # ---------------------------------------------------------

    def _quality(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[bool, list[str], float, float]:

        x, y, w, h = bbox

        frame_height, frame_width = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(frame_width, x + w)
        y2 = min(frame_height, y + h)

        if x2 <= x1 or y2 <= y1:
            return (
                False,
                ["invalid_bbox"],
                0.0,
                0.0,
            )

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return (
                False,
                ["empty_face_crop"],
                0.0,
                0.0,
            )

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        brightness = float(
            np.mean(gray)
        )

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        reasons: list[str] = []

        if min(w, h) < self.min_face_size:
            reasons.append("face_too_small")

        if brightness < self.face_brightness_min:
            reasons.append("face_too_dark")

        if blur_score < self.face_blur_threshold:
            reasons.append("face_blurred")

        return (
            len(reasons) == 0,
            reasons,
            brightness,
            blur_score,
        )

    # ---------------------------------------------------------
    # IOU
    # ---------------------------------------------------------

    @staticmethod
    def _iou(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:

        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax2 = ax + aw
        ay2 = ay + ah

        bx2 = bx + bw
        by2 = by + bh

        ix1 = max(ax, bx)
        iy1 = max(ay, by)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        intersection = iw * ih

        if intersection == 0:
            return 0.0

        union = (
            aw * ah
            + bw * bh
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # ---------------------------------------------------------
    # TRACKING
    # ---------------------------------------------------------

    def _track(
        self,
        detections: list[FaceDetection],
    ) -> None:

        unmatched_tracks = set(
            range(len(self._tracks))
        )

        matched_detections = set()

        pairs: list[
            tuple[float, int, int]
        ] = []

        # Build all IoU pairs.
        for track_index, track in enumerate(
            self._tracks
        ):
            for detection_index, detection in enumerate(
                detections
            ):

                iou = self._iou(
                    track.bbox,
                    tuple(detection.bbox),
                )

                pairs.append(
                    (
                        iou,
                        track_index,
                        detection_index,
                    )
                )

        # Highest IoU first.
        pairs.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # Match existing tracks.
        for (
            iou,
            track_index,
            detection_index,
        ) in pairs:

            if (
                iou < self.track_iou_threshold
                or track_index not in unmatched_tracks
                or detection_index in matched_detections
            ):
                continue

            detection = detections[
                detection_index
            ]

            track = self._tracks[
                track_index
            ]

            track.bbox = tuple(
                detection.bbox
            )

            track.confidence = (
                detection.confidence
            )

            track.missed = 0

            detection.track_id = (
                track.track_id
            )

            unmatched_tracks.remove(
                track_index
            )

            matched_detections.add(
                detection_index
            )

        # Create tracks for new detections.
        for detection_index, detection in enumerate(
            detections
        ):

            if (
                detection_index
                in matched_detections
            ):
                continue

            track = FaceTrack(
                track_id=self._next_track_id,
                bbox=tuple(
                    detection.bbox
                ),
                confidence=detection.confidence,
            )

            self._next_track_id += 1

            self._tracks.append(track)

            detection.track_id = (
                track.track_id
            )

        # Age unmatched tracks.
        for track_index in sorted(
            unmatched_tracks,
            reverse=True,
        ):

            self._tracks[
                track_index
            ].missed += 1

            if (
                self._tracks[
                    track_index
                ].missed
                > self.max_missed_frames
            ):

                self._tracks.pop(
                    track_index
                )

    # ---------------------------------------------------------
    # DETECTION
    # ---------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[FaceDetection]:

        if not self.loaded:

            if not self.load():

                self.last_detections = []

                return []

        if (
            frame is None
            or frame.size == 0
        ):

            self.last_detections = []

            return []

        started = time.perf_counter()

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # YuNet must know the actual frame size.
        self.detector.setInputSize(
            (
                frame_width,
                frame_height,
            )
        )

        _, raw_faces = (
            self.detector.detect(frame)
        )

        detections: list[
            FaceDetection
        ] = []

        if raw_faces is not None:

            for row in raw_faces:

                # -------------------------------------------------
                # CRITICAL:
                #
                # Keep the COMPLETE YuNet 15-value row.
                #
                # YuNet:
                #
                # [0]  x
                # [1]  y
                # [2]  width
                # [3]  height
                #
                # [4:14] = five landmark pairs
                #
                # [14] = confidence
                #
                # SFace alignCrop() requires these landmarks.
                # -------------------------------------------------

                model_row = np.asarray(
                    row,
                    dtype=np.float32,
                ).copy()

                if model_row.size < 15:
                    continue

                x = float(
                    model_row[0]
                )

                y = float(
                    model_row[1]
                )

                w = float(
                    model_row[2]
                )

                h = float(
                    model_row[3]
                )

                confidence = float(
                    model_row[14]
                )

                bbox = (
                    int(round(x)),
                    int(round(y)),
                    int(round(w)),
                    int(round(h)),
                )

                (
                    quality_valid,
                    quality_reasons,
                    brightness,
                    blur_score,
                ) = self._quality(
                    frame,
                    bbox,
                )

                detection = FaceDetection(
                    x=bbox[0],
                    y=bbox[1],
                    width=bbox[2],
                    height=bbox[3],
                    confidence=confidence,
                    quality_valid=quality_valid,
                    quality_reasons=quality_reasons,
                    brightness=brightness,
                    blur_score=blur_score,

                    # IMPORTANT:
                    # Pass the COMPLETE YuNet row
                    # to SFace.
                    model_row=model_row,
                )

                detections.append(
                    detection
                )

        # Assign stable face track IDs.
        self._track(
            detections
        )

        self.last_detections = (
            detections
        )

        self.inference_count += 1

        self.total_faces += len(
            detections
        )

        self.last_inference_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        return detections

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def status(
        self,
    ) -> dict[str, Any]:

        return {
            "model": "YuNet",

            "model_path": str(
                self.model_path
            ),

            "loaded": self.loaded,

            "confidence_threshold": (
                self.confidence_threshold
            ),

            "nms_threshold": (
                self.nms_threshold
            ),

            "min_face_size": (
                self.min_face_size
            ),

            "face_brightness_min": (
                self.face_brightness_min
            ),

            "face_blur_threshold": (
                self.face_blur_threshold
            ),

            "inference_count": (
                self.inference_count
            ),

            "total_faces": (
                self.total_faces
            ),

            "last_inference_ms": round(
                self.last_inference_ms,
                3,
            ),

            "active_tracks": len(
                self._tracks
            ),

            "detections": [
                detection.as_dict()
                for detection in
                self.last_detections
            ],

            "last_error": (
                self.last_error
            ),
        }

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self) -> None:

        self.last_detections = []

        self.inference_count = 0

        self.total_faces = 0

        self.last_inference_ms = 0.0

        self._tracks = []

        self._next_track_id = 1

        self.last_error = None