import numpy as np
import pytest
from pathlib import Path

from face_attendance.face_detector import FaceDetector, FaceDetection


@pytest.fixture
def detector():
    model_path = Path("models/face_detection_yunet_2023mar.onnx")
    if not model_path.exists():
        pytest.skip("YuNet model not found.")
    d = FaceDetector(model_path, confidence_threshold=0.6)
    loaded = d.load()
    assert loaded is True
    return d


def test_face_detector_initialization(detector):
    assert detector.loaded is True
    status = detector.status()
    assert status["loaded"] is True
    assert status["confidence_threshold"] == 0.6
    assert status["active_tracks"] == 0


def test_face_detector_on_blank_frame(detector):
    blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank_frame)
    assert isinstance(detections, list)
    assert len(detections) == 0
    assert detector.last_inference_ms > 0.0


def test_face_detection_dataclass():
    det = FaceDetection(
        x=50,
        y=60,
        width=80,
        height=90,
        confidence=0.88,
        quality_valid=True,
        quality_reasons=[],
        brightness=120.0,
        blur_score=150.0,
        track_id=1,
    )
    assert det.bbox == [50, 60, 80, 90]
    d = det.as_dict()
    assert d["confidence"] == 0.88
    assert d["track_id"] == 1
    assert d["quality_valid"] is True


def test_face_detector_reset(detector):
    detector.reset()
    status = detector.status()
    assert status["loaded"] is True
    assert status["active_tracks"] == 0
