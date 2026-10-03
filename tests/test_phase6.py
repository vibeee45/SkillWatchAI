from pathlib import Path

import numpy as np

from face_attendance.face_detector import FaceDetection
from face_attendance.protected_store import ProtectedRepresentationStore
from face_attendance.recognition import FaceRecognitionService
from face_attendance.student_store import StudentStore


class FakeRecognizer:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.last_inference_ms = 0.1

    def extract(self, frame, face_row):
        return self.vector.copy()


def make_detection(track_id=1, quality_valid=True):
    row = np.zeros(15, dtype=np.float32)
    row[0:4] = [10, 10, 80, 80]
    row[14] = 0.95
    return FaceDetection(
        x=10, y=10, width=80, height=80, confidence=0.95,
        quality_valid=quality_valid, brightness=80, blur_score=100,
        track_id=track_id, model_row=row,
    )


def make_service(tmp_path: Path, enrolled_vector):
    students = StudentStore(tmp_path / "students.db")
    protected = ProtectedRepresentationStore(tmp_path / ".face_key")
    token = protected.protect(np.asarray(enrolled_vector, dtype=np.float32))
    students.upsert("STU-001", "Demo Student", "Batch A", token, 3, 0.9, True)
    return FaceRecognitionService(
        students,
        protected,
        FakeRecognizer(enrolled_vector),
        tmp_path / "recognition_log.jsonl",
        threshold=0.45,
        confirmation_frames=3,
        switch_confirmation_frames=3,
        switch_margin=0.05,
    )


def test_known_student_requires_temporal_confirmation(tmp_path):
    service = make_service(tmp_path, [1.0, 0.0, 0.0])
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    detection = make_detection()

    results = [service.recognize(frame, [detection])[0] for _ in range(3)]

    assert results[0].status == "candidate"
    assert results[1].status == "candidate"
    assert results[2].status == "confirmed"
    assert results[2].student_id == "STU-001"
    assert results[2].similarity > 0.99


def test_unknown_below_threshold(tmp_path):
    service = make_service(tmp_path, [1.0, 0.0, 0.0])
    service.recognizer.vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    result = service.recognize(frame, [make_detection()])[0]
    assert result.status == "unknown"
    assert result.student_id is None
    assert result.reason == "below_threshold"


def test_low_quality_is_unknown(tmp_path):
    service = make_service(tmp_path, [1.0, 0.0, 0.0])
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    result = service.recognize(frame, [make_detection(quality_valid=False)])[0]
    assert result.status == "unknown"
    assert result.reason == "low_quality"


def test_recognition_log_contains_timestamp_and_confidence(tmp_path):
    service = make_service(tmp_path, [1.0, 0.0, 0.0])
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    service.recognize(frame, [make_detection()], camera_id="CAM-001", frame_id="CAM-001-00000001")
    logs = service.logger.recent(10)
    assert len(logs) == 1
    assert logs[0]["timestamp"]
    assert "confidence" in logs[0]
    assert logs[0]["camera_id"] == "CAM-001"
