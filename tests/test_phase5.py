from pathlib import Path

import numpy as np

from face_attendance.protected_store import ProtectedRepresentationStore
from face_attendance.student_store import StudentStore


def test_student_store_and_protected_representation(tmp_path: Path):
    store = StudentStore(tmp_path / "students.db")
    protected = ProtectedRepresentationStore(tmp_path / ".face_key")

    vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    token = protected.protect(vector)
    assert "0.1" not in token
    restored = protected.reveal(token, dimension=4)
    assert np.allclose(vector, restored)

    student = store.upsert(
        student_id="STU-001",
        name="Demo Student",
        batch="Batch A",
        representation=token,
        sample_count=3,
        quality_score=0.91,
        consent_confirmed=True,
    )
    assert student["student_id"] == "STU-001"
    assert student["representation_protected"] is True
    assert store.list("Demo")[0]["name"] == "Demo Student"
    assert store.delete("STU-001") is True
    assert store.get("STU-001") is None


def test_camera_latest_frame_does_not_read_twice():
    from camera.camera_manager import CameraManager

    class FakeCapture:
        def __init__(self):
            self.read_calls = 0
        def isOpened(self):
            return True
        def read(self):
            self.read_calls += 1
            return True, np.zeros((20, 30, 3), dtype=np.uint8)
        def release(self):
            pass
        def get(self, prop):
            return 30.0 if prop == 5 else (30.0 if prop == 3 else 20.0)

    manager = CameraManager()
    try:
        fake = FakeCapture()
        assert manager._set_capture(fake, 0, "device", "CAM-001", "Room 01")
        ok, packet = manager.read_frame_packet()
        assert ok and packet is not None
        calls_after_read = fake.read_calls

        ok2, latest = manager.get_latest_frame_packet()
        assert ok2 and latest is not None
        assert fake.read_calls == calls_after_read
    finally:
        manager.shutdown()
