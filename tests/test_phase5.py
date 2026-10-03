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
