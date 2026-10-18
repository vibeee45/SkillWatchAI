import pytest
from pathlib import Path
from face_attendance.attendance import AttendanceStore


@pytest.fixture
def store(tmp_path):
    db_file = tmp_path / "test_attendance.db"
    return AttendanceStore(db_file)


def test_classroom_and_batch_creation(store):
    room = store.upsert_classroom("CR-101", "Room 101", "Building A")
    assert room["classroom_id"] == "CR-101"

    batch = store.upsert_batch("BATCH-2026-A", "Electrician Batch A")
    assert batch["batch_id"] == "BATCH-2026-A"

    classrooms = store.list_classrooms()
    assert any(c["classroom_id"] == "CR-101" for c in classrooms)

    batches = store.list_batches()
    assert any(b["batch_id"] == "BATCH-2026-A" for b in batches)


def test_session_lifecycle(store):
    store.upsert_classroom("CR-102", "Room 102", "Main Hall")
    store.upsert_batch("BATCH-B", "Welder Batch B")

    session = store.create_session(
        classroom_id="CR-102",
        batch_id="BATCH-B",
        session_date="2026-10-19",
        start_time="09:00",
        end_time="11:00",
        camera_id="CAM-001",
    )
    assert session["id"] is not None
    assert session["status"] == "SCHEDULED"

    # Start session
    active = store.start_session(session["id"])
    assert active["status"] == "ACTIVE"
    assert active["started_at"] is not None

    active_camera_session = store.active_session_for_camera("CAM-001")
    assert active_camera_session["id"] == session["id"]

    # End session
    ended = store.end_session(session["id"])
    assert ended["status"] == "ENDED"
    assert ended["ended_at"] is not None

    assert store.active_session_for_camera("CAM-001") is None


def test_duplicate_attendance_prevention(store):
    session = store.create_session(
        classroom_id="CR-103",
        batch_id="BATCH-C",
        session_date="2026-10-19",
        start_time="10:00",
        end_time="12:00",
        camera_id="CAM-002",
    )
    store.start_session(session["id"])

    # First check-in -> created is True
    record1, created1 = store.mark_present(session["id"], "STU-001", confidence=0.85, camera_id="CAM-002")
    assert created1 is True
    assert record1["student_id"] == "STU-001"
    assert record1["status"] == "PRESENT"

    # Second check-in for the same student in same session -> created is False
    record2, created2 = store.mark_present(session["id"], "STU-001", confidence=0.92, camera_id="CAM-002")
    assert created2 is False
    assert record2["student_id"] == "STU-001"


def test_attendance_summary_and_manual_correction(store):
    session = store.create_session(
        classroom_id="CR-104",
        batch_id="BATCH-D",
        session_date="2026-10-19",
        start_time="14:00",
        end_time="16:00",
        camera_id="CAM-003",
    )
    store.start_session(session["id"])

    store.mark_present(session["id"], "STU-001", confidence=0.90, camera_id="CAM-003")

    dummy_students = [
        {"student_id": "STU-001", "name": "Alice", "batch": "BATCH-D"},
        {"student_id": "STU-002", "name": "Bob", "batch": "BATCH-D"},
    ]

    summary = store.summary(session["id"], dummy_students)
    assert summary["total_students"] == 2
    assert summary["present_count"] == 1
    assert summary["absent_count"] == 1
    assert summary["attendance_percentage"] == 50.0

    # Manual correction: mark Bob present using store.correct()
    corrected = store.correct(
        session_id=session["id"],
        student_id="STU-002",
        status="PRESENT",
        reason="Biometric device lag, verified by trainer",
        actor="trainer_raj",
    )
    assert corrected["status"] == "PRESENT"

    # Verify audit log using store.audit()
    audit_records = store.audit(session["id"])
    assert len(audit_records) >= 1
    assert any(a["student_id"] == "STU-002" and a["actor"] == "trainer_raj" for a in audit_records)
