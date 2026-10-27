import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_camera_connect_local(client):
    """Verify Local PC Camera connection works via /api/camera/connect."""
    res = client.post("/api/camera/connect", json={
        "source_type": "local",
        "camera_id": "TEST-CAM-LOCAL",
        "room": "Room 101"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"]["connected"] is True
    assert data["status"]["source_type"] == "local"

    # Feed a simulated browser frame
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)
    frame_bytes = enc.tobytes()

    upload_res = client.post(
        "/api/camera/browser-frame",
        files={"frame": ("frame.jpg", io.BytesIO(frame_bytes), "image/jpeg")}
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["status"] == "ok"
    assert upload_data["width"] == 640
    assert upload_data["height"] == 360
    assert "annotated_image" in upload_data
    assert "recognition" in upload_data
    assert "occupancy" in upload_data

    # Disconnect
    disc_res = client.post("/api/camera/disconnect")
    assert disc_res.status_code == 200
    assert disc_res.json()["status"]["connected"] is False

def test_camera_connect_classroom_validation(client):
    """Verify Classroom Camera connection checks for RTSP URL."""
    res = client.post("/api/camera/connect", json={
        "source_type": "classroom",
        "camera_id": "TEST-CAM-CLASSROOM",
        "room": "Room 102"
        # missing rtsp_url
    })
    assert res.status_code == 400
    assert "rtsp" in res.json()["detail"].lower()


def test_quick_add_student_validation_and_camera_check(client):
    """Verify quick add student endpoint validates inputs and checks camera connection."""
    # Ensure camera is disconnected
    client.post("/api/camera/disconnect")

    # 1. Reject without camera connection
    res = client.post("/api/students/quick-add", json={
        "student_id": "STU-999",
        "name": "Test Student",
        "batch": "Batch A",
        "consent_confirmed": True
    })
    assert res.status_code == 409
    assert "connect camera" in res.json()["detail"].lower()

    # 2. Reject missing name
    res = client.post("/api/students/quick-add", json={
        "student_id": "STU-999",
        "name": "",
        "batch": "Batch A",
        "consent_confirmed": True
    })
    assert res.status_code == 400

    # 3. Reject missing consent
    res = client.post("/api/students/quick-add", json={
        "student_id": "STU-999",
        "name": "Test Student",
        "batch": "Batch A",
        "consent_confirmed": False
    })
    assert res.status_code == 400
    assert "consent" in res.json()["detail"].lower()


def test_quick_add_student_with_camera_stream(client, monkeypatch):
    """Verify quick add student captures face representation and upserts into student store."""
    from backend.main import enrollment_service
    from face_attendance.enrollment import EnrollmentSample

    # Connect local camera
    client.post("/api/camera/connect", json={
        "source_type": "local",
        "camera_id": "CAM-TEST",
        "room": "Room Test"
    })

    # Feed a frame
    img = np.full((360, 640, 3), 120, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", img)
    client.post(
        "/api/camera/browser-frame",
        files={"frame": ("frame.jpg", io.BytesIO(enc.tobytes()), "image/jpeg")}
    )

    # Mock enrollment_service.process_frame to return a mock 128-d sample
    mock_feature = np.ones((128,), dtype=np.float32)
    mock_sample = EnrollmentSample(
        feature=mock_feature,
        confidence=0.98,
        brightness=120.0,
        blur_score=150.0
    )
    monkeypatch.setattr(enrollment_service, "process_frame", lambda frame: mock_sample)

    res = client.post("/api/students/quick-add", json={
        "student_id": "STU-QUICK-01",
        "name": "Priya Patel",
        "batch": "Batch A - 2026",
        "consent_confirmed": True
    })
    assert res.status_code == 200
    data = res.json()
    assert data["student"]["student_id"] == "STU-QUICK-01"
    assert data["student"]["name"] == "Priya Patel"
    assert data["sample_count"] >= 1

    # Verify student now appears in student list
    get_res = client.get("/api/students?query=Priya")
    assert get_res.status_code == 200
    students = get_res.json()["students"]
    assert any(s["student_id"] == "STU-QUICK-01" for s in students)

    # Cleanup student & disconnect camera
    client.delete("/api/students/STU-QUICK-01")
    client.post("/api/camera/disconnect")

