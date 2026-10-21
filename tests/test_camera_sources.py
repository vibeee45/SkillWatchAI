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
