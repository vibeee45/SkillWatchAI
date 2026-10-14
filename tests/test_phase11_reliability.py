import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from camera.camera_manager import CameraManager
from offline.event_queue import SQLiteEventQueue
from offline.sync_worker import EventSyncWorker


def test_sync_worker_delivers_event_and_marks_synced(tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            received.append((json.loads(self.rfile.read(length)), self.headers.get("Idempotency-Key")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        queue = SQLiteEventQueue(tmp_path / "events.db")
        event = queue.enqueue("attendance_created", {"student_id": "S-1"}, event_id="attendance:1:S-1")
        worker = EventSyncWorker(queue, f"http://127.0.0.1:{server.server_port}/events", interval_seconds=1)
        result = worker.sync_once()
        assert result["synced"] == 1
        assert result["failed"] == 0
        assert queue.get(event["event_id"])["status"] == "synced"
        assert received[0][0]["event_id"] == "attendance:1:S-1"
        assert received[0][1] == "attendance:1:S-1"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_no_remote_endpoint_keeps_events_local(tmp_path):
    queue = SQLiteEventQueue(tmp_path / "events.db")
    queue.enqueue("camera_health", {"health": "offline"})
    worker = EventSyncWorker(queue, endpoint="")
    result = worker.sync_once()
    assert result["configured"] is False
    assert queue.status()["pending"] == 1


def test_queue_recovers_events_left_in_syncing_state(tmp_path):
    db = tmp_path / "events.db"
    queue = SQLiteEventQueue(db)
    event = queue.enqueue("attendance", {"student_id": "S-2"})
    assert queue.mark_syncing(event["event_id"])
    reopened = SQLiteEventQueue(db)
    assert reopened.get(event["event_id"])["status"] == "pending"


def test_camera_manager_detects_frozen_frames():
    manager = CameraManager(reconnect_interval=30)
    try:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for _ in range(manager.frozen_frame_limit + 1):
            manager._mark_frame_success(frame)
        status = manager.get_status()
        assert status["health"] == "frozen"
        assert status["connected"] is False
        assert status["frozen_frame_count"] >= manager.frozen_frame_limit
    finally:
        manager.shutdown()


def test_camera_manager_flags_poor_image_quality():
    manager = CameraManager(reconnect_interval=30)
    try:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        for _ in range(manager.poor_quality_frame_limit):
            manager._mark_frame_success(frame)
        status = manager.get_status()
        assert status["health"] == "poor_quality"
        assert status["frame_quality"]["state"] == "low_light"
    finally:
        manager.shutdown()


def test_camera_manager_reports_frame_age_and_quality_metrics():
    manager = CameraManager(reconnect_interval=30)
    try:
        rng = np.random.default_rng(42)
        frame = rng.integers(30, 220, size=(120, 160, 3), dtype=np.uint8)
        manager._mark_frame_success(frame)
        status = manager.get_status()
        assert status["frame_age_seconds"] is not None
        assert "brightness" in status["frame_quality"]
        assert "sharpness" in status["frame_quality"]
    finally:
        manager.shutdown()


def test_sync_worker_recovers_after_receiver_failure(tmp_path):
    state = {"fail": True}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(503 if state["fail"] else 200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        queue = SQLiteEventQueue(tmp_path / "events.db")
        event = queue.enqueue("attendance_created", {"student_id": "S-3"})
        worker = EventSyncWorker(queue, f"http://127.0.0.1:{server.server_port}/events")
        failed = worker.sync_once()
        assert failed["failed"] == 1
        assert queue.get(event["event_id"])["status"] == "pending"
        assert queue.get(event["event_id"])["attempts"] == 1

        # Simulate the retry delay expiring, then restore receiver connectivity.
        with queue._connect() as conn:
            conn.execute("UPDATE queued_events SET next_attempt_at=NULL WHERE event_id=?", (event["event_id"],))
        state["fail"] = False
        recovered = worker.sync_once()
        assert recovered["synced"] == 1
        assert queue.get(event["event_id"])["status"] == "synced"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_camera_manager_reconnects_live_source(monkeypatch):
    class FakeCapture:
        def __init__(self, *_args):
            self.released = False
        def isOpened(self):
            return True
        def release(self):
            self.released = True
        def get(self, _prop):
            return 0

    monkeypatch.setattr("camera.camera_manager.cv2.VideoCapture", FakeCapture)
    manager = CameraManager(reconnect_interval=30)
    try:
        manager.source = 0
        manager.source_type = "device"
        manager.camera_id = "CAM-TEST"
        manager.room = "Test Room"
        manager.is_connected = False
        manager.health = "offline"
        manager._attempt_reconnect()
        status = manager.get_status()
        assert status["connected"] is True
        assert status["health"] == "healthy"
        assert status["reconnect_attempts"] == 1
    finally:
        manager.shutdown()


def test_camera_manager_detects_missing_heartbeat(monkeypatch):
    import time

    manager = CameraManager(reconnect_interval=0.05)
    reconnect_requests = []
    try:
        manager.source = 0
        manager.source_type = "device"
        manager.camera_id = "CAM-HEARTBEAT"
        manager.room = "Test Room"
        manager.is_connected = True
        manager.last_success_monotonic = time.monotonic() - 10
        manager.heartbeat_timeout_seconds = 0.01
        manager._attempt_reconnect = lambda: reconnect_requests.append(True)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not reconnect_requests:
            time.sleep(0.01)
        status = manager.get_status()
        assert reconnect_requests
        assert status["connected"] is False
        assert status["health"] == "offline"
        assert "heartbeat" in status["last_error"].lower()
    finally:
        manager.shutdown()
