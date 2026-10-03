import threading
import time
from datetime import datetime, timezone
from typing import Optional

import cv2

from camera.frame_output import FramePacket


class CameraManager:
    """Unified camera manager with health monitoring and automatic reconnection."""

    LIVE_SOURCES = {"device", "rtsp"}

    def __init__(self, reconnect_interval: float = 3.0, max_reconnect_attempts: int = 0):
        self.capture: Optional[cv2.VideoCapture] = None
        self._latest_frame = None
        self.source = None
        self.source_type = None
        self.camera_id = None
        self.room = None
        self.is_connected = False
        self.health = "disconnected"
        self.last_frame_at: Optional[str] = None
        self.last_success_monotonic: Optional[float] = None
        self.frame_count = 0
        self._fps_window_start = time.monotonic()
        self._fps_window_frames = 0
        self.actual_fps = 0.0
        self.consecutive_failures = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = max_reconnect_attempts
        self.last_error: Optional[str] = None
        self.reconnect_interval = reconnect_interval
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._monitor = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor.start()

    def _open_device(self, index: int):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        return capture

    def _open_rtsp(self, url: str):
        # Use FFmpeg timeout parameters when supported; fall back to normal OpenCV.
        try:
            capture = cv2.VideoCapture(
                url,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
            )
            if capture.isOpened():
                return capture
            capture.release()
        except Exception:
            pass
        return cv2.VideoCapture(url)

    def _set_capture(self, capture, source, source_type, camera_id, room) -> bool:
        with self._lock:
            if not capture or not capture.isOpened():
                if capture:
                    capture.release()
                self.capture = None
                self.is_connected = False
                self.health = "error"
                self.last_error = "Could not open camera source."
                return False
            self.capture = capture
            self.source = source
            self.source_type = source_type
            self.camera_id = camera_id
            self.room = room
            self.is_connected = True
            self.health = "healthy"
            self.last_error = None
            self.consecutive_failures = 0
            self.reconnect_attempts = 0
            self.frame_count = 0
            self.actual_fps = 0.0
            self._fps_window_start = time.monotonic()
            self._fps_window_frames = 0
            return True

    def connect_device_camera(self, camera_index: int = 0, camera_id: str = "CAM-001", room: str = "Training Room 01") -> bool:
        self.disconnect(clear_config=False)
        return self._set_capture(self._open_device(camera_index), camera_index, "device", camera_id, room)

    def connect_rtsp(self, rtsp_url: str, camera_id: str = "CAM-001", room: str = "Training Room 01") -> bool:
        self.disconnect(clear_config=False)
        if not rtsp_url:
            return False
        return self._set_capture(self._open_rtsp(rtsp_url), rtsp_url, "rtsp", camera_id, room)

    def connect_video_file(self, file_path: str, camera_id: str = "CAM-001", room: str = "Training Room 01") -> bool:
        self.disconnect(clear_config=False)
        if not file_path:
            return False
        return self._set_capture(cv2.VideoCapture(file_path), file_path, "file", camera_id, room)

    def _mark_frame_success(self):
        now = time.monotonic()
        self.last_success_monotonic = now
        self.last_frame_at = datetime.now(timezone.utc).isoformat()
        self.frame_count += 1
        self._fps_window_frames += 1
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self.actual_fps = self._fps_window_frames / elapsed
            self._fps_window_start = now
            self._fps_window_frames = 0
        self.consecutive_failures = 0
        self.health = "healthy"
        self.last_error = None

    def _mark_frame_failure(self, error: str):
        self.consecutive_failures += 1
        self.health = "degraded" if self.consecutive_failures < 3 else "offline"
        self.last_error = error
        if self.consecutive_failures >= 3 and self.source_type in self.LIVE_SOURCES:
            self.is_connected = False
            if self.capture:
                self.capture.release()
                self.capture = None

    def read_frame(self):
        with self._lock:
            if not self.capture or not self.is_connected:
                return False, None

            success, frame = self.capture.read()

            if not success:
                self._mark_frame_failure("Frame read failed.")
                return False, None

            self._mark_frame_success()

        # Store a copy of the latest successful frame.
        # Other consumers such as enrollment must use this cached frame
        # instead of calling VideoCapture.read() again.
            self._latest_frame = frame.copy()

            return True, frame
    def get_latest_frame_packet(self):
        """
    Return the most recently captured frame.

    This does NOT call VideoCapture.read().
    The live preview owns the camera read loop.
    """

        with self._lock:
            if not self.is_connected or self._latest_frame is None:
                return False, None

            frame = self._latest_frame.copy()

            height, width = frame.shape[:2]

            packet = FramePacket(
            frame=frame,
            camera_id=self.camera_id,
            room=self.room,
            source_type=self.source_type,
            timestamp=self.last_frame_at or FramePacket.now_iso(),
            frame_number=self.frame_count,
            width=width,
            height=height,
            fps=self.actual_fps or self.get_fps(),
        )

            return True, packet
    def read_frame_packet(self) -> tuple[bool, Optional[FramePacket]]:
        success, frame = self.read_frame()
        if not success or frame is None:
            return False, None
        height, width = frame.shape[:2]
        return True, FramePacket(
            frame=frame,
            camera_id=self.camera_id,
            room=self.room,
            source_type=self.source_type,
            timestamp=self.last_frame_at or FramePacket.now_iso(),
            frame_number=self.frame_count,
            width=width,
            height=height,
            fps=self.actual_fps or self.get_fps(),
        )

    def _attempt_reconnect(self):
        with self._lock:
            if self.source_type not in self.LIVE_SOURCES or not self.source:
                return
            if self.max_reconnect_attempts and self.reconnect_attempts >= self.max_reconnect_attempts:
                return
            source = self.source
            source_type = self.source_type
            camera_id = self.camera_id
            room = self.room
            self.reconnect_attempts += 1
            attempt = self.reconnect_attempts
        try:
            capture = self._open_device(int(source)) if source_type == "device" else self._open_rtsp(str(source))
            if self._set_capture(capture, source, source_type, camera_id, room):
                self.last_error = None
            else:
                self.last_error = f"Reconnect attempt {attempt} failed."
        except Exception as exc:
            self.last_error = f"Reconnect error: {exc}"

    def _monitor_loop(self):
        while not self._stop_event.wait(self.reconnect_interval):
            with self._lock:
                should_reconnect = self.source_type in self.LIVE_SOURCES and not self.is_connected and self.source is not None
            if should_reconnect:
                self._attempt_reconnect()

    def get_status(self) -> dict:
        with self._lock:
            width, height = self.get_resolution()
            configured_fps = self.get_fps()
            return {
                "connected": self.is_connected,
                "health": self.health,
                "camera_id": self.camera_id,
                "room": self.room,
                "source_type": self.source_type,
                "source": str(self.source) if self.source is not None else None,
                "fps": configured_fps,
                "actual_fps": round(self.actual_fps, 2),
                "width": width,
                "height": height,
                "frame_count": self.frame_count,
                "last_frame_at": self.last_frame_at,
                "consecutive_failures": self.consecutive_failures,
                "reconnect_attempts": self.reconnect_attempts,
                "last_error": self.last_error,
            }

    def get_fps(self) -> float:
        if not self.capture:
            return 0.0
        return float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)

    def get_resolution(self):
        if not self.capture:
            return 0, 0
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def disconnect(self, clear_config: bool = True):
        with self._lock:
            if self.capture:
                self.capture.release()
            self.capture = None
            self._latest_frame = None
            self.is_connected = False
            self.health = "disconnected"
            self.last_error = None
            self.consecutive_failures = 0
            if clear_config:
                self.source = None
                self.source_type = None
                self.camera_id = None
                self.room = None

    def shutdown(self):
        self._stop_event.set()
        self.disconnect()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
