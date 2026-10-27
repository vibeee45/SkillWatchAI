import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from camera.camera_manager import CameraManager
from camera.camera_store import CameraStore
from processing.frame_pipeline import FramePipeline
from face_attendance.face_detector import FaceDetector
from face_attendance.processor import FaceDetectionProcessor
from face_attendance.enrollment import EnrollmentService
from face_attendance.protected_store import ProtectedRepresentationStore
from face_attendance.recognizer import SFaceRecognizer
from face_attendance.recognition import FaceRecognitionService
from face_attendance.student_store import StudentStore
from face_attendance.attendance import AttendanceStore
from detection.person_detector import LightweightPersonDetector
from detection.occupancy import OccupancyService
from detection.infrastructure import InfrastructureDetector
from compliance.rules import InfrastructureObservationService
from compliance.engine import ComplianceEngine
from offline.event_queue import SQLiteEventQueue
from offline.sync_worker import EventSyncWorker
from database import init_db
import json
import io
import base64
from datetime import datetime, timezone
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
# Load optional local-only settings (not included in the distributable ZIP).
load_dotenv(BASE_DIR / ".env", override=False)
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "camera" / "uploaded_videos"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SkillWatch AI", description="AI-based real-time monitoring of training centres", version="0.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSETS_DIR = FRONTEND_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.get("/logo.png")
def get_logo():
    logo_file = ASSETS_DIR / "logo.png"
    if not logo_file.exists():
        logo_file = FRONTEND_DIR / "logo.png"
    if logo_file.exists():
        return FileResponse(logo_file, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")


@app.get("/favicon.ico")
def get_favicon():
    fav_file = ASSETS_DIR / "favicon.png"
    if not fav_file.exists():
        fav_file = FRONTEND_DIR / "favicon.png"
    if fav_file.exists():
        return FileResponse(fav_file, media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon not found")
camera_manager = CameraManager()
camera_store = CameraStore(DATA_DIR / "cameras.json")
camera_lock = threading.Lock()
face_detector = FaceDetector(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx", confidence_threshold=0.6)
face_processor = FaceDetectionProcessor(face_detector)
frame_pipeline = FramePipeline(
    sample_every_n=int(os.getenv("SKILLWATCH_PIPELINE_SAMPLE_EVERY_N", "2")),
    target_width=640,
    target_height=360,
    ai_processor=face_processor,
)

student_store = StudentStore(DATA_DIR / "students.db")
representation_store = ProtectedRepresentationStore(DATA_DIR / ".face_key")
sface_recognizer = SFaceRecognizer(BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx")
enrollment_service = EnrollmentService(face_detector, sface_recognizer)
attendance_store = AttendanceStore(DATA_DIR / "attendance.db")
person_detector = LightweightPersonDetector(
    BASE_DIR / "models" / "yolov8n.onnx",
    confidence_threshold=float(
        os.getenv("SKILLWATCH_PERSON_CONFIDENCE", "0.25")
    ),
    nms_threshold=float(
        os.getenv("SKILLWATCH_PERSON_NMS", "0.45")
    ),
)
occupancy_service = OccupancyService(history_size=int(os.getenv("SKILLWATCH_OCCUPANCY_SMOOTHING_FRAMES", "5")))
infrastructure_detector = InfrastructureDetector(
    BASE_DIR / "models" / "yolov8n.onnx",
    confidence_threshold=float(os.getenv("SKILLWATCH_INFRA_CONFIDENCE", "0.15")),
    nms_threshold=float(os.getenv("SKILLWATCH_INFRA_NMS", "0.45")),
)
infrastructure_observations = InfrastructureObservationService()
infrastructure_config_path = DATA_DIR / "infrastructure_config.json"
compliance_engine = ComplianceEngine(DATA_DIR / "compliance_alerts.db", DATA_DIR / "compliance_config.json")

# Events are always persisted locally first. Remote delivery is opt-in through
# SKILLWATCH_SYNC_URL; no student/compliance data leaves this machine by default.
event_queue = SQLiteEventQueue(DATA_DIR / "offline_events.db")
event_sync_worker = EventSyncWorker(event_queue)
event_sync_worker.start()
_last_camera_health_signature: tuple | None = None


def _queue_event(event_type: str, payload: dict[str, Any], event_id: str | None = None) -> None:
    """Best-effort local event persistence; never interrupt core AI processing."""
    try:
        event_queue.enqueue(event_type, payload, event_id=event_id)
    except Exception:
        logging.getLogger("skillwatch.offline").exception("Could not persist %s event", event_type)


def _record_camera_health_transition(status: dict) -> None:
    """Queue only camera health transitions, not every 3-second status poll."""
    global _last_camera_health_signature
    quality = status.get("frame_quality") or {}
    signature = (
        status.get("camera_id"), bool(status.get("connected")), status.get("health"),
        quality.get("state"),
    )
    if signature == _last_camera_health_signature:
        return
    _last_camera_health_signature = signature
    _queue_event("camera_health", {
        "camera_id": status.get("camera_id"),
        "room": status.get("room"),
        "connected": bool(status.get("connected")),
        "health": status.get("health"),
        "frame_age_seconds": status.get("frame_age_seconds"),
        "frame_quality": quality,
        "reconnect_attempts": status.get("reconnect_attempts", 0),
        "last_error": status.get("last_error"),
    })

recognition_service = FaceRecognitionService(
    student_store=student_store,
    representation_store=representation_store,
    recognizer=sface_recognizer,
    log_path=DATA_DIR / "recognition_log.jsonl",
    threshold=float(os.getenv("SKILLWATCH_RECOGNITION_THRESHOLD", "0.38")),
    confirmation_frames=int(os.getenv("SKILLWATCH_RECOGNITION_CONFIRMATION_FRAMES", "2")),
    switch_confirmation_frames=int(os.getenv("SKILLWATCH_RECOGNITION_SWITCH_FRAMES", "2")),
    switch_margin=float(os.getenv("SKILLWATCH_RECOGNITION_SWITCH_MARGIN", "0.05")),
)
enrollment_sessions: dict[str, dict] = {}
enrollment_lock = threading.Lock()


class AttendanceSessionRequest(BaseModel):
    classroom_id: str
    batch_id: str
    session_date: str
    start_time: str
    end_time: str
    camera_id: str


class ClassroomRequest(BaseModel):
    classroom_id: str
    name: str
    room: str


class BatchRequest(BaseModel):
    batch_id: str
    name: str


class AttendanceCorrectionRequest(BaseModel):
    session_id: int
    student_id: str
    status: str
    reason: str = ""
    actor: str = "admin"


class OccupancyZoneRequest(BaseModel):
    zones: list[dict]


class InfrastructureZonesRequest(BaseModel):
    zones: list[dict]


class ComplianceResolveRequest(BaseModel):
    reason: str = "Manually resolved by operator"


class CameraConnectRequest(BaseModel):
    source_type: str
    camera_id: str = "CAM-001"
    room: str = "Training Room 01"
    camera_index: int = 0
    rtsp_url: str | None = None
    file_path: str | None = None


def _config_from_status(status: dict) -> dict:
    return {
        "camera_id": status["camera_id"],
        "room": status["room"],
        "source_type": status["source_type"],
        "source": status["source"],
        "camera_index": int(status["source"]) if status["source_type"] == "device" and str(status["source"]).isdigit() else None,
        "status": "connected" if status["connected"] else "disconnected",
        "width": status["width"],
        "height": status["height"],
        "fps": status["fps"],
    }


def _load_infrastructure_config() -> None:
    """Restore zone geometry only; expected counts and inventory are not used."""
    if not infrastructure_config_path.exists():
        return
    try:
        config = json.loads(infrastructure_config_path.read_text(encoding="utf-8"))
        infrastructure_observations.set_zones(config.get("zones", []))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logging.getLogger("skillwatch.infrastructure").warning(
            "Could not load infrastructure zones: %s", exc
        )


def _save_infrastructure_config() -> None:
    """Persist zone geometry only; never save equipment quantities."""
    infrastructure_config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = infrastructure_config_path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"zones": infrastructure_observations.get_zones()}, indent=2), encoding="utf-8")
    temporary.replace(infrastructure_config_path)


_load_infrastructure_config()

try:
    init_db()
except Exception as _db_init_err:
    logging.getLogger("skillwatch.database").warning("Relational schema initialization skipped: %s", _db_init_err)


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/dashboard/summary")
def dashboard_summary():
    """Unified executive overview for dashboard and monitoring officers."""
    cam_status = camera_manager.get_status()
    _record_camera_health_transition(cam_status)
    occ_status = occupancy_service.status()
    infra_status = infrastructure_observations.status()
    active_session = attendance_store.active_session_for_camera(cam_status.get("camera_id") or "")
    att_summary = attendance_store.summary(active_session["id"], student_store.list()) if active_session else None
    alerts_active = compliance_engine.store.list_alerts("ACTIVE", limit=100)
    queue_status = event_queue.status()
    sync_status = event_sync_worker.status()

    critical_alerts = [a for a in alerts_active if str(a.get("severity", "")).lower() == "high"]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera": cam_status,
        "occupancy": {
            "total": occ_status.get("smoothed_occupancy", 0),
            "raw": occ_status.get("raw_occupancy", 0),
            "recognized": occ_status.get("recognized_students", 0),
            "unknown": occ_status.get("unknown_persons", 0),
            "active_tracks": occ_status.get("active_tracks", 0),
        },
        "attendance": {
            "has_active_session": active_session is not None,
            "session": active_session,
            "summary": att_summary or {
                "present_count": 0,
                "absent_count": 0,
                "total_students": len(student_store.list()),
                "attendance_percentage": 0.0,
            },
        },
        "infrastructure": {
            "detected_counts": infra_status.get("counts_by_class") or infra_status.get("current_counts") or {},
            "detection_valid": infra_status.get("detection_valid", False),
            "detection_state": infra_status.get("detection_state", "UNKNOWN"),
        },
        "compliance": {
            "active_alerts_count": len(alerts_active),
            "critical_alerts_count": len(critical_alerts),
            "recent_alerts": alerts_active[:5],
        },
        "offline": {
            "pending_events": queue_status.get("pending", 0),
            "synced_events": queue_status.get("synced", 0),
            "sync_configured": sync_status.get("configured", False),
        },
        "pipeline": frame_pipeline.get_status().get("stats", {}),
    }


@app.get("/api/camera/status")
def camera_status():
    status = camera_manager.get_status()
    _record_camera_health_transition(status)
    return status


@app.get("/api/offline/status")
def offline_status():
    return {
        "queue": event_queue.status(),
        "sync": event_sync_worker.status(),
        "local_ai_processing": "enabled",
        "note": "Events are stored locally. Remote synchronization is opt-in via SKILLWATCH_SYNC_URL.",
    }


@app.post("/api/offline/sync")
def offline_sync_now():
    """Manually attempt delivery of due queued events to the configured receiver."""
    return event_sync_worker.sync_once()


@app.get("/api/face/status")
def face_status():
    return face_detector.status()


@app.post("/api/face/reset")
def face_reset():
    face_detector.reset()
    return {"message": "Face detector metrics reset.", "status": face_detector.status()}


@app.get("/api/pipeline/status")
def pipeline_status():
    """Return Phase 3 frame-processing metrics and configuration."""
    return frame_pipeline.get_status()


@app.post("/api/pipeline/reset")
def pipeline_reset():
    frame_pipeline.reset()
    return {"message": "Frame pipeline metrics reset.", "status": frame_pipeline.get_status()}


@app.get("/api/enrollment/status")
def enrollment_status():
    return {
        "model": sface_recognizer.status(),
        "active_sessions": len(enrollment_sessions),
        "sample_target": 3,
    }


@app.get("/api/students")
def list_students(query: str = ""):
    return {"students": student_store.list(query)}


@app.get("/api/students/{student_id}")
def get_student(student_id: str):
    student = student_store.get(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
    return student


class EnrollmentStartRequest(BaseModel):
    student_id: str
    name: str
    batch: str
    consent_confirmed: bool = False


class RecognitionConfigRequest(BaseModel):
    threshold: float | None = None
    confirmation_frames: int | None = None
    switch_confirmation_frames: int | None = None
    switch_margin: float | None = None


@app.post("/api/students/enrollment/start")
def start_enrollment(request: EnrollmentStartRequest):
    student_id = request.student_id.strip()
    name = request.name.strip()
    batch = request.batch.strip()
    if not student_id or not name or not batch:
        raise HTTPException(status_code=400, detail="Student ID, name and batch are required.")
    if not request.consent_confirmed:
        raise HTTPException(status_code=400, detail="Consent/demo authorization must be confirmed.")
    with enrollment_lock:
        enrollment_sessions[student_id] = {
            "student_id": student_id,
            "name": name,
            "batch": batch,
            "consent_confirmed": True,
            "samples": [],
        }
    return {"message": "Enrollment session started.", "student_id": student_id, "sample_count": 0, "target": 3}


@app.post("/api/students/enrollment/capture")
def capture_enrollment_sample(student_id: str):
    with enrollment_lock:
        session = enrollment_sessions.get(student_id)
    if not session:
        raise HTTPException(status_code=404, detail="Start an enrollment session first.")
    if not camera_manager.is_connected:
        raise HTTPException(status_code=409, detail="Connect a camera before capturing enrollment samples.")

    # Do not call VideoCapture.read() here: the live MJPEG preview already owns
    # the camera read loop. Consume its latest cached frame instead so enrollment
    # cannot race the preview and turn the camera region black.
    success, packet = camera_manager.get_latest_frame_packet()
    if not success or packet is None:
        raise HTTPException(status_code=503, detail="Waiting for a live camera frame. Keep the preview connected and try again.")
    try:
        sample = enrollment_service.process_frame(packet.frame)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with enrollment_lock:
        session["samples"].append(sample)
        count = len(session["samples"])
    return {
        "message": f"Valid sample {count}/3 captured.",
        "student_id": student_id,
        "sample_count": count,
        "target": 3,
        "quality": {
            "confidence": round(sample.confidence, 3),
            "brightness": round(sample.brightness, 2),
            "blur_score": round(sample.blur_score, 2),
        },
        "sface_inference_ms": round(sface_recognizer.last_inference_ms, 2),
    }


@app.post("/api/students/enrollment/finalize")
def finalize_enrollment(student_id: str):
    with enrollment_lock:
        session = enrollment_sessions.get(student_id)
        if not session:
            raise HTTPException(status_code=404, detail="Enrollment session not found.")
        samples = list(session["samples"])
        if len(samples) < 3:
            raise HTTPException(status_code=422, detail=f"Capture 3 valid samples before saving. Current: {len(samples)}/3.")
        aggregate = sface_recognizer.aggregate([sample.feature for sample in samples])
        protected = representation_store.protect(aggregate)
        quality_score = enrollment_service.quality_score(samples)
        student = student_store.upsert(
            student_id=session["student_id"],
            name=session["name"],
            batch=session["batch"],
            representation=protected,
            sample_count=len(samples),
            quality_score=quality_score,
            consent_confirmed=session["consent_confirmed"],
        )
        enrollment_sessions.pop(student_id, None)
    return {"message": "Student enrollment saved.", "student": student}


class QuickAddStudentRequest(BaseModel):
    student_id: str
    name: str
    batch: str = "Batch A - 2026"
    consent_confirmed: bool = True


@app.post("/api/students/quick-add")
def quick_add_student(request: QuickAddStudentRequest):
    student_id = request.student_id.strip()
    name = request.name.strip()
    batch = request.batch.strip()
    if not student_id or not name or not batch:
        raise HTTPException(status_code=400, detail="Student ID, name and batch are required.")
    if not request.consent_confirmed:
        raise HTTPException(status_code=400, detail="Consent authorization must be confirmed.")
    if not camera_manager.is_connected:
        raise HTTPException(status_code=409, detail="Connect camera before enrolling a student.")

    success, packet = camera_manager.get_latest_frame_packet()
    if not success or packet is None:
        raise HTTPException(status_code=503, detail="No live camera frame available. Ensure camera is streaming.")

    try:
        sample = enrollment_service.process_frame(packet.frame)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Face capture failed: {exc}. Please face the camera clearly.") from exc

    samples = [sample]
    try:
        flipped = cv2.flip(packet.frame, 1)
        samples.append(enrollment_service.process_frame(flipped))
    except Exception:
        pass
    try:
        brightened = cv2.convertScaleAbs(packet.frame, alpha=1.05, beta=5)
        samples.append(enrollment_service.process_frame(brightened))
    except Exception:
        pass

    while len(samples) < 3:
        samples.append(sample)

    aggregate = sface_recognizer.aggregate([s.feature for s in samples])
    protected = representation_store.protect(aggregate)
    quality_score = enrollment_service.quality_score(samples)

    student = student_store.upsert(
        student_id=student_id,
        name=name,
        batch=batch,
        representation=protected,
        sample_count=len(samples),
        quality_score=quality_score,
        consent_confirmed=request.consent_confirmed,
    )
    return {
        "message": f"Student {name} ({student_id}) enrolled successfully!",
        "student": student,
        "sample_count": len(samples),
        "quality_score": round(quality_score, 3),
    }


@app.post("/api/students/enrollment/cancel")
def cancel_enrollment(student_id: str):
    with enrollment_lock:
        removed = enrollment_sessions.pop(student_id, None) is not None
    return {"message": "Enrollment session cancelled.", "removed": removed}


@app.delete("/api/students/{student_id}")
def delete_student(student_id: str):
    with enrollment_lock:
        enrollment_sessions.pop(student_id, None)
    if not student_store.delete(student_id):
        raise HTTPException(status_code=404, detail="Student not found.")
    return {"message": "Student enrollment removed."}


@app.get("/api/recognition/status")
def recognition_status():
    return recognition_service.status()


@app.post("/api/recognition/config")
def configure_recognition(request: RecognitionConfigRequest):
    try:
        return {"message": "Recognition configuration updated.", "status": recognition_service.configure(
            threshold=request.threshold,
            confirmation_frames=request.confirmation_frames,
            switch_confirmation_frames=request.switch_confirmation_frames,
            switch_margin=request.switch_margin,
        )}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/recognition/reset")
def reset_recognition():
    recognition_service.reset()
    return {"message": "Recognition state reset.", "status": recognition_service.status()}


@app.get("/api/recognition/logs")
def recognition_logs(limit: int = 50):
    return {"logs": recognition_service.logger.recent(limit)}


@app.post("/api/recognition/test")
def recognition_test():
    if not camera_manager.is_connected:
        raise HTTPException(status_code=409, detail="Connect a camera first.")
    success, packet = camera_manager.get_latest_frame_packet()
    if not success or packet is None:
        raise HTTPException(status_code=503, detail="No live camera frame available.")
    detections = face_detector.detect(packet.frame)
    try:
        results = recognition_service.recognize(
            packet.frame,
            detections,
            camera_id=packet.camera_id,
            frame_id=f"{packet.camera_id or 'UNKNOWN'}-{packet.frame_number:08d}",
            timestamp=packet.timestamp,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Recognition failed: {exc}") from exc
    return {
        "camera_id": packet.camera_id,
        "room": packet.room,
        "frame_number": packet.frame_number,
        "frame_shape": list(packet.frame.shape),
        "face_count": len(detections),
        "face_detections": [
            {
                **detection.as_dict(),
                "model_row_size": int(detection.model_row.size) if detection.model_row is not None else None,
            }
            for detection in detections
        ],
        "results": [result.as_dict() for result in results],
        "recognition_diagnostics": recognition_service.status().get("diagnostics", {}),
        "enrollment_diagnostics": recognition_service.status().get("enrollment_diagnostics", {}),
        "sface_status": sface_recognizer.status(),
    }


@app.get("/api/cameras")
def list_cameras():
    # cameras.json stores configuration, not a live connection. Overlay its
    # status with the manager's current state so it cannot remain falsely
    # "connected" after a disconnect.
    cameras = camera_store.list()
    live = camera_manager.get_status()
    live_camera_id = live.get("camera_id")

    for camera in cameras:
        camera["status"] = (
            "connected"
            if live.get("connected") and live_camera_id == camera.get("camera_id")
            else "disconnected"
        )
        camera["live_health"] = (
            live.get("health", "unknown")
            if live_camera_id == camera.get("camera_id")
            else "disconnected"
        )

    return {"cameras": cameras}


@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str):
    camera = camera_store.get(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera configuration not found.")
    return camera


@app.post("/api/cameras")
def save_camera(config: dict):
    required = ["camera_id", "room", "source_type", "source"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {', '.join(missing)}")
    return {"message": "Camera configuration saved.", "camera": camera_store.upsert(config)}


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: str):
    if not camera_store.delete(camera_id):
        raise HTTPException(status_code=404, detail="Camera configuration not found.")
    return {"message": "Camera configuration deleted."}


@app.post("/api/camera/connect")
def connect_camera(request: CameraConnectRequest):
    with camera_lock:
        if request.source_type in ("local", "browser"):
            connected = camera_manager.connect_browser_stream(request.camera_id, request.room, source_type=request.source_type)
        elif request.source_type in ("classroom", "rtsp"):
            if not request.rtsp_url:
                raise HTTPException(status_code=400, detail="Classroom RTSP / IP camera URL is required.")
            connected = camera_manager.connect_rtsp(request.rtsp_url, request.camera_id, request.room)
        elif request.source_type in ("upload", "file"):
            if not request.file_path:
                raise HTTPException(status_code=400, detail="Video file path is required. Use 'Media Video Upload' to upload a file.")
            connected = camera_manager.connect_video_file(request.file_path, request.camera_id, request.room)
        elif request.source_type == "device":
            connected = camera_manager.connect_device_camera(request.camera_index, request.camera_id, request.room)
            if not connected:
                raise HTTPException(
                    status_code=503,
                    detail="No physical USB camera detected on this server. Select 'Local PC Camera' to stream directly from this device, or 'Media Video Upload'.",
                )
        else:
            raise HTTPException(status_code=400, detail="Unsupported camera source.")
    if not connected:
        raise HTTPException(status_code=503, detail="Could not connect to the selected camera source.")
    status = camera_manager.get_status()
    camera_store.upsert(_config_from_status(status))
    _record_camera_health_transition(status)
    return {"message": "Camera connected successfully.", "status": status}


@app.post("/api/camera/browser-frame")
async def receive_browser_frame(frame: UploadFile = File(...)):
    if not camera_manager.is_connected or camera_manager.source_type not in ("browser", "local"):
        raise HTTPException(status_code=409, detail="Local PC Camera stream is not active.")
    data = await frame.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image frame.")
    camera_manager.feed_browser_frame(img)

    timestamp_iso = datetime.now(timezone.utc).isoformat()
    frame_id = f"{camera_manager.camera_id or 'LOCAL'}-{camera_manager.frame_count:08d}"

    # 1. Face detection & recognition
    face_detections = face_detector.detect(img)
    recognition_results = []
    try:
        recognition_results = recognition_service.recognize(
            img,
            face_detections,
            camera_id=camera_manager.camera_id,
            frame_id=frame_id,
            timestamp=timestamp_iso,
        )
    except Exception as exc:
        logging.getLogger("skillwatch.recognition").warning("Recognition error: %s", exc)

    recognition_dicts = [r.as_dict() for r in recognition_results]

    # 2. Person detection & occupancy (cadence execution for high streaming FPS)
    person_detections = []
    try:
        if camera_manager.frame_count % 3 == 0 or not person_detector.last_detections:
            person_detections = person_detector.detect(img)
        else:
            person_detections = person_detector.last_detections
    except Exception:
        pass

    occupancy_data = {}
    try:
        occupancy_data = occupancy_service.process(
            person_detections,
            recognition_dicts,
            face_detector.last_detections,
            img.shape[1],
            img.shape[0],
        )
    except Exception:
        pass

    # 3. Automatic attendance logging
    active_session = attendance_store.active_session_for_camera(camera_manager.camera_id or "")
    if active_session:
        for r in recognition_results:
            d = r.as_dict()
            if d.get("status") == "confirmed" and d.get("student_id"):
                record, created = attendance_store.mark_present(
                    active_session["id"], d["student_id"], float(d.get("similarity", 0.0)), camera_manager.camera_id
                )
                if created:
                    _queue_event(
                        "attendance_created",
                        {"attendance": record, "session_id": active_session["id"]},
                        event_id=f"attendance:{active_session['id']}:{d['student_id']}",
                    )

    # 4. Construct AI result for annotations
    ai_result = {
        "processor": "face_detection",
        "detections": [d.as_dict() for d in face_detections],
        "recognition": recognition_dicts,
        "occupancy": occupancy_data,
        "infrastructure": infrastructure_observations.status(),
    }
    frame_pipeline.last_ai_result = ai_result

    # 5. Annotate frame with detections and recognition labels
    annotated = _annotate_person_detections(img, ai_result)
    annotated = _annotate_infrastructure_detections(annotated, ai_result)
    annotated = _annotate_face_detections(annotated, ai_result)

    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    b64_image = f"data:image/jpeg;base64,{base64.b64encode(encoded).decode('ascii')}" if ok else None

    return {
        "status": "ok",
        "frames": camera_manager.frame_count,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "annotated_image": b64_image,
        "face_count": len(face_detections),
        "recognition": recognition_dicts,
        "occupancy": {
            "total": occupancy_data.get("smoothed_occupancy", occupancy_data.get("total_occupancy", 0)),
            "recognized": occupancy_data.get("recognized_students", 0),
            "unknown": occupancy_data.get("unknown_persons", 0),
        },
    }


@app.post("/api/camera/connect-upload")
async def connect_uploaded_video(file: UploadFile = File(...), camera_id: str = Form("CAM-001"), room: str = Form("Training Room 01")):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Video file is required.")
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    extension = Path(file.filename).suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported video format. Use: {', '.join(sorted(allowed_extensions))}")
    destination = UPLOAD_DIR / Path(file.filename).name
    with destination.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            output.write(chunk)
    with camera_lock:
        connected = camera_manager.connect_video_file(str(destination), camera_id, room)
    if not connected:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Could not open uploaded video.")
    status = camera_manager.get_status()
    camera_store.upsert(_config_from_status(status))
    _record_camera_health_transition(status)
    return {"message": "Uploaded video connected successfully.", "status": status}


@app.post("/api/camera/disconnect")
def disconnect_camera():
    camera_manager.disconnect()
    status = camera_manager.get_status()
    _record_camera_health_transition(status)
    return {"message": "Camera disconnected.", "status": status}


def _map_face_boxes_to_original(frame, detections):
    """Map boxes from the 640x360 fit canvas back to the original frame."""
    if frame is None or not detections:
        return []
    h, w = frame.shape[:2]
    tw, th = frame_pipeline.target_width, frame_pipeline.target_height
    scale = min(tw / max(w, 1), th / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    mapped = []
    for d in detections:
        x, y, bw, bh = d["bbox"]
        ox = int(round((x - pad_x) / scale))
        oy = int(round((y - pad_y) / scale))
        ow = int(round(bw / scale))
        oh = int(round(bh / scale))
        if ow > 0 and oh > 0:
            mapped.append((ox, oy, ow, oh, d.get("confidence", 0.0), d.get("track_id")))
    return mapped


def _annotate_face_detections(frame, ai_result):
    if not ai_result or ai_result.get("processor") != "face_detection":
        return frame
    output = frame.copy()
    detections = _map_face_boxes_to_original(output, ai_result.get("detections", []))
    recognition_by_track = {
        item.get("track_id"): item
        for item in ai_result.get("recognition", [])
        if item.get("track_id") is not None
    }
    for x, y, w, h, confidence, track_id in detections:
        x = max(0, min(output.shape[1] - 1, x))
        y = max(0, min(output.shape[0] - 1, y))
        x2 = max(x + 1, min(output.shape[1] - 1, x + w))
        y2 = max(y + 1, min(output.shape[0] - 1, y + h))
        recognition = recognition_by_track.get(track_id)
        if not recognition and len(ai_result.get("recognition", [])) == 1 and len(detections) == 1:
            recognition = ai_result.get("recognition", [])[0]
        color = (0, 220, 0)
        if recognition:
            name = recognition.get("name") or "Unknown"
            status = recognition.get("status", "unknown")
            similarity = float(recognition.get("similarity", 0.0))
            if status == "confirmed":
                label = f"{name} | {similarity:.2f} [CONFIRMED]"
                color = (0, 220, 0)
            elif status == "candidate":
                label = f"Confirming: {name} | {similarity:.2f}"
                color = (0, 215, 255)
            else:
                label = f"Unknown | {similarity:.2f}"
                color = (0, 140, 255)
        else:
            label = f"Face {track_id or '-'} | {confidence:.2f}"
            color = (0, 220, 0)
        cv2.rectangle(output, (x, y), (x2, y2), color, 2)
        cv2.putText(output, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(output, f"Faces: {len(detections)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return output



@app.get("/api/occupancy/status")
def occupancy_status():
    return {
        "detector": person_detector.status(),
        "occupancy": occupancy_service.status(),
    }


@app.post("/api/occupancy/reset")
def occupancy_reset():
    person_detector.reset()
    occupancy_service.reset()
    return {"message": "Occupancy state reset.", "status": occupancy_service.status()}


@app.get("/api/occupancy/zones")
def occupancy_zones():
    return {"zones": [z.as_dict() for z in occupancy_service.zones]}


@app.post("/api/occupancy/zones")
def occupancy_zones_update(request: OccupancyZoneRequest):
    try:
        zones = occupancy_service.set_zones(request.zones)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid zone configuration: {exc}") from exc
    return {"zones": zones}


def _map_person_boxes_to_original(frame, detections):
    if frame is None or not detections:
        return []
    h, w = frame.shape[:2]
    tw, th = frame_pipeline.target_width, frame_pipeline.target_height
    scale = min(tw / max(w, 1), th / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    mapped = []
    for d in detections:
        x, y, bw, bh = d["bbox"]
        ox = int(round((x - pad_x) / scale))
        oy = int(round((y - pad_y) / scale))
        ow = int(round(bw / scale))
        oh = int(round(bh / scale))
        mapped.append((ox, oy, ow, oh, d.get("track_id"), d.get("zone")))
    return mapped


def _annotate_infrastructure_detections(frame, ai_result):
    if not ai_result:
        return frame
    infrastructure = ai_result.get("infrastructure") or {}
    detections = infrastructure.get("detections", [])
    if not detections:
        return frame
    output = frame.copy()
    h, w = output.shape[:2]
    tw, th = frame_pipeline.target_width, frame_pipeline.target_height
    scale = min(tw / max(w, 1), th / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    pad_x, pad_y = (tw - nw) // 2, (th - nh) // 2
    for detection in detections:
        x, y, bw, bh = detection.get("bbox", [0, 0, 0, 0])
        x = int(round((x - pad_x) / scale))
        y = int(round((y - pad_y) / scale))
        bw, bh = int(round(bw / scale)), int(round(bh / scale))
        x = max(0, min(w - 1, x)); y = max(0, min(h - 1, y))
        x2 = max(x + 1, min(w - 1, x + bw)); y2 = max(y + 1, min(h - 1, y + bh))
        label = f"{detection.get('class_name', 'equipment')} {float(detection.get('confidence', 0)):.2f}"
        cv2.rectangle(output, (x, y), (x2, y2), (0, 200, 255), 2)
        cv2.putText(output, label, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    return output


def _annotate_person_detections(frame, ai_result):
    if not ai_result:
        return frame
    occupancy = ai_result.get("occupancy") or {}
    detections = occupancy.get("detections", [])
    if not detections:
        return frame
    output = frame.copy()
    for x, y, w, h, track_id, zone in _map_person_boxes_to_original(output, detections):
        x = max(0, min(output.shape[1] - 1, x))
        y = max(0, min(output.shape[0] - 1, y))
        x2 = max(x + 1, min(output.shape[1] - 1, x + w))
        y2 = max(y + 1, min(output.shape[0] - 1, y + h))
        cv2.rectangle(output, (x, y), (x2, y2), (255, 180, 0), 2)
        label = f"Person #{track_id or '-'}" + (f" | {zone}" if zone else "")
        cv2.putText(output, label, (x, min(output.shape[0] - 5, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA)
    cv2.putText(
        output,
        f"Occupancy: {occupancy.get('total_occupancy', 0)} | Recognized: {occupancy.get('recognized_students', 0)} | Unknown: {occupancy.get('unknown_persons', 0)}",
        (10, output.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 0),
        2,
        cv2.LINE_AA,
    )
    return output


@app.get("/api/infrastructure/status")
def infrastructure_status():
    """Return detector diagnostics and observed equipment counts only."""
    return {
        "detector": infrastructure_detector.status(),
        "observation": infrastructure_observations.status(),
    }


@app.post("/api/infrastructure/test")
def infrastructure_test():
    """Run one equipment-detection pass against the most recent live frame."""
    if not camera_manager.is_connected:
        raise HTTPException(status_code=409, detail="Connect a camera first.")
    success, packet = camera_manager.get_latest_frame_packet()
    if not success or packet is None:
        raise HTTPException(status_code=503, detail="No live camera frame available.")
    try:
        detections = infrastructure_detector.detect(packet.frame)
        h, w = packet.frame.shape[:2]
        all_class_max = float(infrastructure_detector.last_all_class_max_score)
        if not detections and all_class_max < float(os.getenv("SKILLWATCH_INFRA_MIN_EVIDENCE_SCORE", "0.05")):
            result = infrastructure_observations.mark_insufficient_evidence(
                f"No confident objects in frame (maximum COCO class score={all_class_max:.4f}). Check the camera preview, framing, lighting, and model input.",
                {"last_frame_stats": infrastructure_detector.last_frame_stats,
                 "last_top_class_scores": infrastructure_detector.last_top_class_scores,
                 "minimum_evidence_score": float(os.getenv("SKILLWATCH_INFRA_MIN_EVIDENCE_SCORE", "0.05"))},
            )
        else:
            result = infrastructure_observations.process(detections, w, h)
    except Exception as exc:
        logging.getLogger("skillwatch.infrastructure").exception("Infrastructure detection failed")
        result = infrastructure_observations.mark_inference_error(str(exc))
        return {
            "camera_id": packet.camera_id,
            "frame_shape": list(packet.frame.shape),
            "detector": infrastructure_detector.status(),
            "result": result,
        }
    return {
        "camera_id": packet.camera_id,
        "frame_shape": list(packet.frame.shape),
        "detector": infrastructure_detector.status(),
        "result": result,
    }


@app.get("/api/infrastructure/debug-frame.jpg")
def infrastructure_debug_frame():
    """Return the exact latest camera frame used by Phase 9 for visual debugging."""
    if not camera_manager.is_connected:
        raise HTTPException(status_code=409, detail="Connect a camera first.")
    success, packet = camera_manager.get_latest_frame_packet()
    if not success or packet is None:
        raise HTTPException(status_code=503, detail="No live camera frame available.")
    ok, encoded = cv2.imencode(".jpg", packet.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode latest camera frame.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/infrastructure/zones")
def infrastructure_zones():
    return {"zones": infrastructure_observations.get_zones()}


@app.put("/api/infrastructure/zones")
def infrastructure_zones_update(request: InfrastructureZonesRequest):
    try:
        zones = infrastructure_observations.set_zones(request.zones)
        _save_infrastructure_config()
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid infrastructure zones: {exc}") from exc
    return {"zones": zones, "message": "Infrastructure zones saved."}


@app.post("/api/infrastructure/reset")
def infrastructure_reset():
    infrastructure_detector.reset()
    infrastructure_observations.reset()
    return {"message": "Infrastructure detection metrics and observation state reset."}


@app.get("/api/attendance/classrooms")
def attendance_classrooms():
    return {"classrooms": attendance_store.list_classrooms()}


@app.post("/api/attendance/classrooms")
def create_attendance_classroom(request: ClassroomRequest):
    return attendance_store.upsert_classroom(request.classroom_id, request.name, request.room)


@app.get("/api/attendance/batches")
def attendance_batches():
    return {"batches": attendance_store.list_batches()}


@app.post("/api/attendance/batches")
def create_attendance_batch(request: BatchRequest):
    return attendance_store.upsert_batch(request.batch_id, request.name)


@app.get("/api/attendance/sessions")
def attendance_sessions():
    return {"sessions": attendance_store.list_sessions()}


@app.post("/api/attendance/sessions")
def create_attendance_session(request: AttendanceSessionRequest):
    attendance_store.upsert_classroom(request.classroom_id, request.classroom_id, request.classroom_id)
    attendance_store.upsert_batch(request.batch_id, request.batch_id)
    return attendance_store.create_session(request.classroom_id, request.batch_id, request.session_date, request.start_time, request.end_time, request.camera_id)


@app.post("/api/attendance/sessions/{session_id}/start")
def start_attendance_session(session_id: int):
    session = attendance_store.start_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Attendance session not found.")
    return session


@app.post("/api/attendance/sessions/{session_id}/end")
def end_attendance_session(session_id: int):
    session = attendance_store.end_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Attendance session not found.")
    return session


@app.get("/api/attendance/sessions/{session_id}/summary")
def attendance_summary(session_id: int):
    result = attendance_store.summary(session_id, student_store.list())
    if not result:
        raise HTTPException(status_code=404, detail="Attendance session not found.")
    return result


@app.get("/api/attendance/sessions/{session_id}/audit")
def attendance_audit(session_id: int):
    if not attendance_store.get_session(session_id):
        raise HTTPException(status_code=404, detail="Attendance session not found.")
    return {"audit": attendance_store.audit(session_id)}


@app.post("/api/attendance/correct")
def correct_attendance(request: AttendanceCorrectionRequest):
    if not attendance_store.get_session(request.session_id):
        raise HTTPException(status_code=404, detail="Attendance session not found.")
    try:
        record = attendance_store.correct(request.session_id, request.student_id, request.status, request.reason, request.actor)
        _queue_event(
            "attendance_corrected",
            {"attendance": record, "reason": request.reason, "actor": request.actor},
            event_id=f"attendance-correction:{request.session_id}:{request.student_id}:{record.get('updated_at', '')}",
        )
        return record
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/attendance/status")
def attendance_status():
    active = attendance_store.active_session_for_camera(camera_manager.get_status().get("camera_id", ""))
    return {"active_session": active}


def _build_compliance_context() -> dict[str, Any]:
    camera = camera_manager.get_status()

    # A manager may have no active identity after startup or explicit
    # disconnect. Recover metadata from the saved configuration without
    # changing the live connected/health fields.
    if not camera.get("camera_id"):
        saved_cameras = camera_store.list()
        if saved_cameras:
            saved_camera = saved_cameras[0]
            camera["camera_id"] = saved_camera.get("camera_id")
            camera["room"] = camera.get("room") or saved_camera.get("room")
            camera["source_type"] = camera.get("source_type") or saved_camera.get("source_type")
            camera["source"] = camera.get("source") or saved_camera.get("source")

    camera_age = None
    last_frame = camera.get("last_frame_at")
    if last_frame:
        try:
            parsed = datetime.fromisoformat(str(last_frame).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            camera_age = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())
        except (ValueError, TypeError):
            camera_age = None
    active = attendance_store.active_session_for_camera(camera.get("camera_id") or "")
    if active is None:
        recent_sessions = attendance_store.list_sessions(limit=1)
        if recent_sessions and str(recent_sessions[0].get("status", "")).upper() == "ENDED":
            active = recent_sessions[0]
    summary = attendance_store.summary(active["id"], student_store.list()) if active else None
    occ = occupancy_service.status()
    infra = infrastructure_observations.status()
    recognition = recognition_service.status().get("last_results", [])
    return {
        "camera": camera,
        "camera_frame_age_seconds": camera_age,
        "active_session": active,
        "attendance_summary": summary or {},
        "occupancy": occ,
        "infrastructure": infra,
        "recognition": recognition,
        "room_id": camera.get("room") or "classroom",
        # Do not resolve sensor-derived alerts based on invalid/missing sensor data.
        "evaluation_valid": bool(camera.get("connected")) and bool(infra.get("detection_valid")) and infra.get("detection_state") == "SUCCESS",
    }


def _evaluate_compliance() -> dict[str, Any]:
    active_before = {item.get("id") for item in compliance_engine.store.list_alerts("ACTIVE", 500)}
    result = compliance_engine.evaluate(_build_compliance_context())
    active_after = {item.get("id") for item in result.get("active_alerts", [])}

    # Queue one event per alert activation (not every repeated video-frame
    # observation). first_seen_at changes when an existing alert is reopened.
    for alert in result.get("new_or_updated_alerts", []):
        alert_id = alert.get("id")
        if alert_id is not None:
            _queue_event(
                "compliance_alert",
                {"alert": alert},
                event_id=f"compliance-alert:{alert_id}:{alert.get('first_seen_at', '')}",
            )

    # Auto-resolutions are transitions too. Queue only IDs that were active
    # before this evaluation and are no longer active afterward.
    resolved_ids = active_before - active_after
    if resolved_ids:
        resolved = {item.get("id"): item for item in compliance_engine.store.list_alerts("RESOLVED", 500)}
        for alert_id in resolved_ids:
            alert = resolved.get(alert_id)
            if alert:
                _queue_event(
                    "compliance_alert_resolved",
                    {"alert": alert},
                    event_id=f"compliance-resolved:{alert_id}:{alert.get('resolved_at', '')}",
                )
    return result


@app.get("/api/compliance/status")
def compliance_status():
    return {
        "configuration": compliance_engine.get_config(),
        "active_alerts": compliance_engine.store.list_alerts("ACTIVE"),
        "recent_alerts": compliance_engine.store.list_alerts(None, 100),
    }


@app.get("/api/compliance/config")
def compliance_config_get():
    return compliance_engine.get_config()


@app.put("/api/compliance/config")
def compliance_config_update(payload: dict[str, Any]):
    try:
        config = compliance_engine.update_config(payload)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Compliance rule configuration saved.", "configuration": config}


@app.post("/api/compliance/evaluate")
def compliance_evaluate():
    return _evaluate_compliance()


@app.get("/api/compliance/alerts")
def compliance_alerts(status: str = "ACTIVE", limit: int = 100):
    normalized = status.upper()
    if normalized not in {"ACTIVE", "RESOLVED", "ALL"}:
        raise HTTPException(status_code=400, detail="status must be ACTIVE, RESOLVED, or ALL")
    return {"alerts": compliance_engine.store.list_alerts(None if normalized == "ALL" else normalized, limit)}


@app.get("/api/compliance/events")
def compliance_events(limit: int = 100):
    return {"events": compliance_engine.store.events(limit)}


@app.post("/api/compliance/alerts/{alert_id}/resolve")
def compliance_alert_resolve(alert_id: int, request: ComplianceResolveRequest):
    result = compliance_engine.store.resolve(alert_id, request.reason)
    if result is None:
        raise HTTPException(status_code=404, detail="Compliance alert not found")
    _queue_event(
        "compliance_alert_resolved",
        {"alert": result, "reason": request.reason},
        event_id=f"compliance-resolved:{alert_id}:{result.get('resolved_at', '')}",
    )
    return {"message": "Alert resolved.", "alert": result}


def frame_generator():
    while True:
        if not camera_manager.is_connected:
            time.sleep(0.1)
            continue
        success, packet = camera_manager.read_frame_packet()
        _record_camera_health_transition(camera_manager.get_status())
        if not success or packet is None:
            time.sleep(0.05)
            continue
        ai_frame = frame_pipeline.process(packet)
        if ai_frame is not None:
            ai_result = frame_pipeline.run_ai(ai_frame)
            recognition_results = []
            try:
                # Phase 8: person detection and tracking on the same standardized
                # 640x360 frame used by the face pipeline.
                person_detections = person_detector.detect(ai_frame.frame)
                recognition_results = recognition_service.recognize(
                    ai_frame.frame,
                    face_detector.last_detections,
                    camera_id=packet.camera_id,
                    frame_id=ai_frame.frame_id,
                    timestamp=ai_frame.timestamp,
                )
                recognition_dicts = [result.as_dict() for result in recognition_results]
                ai_result["recognition"] = recognition_dicts
                ai_result["occupancy"] = occupancy_service.process(
                    person_detections,
                    recognition_dicts,
                    face_detector.last_detections,
                    ai_frame.frame.shape[1],
                    ai_frame.frame.shape[0],
                )
                try:
                    equipment_detections = infrastructure_detector.detect(ai_frame.frame)
                    ai_result["infrastructure"] = infrastructure_observations.process(
                        equipment_detections,
                        ai_frame.frame.shape[1],
                        ai_frame.frame.shape[0],
                    )
                except Exception as infra_exc:
                    logging.getLogger("skillwatch.infrastructure").exception(
                        "Infrastructure inference failed; observation marked unavailable"
                    )
                    ai_result["infrastructure"] = infrastructure_observations.mark_inference_error(str(infra_exc))
                active_session = attendance_store.active_session_for_camera(packet.camera_id)
                if active_session:
                    auto_attendance = []
                    for result in recognition_results:
                        data = result.as_dict()
                        if data.get("status") == "confirmed" and data.get("student_id"):
                            record, created = attendance_store.mark_present(
                                active_session["id"], data["student_id"], float(data.get("similarity", 0.0)), packet.camera_id
                            )
                            if created:
                                auto_attendance.append(record)
                                _queue_event(
                                    "attendance_created",
                                    {"attendance": record, "session_id": active_session["id"]},
                                    event_id=f"attendance:{active_session['id']}:{data['student_id']}",
                                )
                    ai_result["attendance"] = auto_attendance
                else:
                    ai_result["attendance"] = []
                try:
                    ai_result["compliance"] = _evaluate_compliance()
                except Exception as compliance_exc:
                    logging.getLogger("skillwatch.compliance").exception("Compliance evaluation failed")
                    ai_result["compliance"] = {"error": str(compliance_exc)}
            except Exception as exc:
                logging.getLogger("skillwatch.ai").warning("Occupancy/recognition skipped: %s", exc)
                ai_result["recognition"] = [result.as_dict() for result in recognition_results]
                ai_result["occupancy"] = occupancy_service.status()
                ai_result["infrastructure"] = infrastructure_observations.mark_inference_error(str(exc))
                ai_result["ai_error"] = str(exc)
            frame_pipeline.last_ai_result = ai_result
        preview_frame = _annotate_person_detections(packet.frame, frame_pipeline.last_ai_result)
        preview_frame = _annotate_infrastructure_detections(preview_frame, frame_pipeline.last_ai_result)
        preview_frame = _annotate_face_detections(preview_frame, frame_pipeline.last_ai_result)
        ok, encoded = cv2.imencode(".jpg", preview_frame)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        time.sleep(0.04)


@app.get("/api/camera/stream")
def camera_stream():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/latest-frame")
def camera_latest_frame():
    success, packet = camera_manager.get_latest_frame_packet()
    if not success or packet is None:
        raise HTTPException(status_code=503, detail="No camera frame available.")
    preview_frame = _annotate_person_detections(packet.frame, frame_pipeline.last_ai_result)
    preview_frame = _annotate_infrastructure_detections(preview_frame, frame_pipeline.last_ai_result)
    preview_frame = _annotate_face_detections(preview_frame, frame_pipeline.last_ai_result)
    ok, encoded = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame.")
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/camera/uploads/{filename}")
def uploaded_video(filename: str):
    path = UPLOAD_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(path)
