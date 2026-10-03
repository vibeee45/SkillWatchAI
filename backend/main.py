import logging
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from camera.camera_manager import CameraManager
from camera.camera_store import CameraStore
from processing.frame_pipeline import FramePipeline
from face_attendance.face_detector import FaceDetector
from face_attendance.processor import FaceDetectionProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "camera" / "uploaded_videos"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SkillWatch AI", description="AI-based real-time monitoring of training centres", version="0.3.0")
camera_manager = CameraManager()
camera_store = CameraStore(DATA_DIR / "cameras.json")
camera_lock = threading.Lock()
face_detector = FaceDetector(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx", confidence_threshold=0.6)
face_processor = FaceDetectionProcessor(face_detector)
frame_pipeline = FramePipeline(sample_every_n=3, target_width=640, target_height=360, ai_processor=face_processor)


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


@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/camera/status")
def camera_status():
    return camera_manager.get_status()


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


@app.get("/api/cameras")
def list_cameras():
    return {"cameras": camera_store.list()}


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
        if request.source_type == "device":
            connected = camera_manager.connect_device_camera(request.camera_index, request.camera_id, request.room)
        elif request.source_type == "rtsp":
            if not request.rtsp_url:
                raise HTTPException(status_code=400, detail="RTSP URL is required.")
            connected = camera_manager.connect_rtsp(request.rtsp_url, request.camera_id, request.room)
        elif request.source_type == "file":
            if not request.file_path:
                raise HTTPException(status_code=400, detail="Video file path is required.")
            connected = camera_manager.connect_video_file(request.file_path, request.camera_id, request.room)
        else:
            raise HTTPException(status_code=400, detail="Unsupported camera source.")
    if not connected:
        raise HTTPException(status_code=503, detail="Could not connect to the selected camera source.")
    status = camera_manager.get_status()
    camera_store.upsert(_config_from_status(status))
    return {"message": "Camera connected successfully.", "status": status}


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
    return {"message": "Uploaded video connected successfully.", "status": status}


@app.post("/api/camera/disconnect")
def disconnect_camera():
    camera_manager.disconnect()
    return {"message": "Camera disconnected.", "status": camera_manager.get_status()}


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
    for x, y, w, h, confidence, track_id in detections:
        x = max(0, min(output.shape[1] - 1, x))
        y = max(0, min(output.shape[0] - 1, y))
        x2 = max(x + 1, min(output.shape[1] - 1, x + w))
        y2 = max(y + 1, min(output.shape[0] - 1, y + h))
        cv2.rectangle(output, (x, y), (x2, y2), (0, 255, 0), 2)
        label = f"Face {track_id or '-'} | {confidence:.2f}"
        cv2.putText(output, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(output, f"Faces: {len(detections)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return output


def frame_generator():
    while True:
        if not camera_manager.is_connected:
            time.sleep(0.1)
            continue
        success, packet = camera_manager.read_frame_packet()
        if not success or packet is None:
            time.sleep(0.05)
            continue
        ai_frame = frame_pipeline.process(packet)
        if ai_frame is not None:
            frame_pipeline.run_ai(ai_frame)
        preview_frame = _annotate_face_detections(packet.frame, frame_pipeline.last_ai_result)
        ok, encoded = cv2.imencode(".jpg", preview_frame)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"


@app.get("/api/camera/stream")
def camera_stream():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/camera/uploads/{filename}")
def uploaded_video(filename: str):
    path = UPLOAD_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(path)
