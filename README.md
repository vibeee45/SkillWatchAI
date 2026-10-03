# SkillWatch AI

AI-Based Real-Time Monitoring of Training Centres for Attendance and Infrastructure Compliance.

## Phase 1 — Project Foundation

This repository is the Phase 1 foundation for the SkillWatch AI prototype.

### Core planned capabilities

- Classroom camera integration
- Device camera support
- USB webcam support
- RTSP/IP camera support
- Video-file input
- Face detection
- Face recognition
- Automatic attendance
- Person detection
- Occupancy monitoring
- Infrastructure detection
- Compliance monitoring
- Camera health monitoring
- Offline event processing
- Officer dashboard

### Architecture

Camera
→ Frame Processing
→ AI Detection
→ Attendance / Occupancy / Infrastructure
→ Compliance Engine
→ Alerts
→ FastAPI
→ Dashboard

### Run locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the backend:

```powershell
uvicorn backend.main:app --reload
```

Then open:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

Expected health response:

```json
{"status": "healthy"}
```

## Development order

1. Project Foundation
2. Classroom Camera Integration
3. Frame Processing Pipeline
4. Face Detection
5. Student Enrollment
6. Face Recognition
7. Automatic Attendance
8. Person Detection & Occupancy
9. Infrastructure Detection
10. Compliance & Alert Engine
11. Offline Mode & Camera Reliability
12. FastAPI & PostgreSQL Backend
13. Frontend Dashboard
14. Testing, Accuracy & Privacy
15. Final Integration & SIH Demo

## Current status

Phase 1 foundation only. AI and camera implementations are intentionally left for their respective phases.


## Phase 3
Phase 3 adds frame sampling, inference resizing, frame metadata, image-quality checks, FPS/latency metrics, and a common AIProcessor interface. See PHASE_3_COMPLETE_CHECKLIST.md.

## Phase 4 — Face Detection

Phase 4 adds local YuNet face detection, confidence scores, bounding boxes, face-quality checks, multiple-face support, and lightweight IoU tracking. Face detection is local and does not perform identity recognition.


## Phase 5 — Student Enrollment

Student enrollment is implemented with local SQLite metadata, YuNet quality validation, OpenCV SFace representations, and encrypted-at-rest face representations. Raw enrollment frames are not persisted by the enrollment service. Recognition and attendance matching are intentionally deferred to a later phase.

Download the SFace model:

```powershell
python tools/download_sface_model.py
```

Run the backend:

```powershell
uvicorn backend.main:app --reload
```

Then open the dashboard and use **Student Enrollment**. Connect a camera first, enter Student ID/name/batch, confirm consented/staged demo use, capture 3 valid samples, and save.
