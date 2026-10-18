# SkillWatch AI — Privacy-Preserving Compliance Intelligence

> **Smart India Hackathon (SIH 2026)**  
> **Problem Statement ID:** 26245  
> **Title:** AI-Based Real-Time Monitoring of Training Centres for Attendance and Infrastructure Compliance  
> **Ministry / Organization:** Ministry of Skill Development and Entrepreneurship (MSDE)  
> **Category / Theme:** Smart Education / Software  
> **Tagline:** *"From passive CCTV to privacy-preserving compliance intelligence."*

---

## 1. Overview & Value Proposition

Government-funded skilling schemes depend on geographically distributed training centres. Manual inspections are infrequent, expensive, and unable to detect daily variations in attendance or disappearing infrastructure between visits. At the same time, central video streaming from rural centres is cost-prohibitive, and constant facial surveillance creates severe privacy concerns.

**SkillWatch AI** converts passive camera feeds into structured, privacy-preserving compliance intelligence:
- **Privacy-First:** Aggregate physical room occupancy is separated from student identity. Only consented students are recognized via salted/encrypted local feature representations; raw enrollment images are discarded.
- **Edge-First & Low-Bandwidth:** AI models execute locally on CPU (ONNX Runtime / OpenCV). Raw video is never continuously transmitted; only small event payloads (< 1.8 KB) are queued and synced.
- **Reliable & Camera-Health Aware:** A disconnected, dark, or frozen feed creates a camera maintenance alert, avoiding false fraud allegations.
- **Human-in-the-Loop:** SkillWatch AI does not automatically penalize centres; it prioritizes discrepancies for officer review and verification.

---

## 2. End-to-End Architecture

```
+---------------------------------------------------------------------------------------+
|                                    EDGE DEVICE / CCTV                                  |
|                                                                                       |
|   [Laptop / USB Webcam / RTSP IP Camera / Video File]                                 |
|                            │                                                          |
|                            ▼                                                          |
|                     Camera Manager                                                    |
|            (Reconnect watchdog, Heartbeat, Freeze check)                              |
|                            │                                                          |
|                            ▼                                                          |
|                 Frame Processing Pipeline                                             |
|        (640x360 resize, Laplacian blur check, sampling 1:3)                           |
|                            │                                                          |
|         ┌──────────────────┼─────────────────────────┐                                |
|         ▼                  ▼                         ▼                                |
|    YuNet ONNX         YOLOv8n ONNX             YOLOv8n ONNX                           |
|  Face Detection     Person Detection      Infrastructure Detection                    |
|         │                  │                         │                                |
|         ▼                  ▼                         ▼                                |
|    SFace ONNX         IoU Tracker              Zone Filter                            |
| Face Recognition   5-Frame Smoothing      Temporal Observation                        |
|         │                  │                         │                                |
|         ▼                  ▼                         ▼                                |
|     Automatic         Classroom                  Equipment                            |
|    Attendance         Occupancy               Observed Counts                         |
|         │                  │                         │                                |
|         └──────────────────┼─────────────────────────┘                                |
|                            ▼                                                          |
|                Compliance & Rule Engine                                               |
|      (Attendance Discrepancy, Low Occupancy, Infra Gaps, Health)                      |
|                            │                                                          |
|                            ▼                                                          |
|                SQLite WAL Offline Queue                                               |
|           (Zero data loss during network outages)                                     |
|                            │                                                          |
+────────────────────────────┼──────────────────────────────────────────────────────────+
                             │
                             ▼ HTTP Sync / Local Dashboard
+───────────────────────────────────────────────────────────────────────────────────────+
|                                  OFFICER DASHBOARD                                    |
|                                                                                       |
|  - Real-time Camera Preview (MJPEG)                                                   |
|  - Executive KPI Summary Cards                                                        |
|  - Automatic Attendance & Manual Override Audit Log                                   |
|  - Infrastructure Asset Live Counter & Zone Editor                                    |
|  - Compliance Alerts Panel with One-Click Officer Resolution                          |
+───────────────────────────────────────────────────────────────────────────────────────+
```

---

## 3. 15-Phase Development Status

| Phase | Milestone | Status | Key Deliverable |
| :---: | :--- | :---: | :--- |
| **Phase 1** | Project Setup & Architecture | ✅ Completed | FastAPI, environment configuration, virtual environment |
| **Phase 2** | Classroom Camera Integration | ✅ Completed | Multi-source CameraManager (webcam, RTSP, video file) |
| **Phase 3** | Frame Processing Pipeline | ✅ Completed | Frame sampling, quality checks (blur/brightness), metrics |
| **Phase 4** | Face Detection | ✅ Completed | YuNet ONNX detector, bounding boxes, landmark alignment |
| **Phase 5** | Student Enrollment | ✅ Completed | 3-sample capture, consent checkbox, encrypted embeddings |
| **Phase 6** | Face Recognition | ✅ Completed | SFace ONNX cosine similarity, 3-frame confirmation |
| **Phase 7** | Automatic Attendance Engine | ✅ Completed | Session lifecycle, duplicate prevention, audit history |
| **Phase 8** | Person Detection & Occupancy | ✅ Completed | YOLOv8n ONNX person detector, IoU tracking, zone smoothing |
| **Phase 9** | Infrastructure Detection | ✅ Completed | YOLOv8n ONNX detection of chairs, tables, laptops, TVs |
| **Phase 10** | Compliance & Alert Engine | ✅ Completed | Multi-rule engine with severity scoring and persistence |
| **Phase 11** | Offline Mode & Reliability | ✅ Completed | SQLite WAL queue, retry worker, camera health watchdog |
| **Phase 12** | Backend & Database Models | ✅ Completed | Unified `/api/dashboard/summary`, SQLAlchemy relational schema |
| **Phase 13** | Frontend Dashboard | ✅ Completed | Interactive cards for preview, KPI, attendance, infra, alerts |
| **Phase 14** | Testing & Empirical Benchmarking | ✅ Completed | 55 unit tests passing, `MEASURED_ACCURACY_REPORT.md` |
| **Phase 15** | Demonstration Protocol | ✅ Completed | 7-step evaluation script in `SIH_DEMO_GUIDE.md` |

---

## 4. Key Performance Benchmarks (Measured)

*From empirical tests run on standard CPU without GPU acceleration:*

- **Face Detection Latency:** ~80 ms (YuNet ONNX, 0.22 MB)
- **Person Detection Latency:** ~359 ms (YOLOv8n ONNX, 12.26 MB)
- **Infrastructure Detection Latency:** ~380 ms (YOLOv8n ONNX, 12.26 MB)
- **Compliance Rule Evaluation:** **4.6 ms** per cycle
- **Total Model Footprint:** **49.38 MB**
- **Bandwidth Reduction:** **> 99.4% saved** vs continuous video streaming

For full confusion matrices, latency percentiles, and protocol details, see [MEASURED_ACCURACY_REPORT.md](file:///c:/Users/91958/Desktop/SkillWatchAI/MEASURED_ACCURACY_REPORT.md).

---

## 5. Quickstart & Installation

### Requirements
- Python 3.10+
- Windows, Linux, or macOS

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/vibeee45/SkillWatchAI.git
cd SkillWatchAI
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Download Pretrained Models (if not already in `models/`)
```bash
python tools/download_yunet_model.py
python tools/download_sface_model.py
```

### 3. Launch the Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in any web browser.

---

## 6. Running Tests & Benchmarks

Run the complete 55-test test suite:
```bash
python -m pytest tests/
```

Run the empirical benchmark evaluation:
```bash
python tools/benchmark_evaluation.py
```

---

## 7. API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/dashboard/summary` | `GET` | Executive overview (camera, occupancy, attendance, infra, alerts) |
| `/api/camera/status` | `GET` | Real-time camera health, FPS, frame age, and quality state |
| `/api/camera/connect` | `POST` | Connect device camera, USB webcam, or RTSP URL |
| `/api/camera/stream` | `GET` | Live annotated MJPEG video preview |
| `/api/students/enrollment/start` | `POST` | Begin consented student face enrollment session |
| `/api/attendance/sessions` | `POST` | Create a training session for a room and batch |
| `/api/attendance/correct` | `POST` | Manual attendance override with immutable audit log |
| `/api/occupancy/status` | `GET` | Smoothed total occupancy, recognized vs unknown counts |
| `/api/infrastructure/status` | `GET` | Observed counts of chairs, tables, laptops, and monitors |
| `/api/compliance/alerts` | `GET` | Query active or resolved compliance alerts |
| `/api/compliance/alerts/{id}/resolve` | `POST` | Officer resolution action with recorded reason |
| `/api/offline/status` | `GET` | SQLite offline queue pending, synced, and failed metrics |

---

## 8. Live Demonstration Guide

To replicate the official SIH live presentation workflow:
👉 Follow the step-by-step instructions in [SIH_DEMO_GUIDE.md](file:///c:/Users/91958/Desktop/SkillWatchAI/SIH_DEMO_GUIDE.md).
