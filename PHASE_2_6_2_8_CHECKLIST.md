# Phase 2.6–2.8 Checklist

- [x] Camera source selection UI
- [x] Laptop/device camera selection
- [x] USB webcam support through camera index
- [x] RTSP/IP/CCTV URL input
- [x] MP4/video upload
- [x] Camera ID and room fields
- [x] Connect control
- [x] Disconnect control
- [x] Live MJPEG preview endpoint
- [x] Status display: connected, source, room, resolution, FPS

## Run

From the SkillWatchAI root:

```powershell
uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```
