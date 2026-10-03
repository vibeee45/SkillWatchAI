# Phase 4 — Face Detection

This phase adds a lightweight local YuNet face detector to the standardized Phase 3 AI pipeline.

## Setup

From the project root:

```powershell
python tools/download_yunet_model.py
```

Then start the backend:

```powershell
uvicorn backend.main:app --reload
```

## Test

Open the dashboard and connect the desktop camera or MP4. Detected faces are drawn on the live preview with a track ID and confidence score.

Face detector status:

```text
GET /api/face/status
```

Reset detector metrics:

```text
POST /api/face/reset
```

Standalone detector test:

```powershell
python tools/test_face_detector.py
```

## Thresholds

Default confidence threshold is `0.60`. Face quality checks flag very dark, blurred, or too-small face crops. These are quality flags; they do not identify a person.
