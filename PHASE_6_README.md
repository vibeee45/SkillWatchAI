# SkillWatch AI — Phase 6 Face Recognition

Implemented checklist:

- 6.1 Extract SFace representation from each detected YuNet face.
- 6.2 Compare against protected enrolled student representations using cosine similarity.
- 6.3 Runtime recognition threshold configuration (default 0.45).
- 6.4 Identify enrolled students.
- 6.5 Low-confidence matches are returned as `unknown`.
- 6.6 Maintain identity state per YuNet track ID.
- 6.7 Require temporal confirmation (default 3 consistent frames).
- 6.8 Require repeated, margin-qualified evidence before switching a stable identity.
- 6.9 Append timestamp, camera/frame, track, student, similarity/confidence and status to `data/recognition_log.jsonl`.
- 6.10 Test endpoint and automated unit tests for known/unknown/temporal behaviour.

## Recognition endpoints

- `GET /api/recognition/status`
- `POST /api/recognition/config`
- `POST /api/recognition/test`
- `POST /api/recognition/reset`
- `GET /api/recognition/logs?limit=50`

## Default configuration

- Threshold: `0.45`
- Temporal confirmation: `3` frames
- Identity-switch confirmation: `3` frames
- Switch margin: `0.05`

## Models

The ZIP intentionally does not bundle ONNX model binaries. Run:

```powershell
python tools/download_yunet_model.py
python tools/download_sface_model.py
```

Then start:

```powershell
uvicorn backend.main:app --reload
```

## Manual validation

1. Connect camera.
2. Enroll a consented/staged demo student using 3 samples.
3. Keep the enrolled student in front of the camera.
4. Wait for `candidate` for the first frames, then `confirmed`.
5. Test an unknown face; it should remain `unknown` when similarity is below threshold.
6. Check `GET /api/recognition/logs` or `data/recognition_log.jsonl`.
