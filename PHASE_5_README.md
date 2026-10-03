# Phase 5 — Student Enrollment

Phase 5 adds a local enrollment workflow for consented/staged demo students.

Implemented:
- Student metadata stored in local SQLite.
- Add/update/remove/search student workflow.
- Capture of 3 face samples from the active camera.
- YuNet validates exactly one face and sample quality.
- OpenCV SFace generates a normalized face representation.
- Samples are aggregated into one representation.
- Representation is encrypted at rest with Fernet.
- Raw enrollment images are not stored by the enrollment service.

This phase does **not** perform attendance recognition yet. Recognition/matching is a later phase.

## Models

Download SFace after YuNet is already present:

```powershell
python tools/download_sface_model.py
```

## Test enrollment without the web UI

```powershell
python -m tools.test_enrollment
```

Press `C` three times with one face clearly visible, then `Q`.
