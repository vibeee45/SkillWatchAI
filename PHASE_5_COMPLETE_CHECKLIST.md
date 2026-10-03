# Phase 5 — Student Enrollment Checklist

- [x] 5.1 Create student database model
- [x] 5.2 Create student ID/name/batch fields
- [x] 5.3 Build Add Student UI
- [x] 5.4 Capture consented/staged demo face samples
- [x] 5.5 Generate protected face representations
- [x] 5.6 Store face representations against student IDs
- [x] 5.7 Create student list/search
- [x] 5.8 Add/update/remove enrollment workflow
- [x] 5.9 Validate enrollment quality

## Privacy / prototype boundary

- Consent/demo confirmation is required in the UI.
- Only encrypted face representations are persisted; raw enrollment frames are not stored.
- Enrollment requires exactly one detected face and acceptable quality.
- Recognition/attendance decisions are not part of this phase.

## Manual validation

Run:

```powershell
python tools/download_sface_model.py
python -m tools.test_enrollment
```

Capture 3 valid samples and verify the student appears in the dashboard.
