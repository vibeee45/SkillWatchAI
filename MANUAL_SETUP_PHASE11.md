# SkillWatchAI Phase 11 — Manual Setup Checklist (Windows)

## 1. Install/run locally

Open PowerShell in the extracted `SkillWatchAI` folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Local configuration

The ZIP intentionally does not include `.env` because it can contain private tokens. Create one from the example:

```powershell
Copy-Item .\.env.example .\.env
```

If you already have a working `.env`, keep it and merge only the Phase 11 variables from `.env.example`. Remote synchronization remains disabled unless `SKILLWATCH_SYNC_URL` is set.

## 3. Start the application

```powershell
.\run_backend.ps1
```

Or run directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000` in a browser. The local dashboard and local AI processing do not need internet access after Python packages/models are installed.

## 4. Verify Phase 11

```powershell
python -m pytest .\tests -v
```

The prepared package was tested with 43 passing tests, including Phase 10 regression tests. The backend smoke test also returned HTTP 200 for `/health`, `/api/offline/status`, and `/api/camera/status`.

Use the dashboard's **Offline event queue** panel to view pending/synced/failed counts. **Sync queued events** triggers one delivery attempt for due events.

## 5. Required only for remote synchronization

SkillWatchAI does not include your organization's central event receiver, so you must supply its URL before queued events can be sent to another server:

```dotenv
SKILLWATCH_SYNC_URL=https://your-trusted-server.example/api/events
SKILLWATCH_SYNC_TOKEN=your-token-if-required
```

Replace both example values with details from the actual receiver, then restart the backend. Do not use these placeholders literally. The receiver must accept JSON fields `event_id`, `event_type`, `created_at`, and `payload`, and deduplicate `event_id` / `Idempotency-Key`. If no receiver is available, leave these variables unset; events remain stored in `data/offline_events.db` locally.

## 6. Camera reliability checks

- Connect a laptop/USB webcam and verify the camera health label and frame age update.
- Disconnect/reconnect the live camera and verify reconnection attempts are reflected.
- Use a dark or deliberately blurred test view to see image-quality warnings.
- Use a test stream with repeated frames to verify frozen-feed detection.
- Device and RTSP cameras auto-reconnect; local video files are not automatically reopened after reaching end-of-file.

## 7. Preserve your existing records

Before copying files into an existing installation, back up `data\`, especially `students.db`, `attendance.db`, `compliance_alerts.db`, `.face_key`, `cameras.json`, `infrastructure_config.json`, and `recognition_log.jsonl`. Prefer extracting this ZIP into a new folder first. The new event queue is a separate `data\offline_events.db` file.
