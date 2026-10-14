# Phase 11 — Offline Mode & Camera Reliability

## Implemented

| Item | Implementation |
|---|---|
| 11.1 Local SQLite event queue | `data/offline_events.db`, unique event IDs, WAL mode, persistent queue states |
| 11.2 Store events while offline | Attendance creation/corrections, compliance alert transitions, and camera health transitions are written locally first |
| 11.3 Continue local AI during network failure | Recognition, occupancy, attendance persistence, and compliance remain local; remote delivery runs on a separate daemon worker |
| 11.4 Retry logic | Failed deliveries return to pending with exponential backoff; after eight attempts they become terminal failures |
| 11.5 Sync after reconnect | Background worker retries due events every five seconds when a remote receiver is configured; manual trigger: `POST /api/offline/sync` |
| 11.6 Camera reconnect | Live device/RTSP source reconnection after repeated read failures or a stale heartbeat; device index `0` is handled correctly |
| 11.7 Missing heartbeat | Monitor detects a stale last-successful-frame timestamp and marks the camera offline before attempting reconnection |
| 11.8 Frozen feed | Consecutive near-identical downsampled frames trigger `frozen` health and a reconnect attempt |
| 11.9 Poor quality | Persistent low light, overexposure, or low sharpness sets a quality warning |
| 11.10 Camera health display | Dashboard displays health, frame age, image quality metrics, reconnect attempts, and offline queue/sync status |
| 11.11 Offline/recovery tests | Unit tests cover local persistence, deduplication, retry/recovery, sync success, camera reconnect, heartbeat, frozen feed, and poor quality |

## Local data and privacy

The queue uses its own `data/offline_events.db`; it does not replace or migrate `students.db`, `attendance.db`, `compliance_alerts.db`, `data/.face_key`, camera configuration, or recognition logs. Events are retained locally unless remote sync is explicitly configured. Queue payloads can include student identifiers and attendance details.

## Remote synchronization configuration

Remote sync is deliberately disabled by default. The project did not include a central receiver URL or receiver API contract, so one must be provided before queued events can leave this machine.

1. Copy `.env.example` to `.env` (if `.env` does not already exist).
2. Set `SKILLWATCH_SYNC_URL` to a trusted receiver's event-ingestion endpoint.
3. Optionally set `SKILLWATCH_SYNC_TOKEN` to its bearer token.
4. Restart the backend.
5. Check `GET /api/offline/status`, then use `POST /api/offline/sync` for a manual attempt.

The receiver must accept POST JSON with `event_id`, `event_type`, `created_at`, and `payload`; it must deduplicate by `event_id` or `Idempotency-Key`. Delivery is at-least-once: the receiver must be idempotent to handle a process interruption between remote acceptance and local acknowledgement. Prefer HTTPS for non-local receivers. Do not share or commit the real `.env` file.

## Camera testing

- Test with a local webcam first; then test RTSP if available.
- Disconnect/reconnect the camera and verify health/reconnect labels.
- Cover the lens or use a deliberately dark test frame to verify quality warning. Do not use student biometric images as test fixtures.
- To test remote outage, configure a test receiver, stop it temporarily, create attendance/compliance events, and confirm the queue grows. Restart the receiver and wait for retry backoff or click **Sync queued events**.
- Video files are not auto-reopened at end-of-file; automatic reconnect applies to live device and RTSP sources.
