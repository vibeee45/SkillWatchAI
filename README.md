# SkillWatch AI — Phase 8

Phase 8 adds local person detection and occupancy analytics on top of Phases 1–7.

## Phase 8 checklist
- 8.1 Lightweight person detector: OpenCV HOG (CPU)
- 8.2 Detect people in classroom frames
- 8.3 Track people across frames with IoU tracking
- 8.4 Stable track IDs prevent repeated counting
- 8.5 Configurable normalized classroom zones
- 8.6 Total occupancy
- 8.7 Five-frame median temporal smoothing
- 8.8 Recognized students vs total persons
- 8.9 Unknown persons = smoothed occupancy minus confirmed recognized tracks
- 8.10 Occupancy statistics in dashboard and API

## API
- `GET /api/occupancy/status`
- `POST /api/occupancy/reset`
- `GET /api/occupancy/zones`
- `POST /api/occupancy/zones`

## Important prototype note
The person detector is deliberately lightweight and CPU-friendly. HOG is not a modern deep-learning detector and may miss people in difficult classroom views. Phase 8 should therefore be validated on the target camera before being treated as a final production detector.

# Phase 9 — Observation-Only Infrastructure Detection

Phase 9 uses YOLOv8n ONNX to report visible equipment detections. Because classrooms and training centres have different equipment quantities, the prototype does not set expected counts, configure inventory, calculate shortages, or generate inventory-based alerts. See `PHASE9_INFRASTRUCTURE.md` for details and limitations.

Main endpoints:
- `GET /api/infrastructure/status`
- `POST /api/infrastructure/test`
- `GET /api/infrastructure/debug-frame.jpg`
- `GET /api/infrastructure/zones`
- `PUT /api/infrastructure/zones`
- `POST /api/infrastructure/reset`

The current supported classes are chair, table (COCO dining-table category), TV/monitor (COCO TV category), and laptop. Counts are observed detections in the current frame, not expected inventory. Zone geometry is stored in `data/infrastructure_config.json`; equipment quantities are not stored or used.

## Phase 10 — Compliance & Alert Engine

Phase 10 adds persistent, configurable rules for attendance discrepancies, individual attendance mismatches, low occupancy, observation-only infrastructure changes, overcapacity, camera health, and low-confidence/manual verification. Alerts record rule ID, severity, confidence, timestamps, repeated evidence, scope, and evidence metadata in SQLite. No expected equipment counts or inventory-shortage rules are used. See `PHASE10_COMPLIANCE_ALERT_ENGINE.md`.

Compliance API: `GET /api/compliance/status`, `GET/PUT /api/compliance/config`, `POST /api/compliance/evaluate`, `GET /api/compliance/alerts`, `GET /api/compliance/events`, and `POST /api/compliance/alerts/{alert_id}/resolve`.
