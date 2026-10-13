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
