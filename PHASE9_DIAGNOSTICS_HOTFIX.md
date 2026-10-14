# Phase 9 diagnostics hotfix

This patch adds camera-frame telemetry and a debug JPEG route to investigate very low YOLO class scores.

## New diagnostics
- `GET /api/infrastructure/debug-frame.jpg`: returns the exact latest cached camera frame. Open it in a browser while the backend is running to confirm it shows the classroom.
- `GET /api/infrastructure/status`: includes `last_frame_stats`, `last_all_class_max_score`, and `last_top_class_scores`.
- `POST /api/infrastructure/test`: if all COCO class scores are below `SKILLWATCH_INFRA_MIN_EVIDENCE_SCORE` (default 0.05) and no equipment is detected, returns `LOW_CONFIDENCE` rather than treating an uncertain frame as a valid zero-object observation. This does not prove the model is accurate; it deliberately avoids inventing counts when inference evidence is weak.

## Test steps
1. Restart backend.
2. Open `http://127.0.0.1:8000/api/infrastructure/debug-frame.jpg` and verify the image is a clear current classroom frame.
3. Call `POST /api/infrastructure/test`.
4. Inspect `last_frame_stats`, `last_top_class_scores`, `last_class_max_scores`, and `detection_state`.
5. If the image is correct but all class scores remain tiny, the next action is to validate the ONNX model artifact itself or test a known-good official YOLOv8n COCO export; do not lower confidence blindly.

The model's real-world detection quality has not been validated by these diagnostics alone.
