# Phase 9 updated package notes

This ZIP includes targeted fixes for infrastructure inference and compliance error handling.

## Changes
- Prefer ONNX Runtime CPU inference, with OpenCV DNN fallback.
- Normalize YOLOv8 COCO output `(1, 84, 8400)` to row-oriented candidates and validate output shape/finite values.
- Expose `last_output_shape`, `last_class_max_scores`, `last_candidate_count`, `failure_count`, `engine`, and `last_error` in detector status.
- Retry OpenCV DNN once after an inference exception by reloading the network.
- On inference failure, return `DETECTION_ERROR` and unknown counts instead of claiming zero equipment.
- Inference errors do not advance the missing-equipment alert streak.
- Add a regression test for error-safe compliance reporting.
- Keep existing student, attendance, camera, and encryption data files untouched; this package does not contain or overwrite runtime `data/` records.

## Apply to an existing checkout
1. Stop the backend with `Ctrl+C`.
2. Back up your project folder (do not delete `data/students.db`, `data/.face_key`, attendance databases, or camera configuration).
3. Extract this ZIP to a temporary folder first.
4. Copy the updated project files over the existing project, accepting replacement for code/docs only. Do not replace your `data/` directory with any archive content.
5. In PowerShell from the project root with `.venv` active, run `python -m pip install -r requirements.txt`.
6. Restart using `uvicorn backend.main:app --reload`.
7. Connect the camera and call `POST /api/infrastructure/test`, then inspect `GET /api/infrastructure/status`.

## Honest limitation
The package cannot guarantee classroom detection accuracy without testing the actual SIH footage. The COCO model has a dining-table class, not a generic desk class. If scores remain low on actual frames, evaluate camera framing and then fine-tune on a labeled classroom dataset. Do not use prototype alerts as a final compliance verdict until validated.
