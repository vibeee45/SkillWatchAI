# Phase 9 — Observation-Only Infrastructure Detection

Phase 9 uses the existing YOLOv8n ONNX model to detect supported equipment classes in camera frames. This is a prototype: different training centres and classrooms have different layouts and equipment quantities, so **the application does not define expected counts or compare observations against an inventory**.

## What it reports

- Detected objects by class in the current valid frame.
- Counts grouped by configured image zone.
- Per-object class, bounding box, confidence, and zone.
- Model diagnostics, inference duration, and errors.
- `SUCCESS`, `LOW_CONFIDENCE`, and `DETECTION_ERROR` states so an invalid frame is not confused with a valid observation containing no detected objects.

## Supported COCO model classes

- `chair` — COCO chair
- `table` — COCO dining table (not a dedicated classroom-desk class)
- `tv_monitor` — COCO TV class
- `laptop` — COCO laptop

These are provisional classes and must be validated on target camera footage. The pretrained COCO model does not guarantee reliable detection of every classroom object or of occluded/off-camera equipment.

## API endpoints

- `GET /api/infrastructure/status` — detector diagnostics and latest observation.
- `POST /api/infrastructure/test` — run detection on the latest camera frame.
- `GET /api/infrastructure/debug-frame.jpg` — inspect the current camera frame.
- `GET /api/infrastructure/zones` — read image-zone definitions.
- `PUT /api/infrastructure/zones` — configure normalized image zones.
- `POST /api/infrastructure/reset` — reset detector metrics and the latest observation.

There are deliberately **no inventory endpoints, expected quantities, shortage calculations, or inventory-based alerts**. Counts are observations, not judgments about whether a classroom has enough equipment.

## Example observation

```json
{
  "detection_valid": true,
  "detection_state": "SUCCESS",
  "counts_by_class": {"chair": 5, "table": 1, "laptop": 2, "tv_monitor": 1},
  "counts_by_zone": {"classroom": {"chair": 5, "table": 1, "laptop": 2, "tv_monitor": 1}},
  "zones": [{"zone_id": "classroom", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}],
  "detections": []
}
```

The `detections` array in a real response contains the actual detected objects and their bounding boxes/confidence. Example counts above are illustrative only.
