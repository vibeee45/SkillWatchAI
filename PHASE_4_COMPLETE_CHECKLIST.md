# PHASE 4 — FACE DETECTION

- [x] 4.1 Select lightweight local face detector — OpenCV YuNet
- [x] 4.2 Load face-detection model — local ONNX model loader + download helper
- [x] 4.3 Detect faces in camera frames
- [x] 4.4 Return face bounding boxes
- [x] 4.5 Return face confidence scores
- [x] 4.6 Support multiple faces in one frame
- [x] 4.7 Add face-quality checks
- [x] 4.8 Add confidence threshold
- [x] 4.9 Track faces across consecutive frames with lightweight IoU tracker
- [ ] 4.10 Test detection under different lighting and angles — manual validation on user machine

## Privacy scope

Phase 4 performs local face **detection only**. It does not perform face recognition, embeddings, identity matching, or attendance decisions.

## Model

YuNet `face_detection_yunet_2023mar.onnx` from OpenCV Zoo. Download with:

```powershell
python tools/download_yunet_model.py
```

The model is stored under `models/` and is not fetched at runtime.
