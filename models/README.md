# Face Detection Model

Phase 4 uses the lightweight local **YuNet** face detector from OpenCV Zoo.

Download the model from the project root:

```powershell
python tools/download_yunet_model.py
```

The expected file is:

```text
models/face_detection_yunet_2023mar.onnx
```

The model is approximately 233 KB. The project does not download models automatically at application startup.
