"""Download and verify all required ONNX models for SkillWatch AI."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "name": "YuNet Face Detector (ONNX)",
        "file": "face_detection_yunet_2023mar.onnx",
        "url": "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "min_size": 200_000,
    },
    {
        "name": "SFace Face Recognizer (ONNX)",
        "file": "face_recognition_sface_2021dec.onnx",
        "url": "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "min_size": 35_000_000,
    },
]


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading from {url} to {dest}...")
    req = Request(url, headers={"User-Agent": "SkillWatchAI/1.0"})
    with urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    print(f"Saved: {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    print("Checking AI ONNX models in:", MODELS_DIR)
    for model in MODELS:
        path = MODELS_DIR / model["file"]
        if path.exists() and path.stat().st_size >= model["min_size"]:
            print(f"[OK] {model['name']}: {path.name} ({path.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"[FETCH] Missing or incomplete {model['name']}, downloading...")
            try:
                download_file(model["url"], path)
            except Exception as e:
                print(f"[WARN] Failed to download {model['name']}: {e}")

    # YOLOv8n check
    yolo_path = MODELS_DIR / "yolov8n.onnx"
    if yolo_path.exists() and yolo_path.stat().st_size > 10_000_000:
        print(f"[OK] YOLOv8n Detector: {yolo_path.name} ({yolo_path.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"[INFO] YOLOv8n detector is tracked directly in git.")


if __name__ == "__main__":
    main()
