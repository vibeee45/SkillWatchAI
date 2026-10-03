from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
URL = "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10_000_000:
        print(f"SFace model already exists: {MODEL_PATH}")
        return
    print("Downloading SFace model (~37 MB)...")
    request = Request(URL, headers={"User-Agent": "SkillWatchAI/1.0"})
    with urlopen(request, timeout=120) as response, MODEL_PATH.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    print(f"Saved: {MODEL_PATH}")
    print(f"Size: {MODEL_PATH.stat().st_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
