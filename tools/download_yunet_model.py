from pathlib import Path
from urllib.request import urlopen

MODEL_URL = "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_detection_yunet_2023mar.onnx"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YuNet model to: {MODEL_PATH}")
    with urlopen(MODEL_URL, timeout=60) as response, MODEL_PATH.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    print(f"Downloaded: {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
