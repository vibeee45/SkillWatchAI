import cv2
import numpy as np
from camera.camera_manager import CameraManager


def make_frames():
    rng = np.random.default_rng(42)

    # Textured image for meaningful brightness and sharpness analysis.
    clear = rng.integers(40, 210, (480, 640, 3), dtype=np.uint8)

    dark = np.zeros((480, 640, 3), dtype=np.uint8)

    bright = np.full((480, 640, 3), 255, dtype=np.uint8)

    blurred = cv2.GaussianBlur(clear, (51, 51), 0)

    return clear, dark, bright, blurred


def classify(frame):
    manager = CameraManager()
    manager._mark_frame_success(frame.copy())
    return manager.frame_quality["state"]


def test_poor_quality_detection():
    clear, dark, bright, blurred = make_frames()

    assert classify(dark) == "low_light"
    assert classify(bright) == "overexposed"
    assert classify(blurred) == "blurry"
    assert classify(clear) == "good"

    manager = CameraManager()

    for _ in range(manager.poor_quality_frame_limit):
        manager._mark_frame_success(dark.copy())

    assert manager._poor_quality_count >= manager.poor_quality_frame_limit
    assert manager.health == "poor_quality"

    print("PASS: Dark, bright, blurred, and clear frames classified correctly.")
    print("PASS: Five consecutive poor-quality frames trigger poor_quality health.")


if __name__ == "__main__":
    test_poor_quality_detection()
