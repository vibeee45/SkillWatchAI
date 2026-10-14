import numpy as np
from camera.camera_manager import CameraManager


def test_identical_frames_trigger_frozen_detection():
    manager = CameraManager()

    # Use a textured image so the frame-quality analysis has meaningful input.
    rng = np.random.default_rng(42)
    frame = rng.integers(
        0, 256, size=(480, 640, 3), dtype=np.uint8
    )

    for _ in range(manager.frozen_frame_limit + 2):
        manager._mark_frame_success(frame.copy())

    try:
        assert manager.frozen_frame_count >= manager.frozen_frame_limit
        assert manager.health == "frozen"
        assert manager.is_connected is False
        assert manager.capture is None
    finally:
        if manager.capture:
            manager.capture.release()


if __name__ == "__main__":
    test_identical_frames_trigger_frozen_detection()
    print("PASS: Identical frames trigger frozen-feed detection.")
