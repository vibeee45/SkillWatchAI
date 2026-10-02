import numpy as np

from camera.frame_output import FramePacket
from processing.frame_pipeline import FramePipeline


def make_packet(frame, number=3):
    return FramePacket(
        frame=frame,
        camera_id="CAM-TEST",
        room="Test Room",
        source_type="device",
        timestamp=FramePacket.now_iso(),
        frame_number=number,
        width=frame.shape[1],
        height=frame.shape[0],
        fps=30.0,
    )


def test_sampling_and_resize():
    pipeline = FramePipeline(sample_every_n=3, target_width=320, target_height=240)
    frame = np.full((100, 200, 3), 128, dtype=np.uint8)
    assert pipeline.process(make_packet(frame, 1)) is None
    result = pipeline.process(make_packet(frame, 3))
    assert result is not None
    assert result.frame.shape == (240, 320, 3)
    assert result.camera_id == "CAM-TEST"
    assert result.frame_id == "CAM-TEST-00000003"


def test_black_frame_detection():
    pipeline = FramePipeline(sample_every_n=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipeline.process(make_packet(frame, 1))
    assert result is not None
    assert result.quality.black is True
    assert result.quality.valid is False


def test_dark_frame_detection():
    pipeline = FramePipeline(sample_every_n=1)
    frame = np.full((100, 100, 3), 20, dtype=np.uint8)
    result = pipeline.process(make_packet(frame, 1))
    assert result is not None
    assert result.quality.very_dark is True


def test_blur_detection():
    pipeline = FramePipeline(sample_every_n=1, blur_threshold=50.0)
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = pipeline.process(make_packet(frame, 1))
    assert result is not None
    assert result.quality.blurred is True


def test_status_and_ai_interface():
    pipeline = FramePipeline(sample_every_n=1)
    frame = np.random.default_rng(1).integers(0, 255, (100, 160, 3), dtype=np.uint8)
    ai_frame = pipeline.process(make_packet(frame, 1))
    assert ai_frame is not None
    result = pipeline.run_ai(ai_frame)
    assert result["processor"] == "noop"
    status = pipeline.get_status()
    assert status["stats"]["processed_frames"] == 1
    assert status["stats"]["last_frame_id"] == "CAM-TEST-00000001"
