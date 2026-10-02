from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camera.camera_store import CameraStore


def test_camera_store(tmp_path):
    store = CameraStore(tmp_path / "cameras.json")
    saved = store.upsert({"camera_id":"CAM-TEST","room":"Room 1","source_type":"device","source":"0"})
    assert saved["camera_id"] == "CAM-TEST"
    assert store.get("CAM-TEST")["room"] == "Room 1"
    assert store.delete("CAM-TEST") is True
    assert store.get("CAM-TEST") is None
