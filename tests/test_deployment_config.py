"""Tests verifying production deployment readiness for Render and Vercel."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_render_blueprint_config():
    render_yaml = ROOT / "render.yaml"
    assert render_yaml.exists(), "render.yaml must exist for Render 1-click deployment"
    content = render_yaml.read_text(encoding="utf-8")
    assert "skillwatch-backend" in content
    assert "skillwatch-db" in content
    assert "uvicorn backend.main:app" in content
    assert "DATABASE_URL" in content


def test_render_build_script():
    build_sh = ROOT / "render_build.sh"
    assert build_sh.exists(), "render_build.sh must exist"
    content = build_sh.read_text(encoding="utf-8")
    assert "pip install" in content
    assert "download_models.py" in content


def test_dockerfile_config():
    dockerfile = ROOT / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile must exist"
    content = dockerfile.read_text(encoding="utf-8")
    assert "python:3.12-slim" in content
    assert "EXPOSE 8000" in content
    assert "uvicorn backend.main:app" in content


def test_vercel_configs():
    root_vercel = ROOT / "vercel.json"
    front_vercel = ROOT / "frontend" / "vercel.json"
    assert root_vercel.exists(), "Root vercel.json must exist"
    assert front_vercel.exists(), "frontend/vercel.json must exist"
    root_content = root_vercel.read_text(encoding="utf-8")
    front_content = front_vercel.read_text(encoding="utf-8")
    assert "/api/:match*" in root_content
    assert "/api/:match*" in front_content


def test_ai_models_present():
    models_dir = ROOT / "models"
    assert (models_dir / "face_detection_yunet_2023mar.onnx").exists()
    assert (models_dir / "face_recognition_sface_2021dec.onnx").exists()
    assert (models_dir / "yolov8n.onnx").exists()
