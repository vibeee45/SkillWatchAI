#!/usr/bin/env bash
# Render Build Script for SkillWatch AI Backend
set -e

echo "=== [1/3] Upgrading pip and build tools ==="
python -m pip install --upgrade pip setuptools wheel

echo "=== [2/3] Installing dependencies with headless OpenCV for Linux ==="
# Install headless opencv-contrib first to satisfy OpenCV imports in cloud without X11
python -m pip install opencv-contrib-python-headless>=4.10.0
python -m pip install -r requirements.txt

echo "=== [3/3] Verifying AI ONNX models ==="
python tools/download_models.py

echo "=== SkillWatch AI Build Complete ==="
