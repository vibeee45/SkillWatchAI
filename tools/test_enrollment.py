from __future__ import annotations

from pathlib import Path

import cv2

from face_attendance.face_detector import FaceDetector
from face_attendance.enrollment import EnrollmentService
from face_attendance.recognizer import SFaceRecognizer

ROOT = Path(__file__).resolve().parents[1]
YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SFACE = ROOT / "models" / "face_recognition_sface_2021dec.onnx"


def main() -> None:
    detector = FaceDetector(YUNET, confidence_threshold=0.6)
    recognizer = SFaceRecognizer(SFACE)
    if not detector.load():
        print("ERROR: YuNet could not load:", detector.last_error)
        return
    if not recognizer.load():
        print("ERROR: SFace could not load:", recognizer.last_error)
        return

    service = EnrollmentService(detector, recognizer)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera 0.")
        return

    print("Enrollment test: keep exactly one face in view.")
    print("Press C to capture a sample, Q to quit.")
    samples = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        for d in detections:
            x, y, w, h = d.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{d.confidence:.2f} {'OK' if d.quality_valid else 'LOW'}", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(frame, f"Samples: {len(samples)} | C=capture Q=quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("SkillWatch AI - Enrollment Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            try:
                sample = service.process_frame(frame)
                samples.append(sample)
                print(f"Captured sample {len(samples)} | confidence={sample.confidence:.2f} brightness={sample.brightness:.1f} blur={sample.blur_score:.1f}")
            except Exception as exc:
                print("Capture rejected:", exc)

    cap.release()
    cv2.destroyAllWindows()
    if samples:
        print(f"Captured {len(samples)} valid sample(s). Quality score: {service.quality_score(samples):.3f}")


if __name__ == "__main__":
    main()
