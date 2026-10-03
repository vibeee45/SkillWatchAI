from pathlib import Path
import cv2

from face_attendance.face_detector import FaceDetector

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"


def main() -> None:
    detector = FaceDetector(MODEL, confidence_threshold=0.6)
    if not detector.load():
        print("ERROR: Could not load YuNet model.")
        print(detector.last_error)
        return

    image_path = input("Enter image path (leave empty for camera 0): ").strip()
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print("ERROR: Could not read image.")
            return
        detections = detector.detect(frame)
        print(f"Faces detected: {len(detections)}")
        for i, detection in enumerate(detections, start=1):
            print(f"Face {i}: {detection.as_dict()}")
        for detection in detections:
            x, y, w, h = detection.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {detection.track_id} {detection.confidence:.2f}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("SkillWatch AI - Face Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera 0.")
        return

    print("Camera running. Press Q to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        for detection in detections:
            x, y, w, h = detection.bbox
            valid = "OK" if detection.quality_valid else "LOW"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Face {detection.track_id} | {detection.confidence:.2f} | {valid}"
            cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Faces: {len(detections)} | YuNet: {detector.last_inference_ms:.1f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("SkillWatch AI - Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
