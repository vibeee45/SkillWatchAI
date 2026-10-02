import cv2
from camera_manager import CameraManager


def main():
    video_path = input("Enter video file path: ").strip()

    if not video_path:
        print("ERROR: Video path cannot be empty.")
        return

    camera = CameraManager()

    print("\nOpening video file...")

    if not camera.connect_video_file(video_path):
        print("ERROR: Could not open video file.")
        return

    print("Video file opened successfully.")

    print("\nVideo status:")
    print(camera.get_status())

    print("\nPress Q to quit.")

    while True:
        success, frame = camera.read_frame()

        if not success:
            print("Video finished or frame could not be read.")
            break

        cv2.imshow("SkillWatch AI - Video File", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.disconnect()
    cv2.destroyAllWindows()

    print("Video file closed.")


if __name__ == "__main__":
    main()