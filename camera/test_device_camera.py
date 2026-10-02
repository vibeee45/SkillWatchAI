import cv2

from camera_manager import CameraManager


def main():
    camera = CameraManager()

    print("Connecting to device camera...")

    if not camera.connect_device_camera(0):
        print("ERROR: Could not connect to device camera.")
        return

    print("Camera connected successfully.")
    print("Camera status:")
    print(camera.get_status())

    print("\nPress Q to quit.")

    while True:
        success, frame = camera.read_frame()

        if not success:
            print("ERROR: Failed to read frame.")
            break

        cv2.imshow("SkillWatch AI - Device Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.disconnect()
    cv2.destroyAllWindows()

    print("Camera disconnected.")


if __name__ == "__main__":
    main()