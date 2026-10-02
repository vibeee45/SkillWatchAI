import cv2
from camera_manager import CameraManager


def main():
    rtsp_url = input("Enter RTSP URL: ").strip()

    if not rtsp_url:
        print("ERROR: RTSP URL cannot be empty.")
        return

    camera = CameraManager()

    print("\nConnecting to RTSP camera...")

    if not camera.connect_rtsp(rtsp_url):
        print("ERROR: Could not connect to RTSP camera.")
        print("Check the RTSP URL, network connection, username/password,")
        print("and whether the camera is actually providing an RTSP stream.")
        return

    print("RTSP camera connected successfully.")

    print("\nCamera status:")
    print(camera.get_status())

    print("\nPress Q to quit.")

    while True:
        success, frame = camera.read_frame()

        if not success:
            print("ERROR: Failed to read RTSP frame.")
            break

        cv2.imshow("SkillWatch AI - RTSP Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.disconnect()
    cv2.destroyAllWindows()

    print("RTSP camera disconnected.")


if __name__ == "__main__":
    main()