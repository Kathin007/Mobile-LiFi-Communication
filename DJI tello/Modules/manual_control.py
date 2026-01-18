from djitellopy import Tello, TelloException
from ultralytics import YOLO
import cv2
import time
from time import sleep
import os
import KeyPressModule as kpm

# ------------------- SETTINGS ------------------- #
MODEL_PATH = r"Models\yolov8n.pt"    # relative to Modules/
SAVE_PATH = r"data\captures"
CONF_THRESHOLD = 0.5
SPEED = 50

# Create save directory if not exists
os.makedirs(SAVE_PATH, exist_ok=True)


def get_keyboard_input(drone):
    """Reads keypresses and returns movement values."""
    lr, fb, ud, yaw = 0, 0, 0, 0

    if kpm.get_key("RIGHT"):
        lr = SPEED
    elif kpm.get_key("LEFT"):
        lr = -SPEED

    if kpm.get_key("UP"):
        fb = SPEED
    elif kpm.get_key("DOWN"):
        fb = -SPEED

    if kpm.get_key("w"):
        ud = SPEED
    elif kpm.get_key("s"):
        ud = -SPEED

    if kpm.get_key("a"):
        yaw = -SPEED
    elif kpm.get_key("d"):
        yaw = SPEED

    if kpm.get_key("q"):
        print("[INFO] Landing (Q pressed)...")
        drone.land()
        return None

    return lr, fb, ud, yaw


def main():
    kpm.init()

    # Connect to Tello
    drone = Tello()
    drone.connect()
    battery = drone.get_battery()
    print(f"[INFO] Battery: {battery}%")

    if battery < 15:
        print("[WARNING] Battery too low for flight! Please charge your drone.")
        return

    drone.streamon()
    print("[INFO] Video stream started")

    # --- Load YOLO model --- #
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading YOLO model from local path: {os.path.abspath(MODEL_PATH)}")
        model = YOLO(MODEL_PATH)
    else:
        print("[INFO] Local YOLO model not found — using default YOLOv8n.pt")
        model = YOLO("yolov8n.pt")

    print("[INFO] YOLO model loaded successfully ✅")

    # --- Takeoff --- #
    try:
        print("[INFO] Taking off...")
        drone.takeoff()
        sleep(2)
    except TelloException as e:
        print(f"[ERROR] Takeoff failed: {e}")
        drone.streamoff()
        cv2.destroyAllWindows()
        return

    try:
        while True:
            movement = get_keyboard_input(drone)
            if movement:
                drone.send_rc_control(*movement)
            else:
                break

            frame = drone.get_frame_read().frame
            frame = cv2.resize(frame, (640, 480))

            # YOLOv8 Detection
            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated_frame = results[0].plot()

            # Display feed
            cv2.imshow("Tello Drone - YOLO Obstacle Detection", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            # Capture image
            if key == ord('c'):
                filename = os.path.join(SAVE_PATH, f"capture_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[INFO] Image saved: {filename}")

            # Exit / land safely
            elif key == 27:  # ESC
                print("[INFO] Landing and closing...")
                drone.land()
                break

            sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — landing safely...")
        drone.land()

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        drone.land()

    finally:
        drone.streamoff()
        cv2.destroyAllWindows()
        print("[INFO] Program ended safely ✅")


if __name__ == "__main__":
    main()