from djitellopy import Tello
from time import sleep
import KeyPressModule as kpm
import cv2
import time   # for image filenames


def get_keyboard_input(drone):
    lr, fb, ud, yaw = 0, 0, 0, 0
    speed = 50

    # LEFT / RIGHT
    if kpm.get_key("RIGHT"):
        lr = speed
    elif kpm.get_key("LEFT"):
        lr = -speed

    # FORWARD / BACKWARD
    if kpm.get_key("UP"):
        fb = speed
    elif kpm.get_key("DOWN"):
        fb = -speed

    # UP / DOWN
    if kpm.get_key("w"):
        ud = speed
    elif kpm.get_key("s"):
        ud = -speed

    # YAW
    if kpm.get_key("a"):
        yaw = -speed
    elif kpm.get_key("d"):
        yaw = speed

    # Land
    if kpm.get_key("q"):
        drone.land()
        return None

    return lr, fb, ud, yaw


# ---------------- MAIN ---------------- #
kpm.init()
drone = Tello()
drone.connect()
drone.streamon()
print(f"Battery: {drone.get_battery()}%")
drone.takeoff()

while True:
    movement = get_keyboard_input(drone)
    if movement:
        drone.send_rc_control(*movement)
        sleep(0.05)
    else:
        break

    img = drone.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Drone Stream", img)

    key = cv2.waitKey(1) & 0xFF

    # ---- IMAGE CAPTURE (from image capture.py) ----
    if key == ord('c'):
        filename = f"capture_{time.time()}.jpg"
        cv2.imwrite(filename, img)
        print(f"Image saved: {filename}")

    # ---- Exit ----
    elif key == 27:  # ESC
        break

drone.streamoff()
cv2.destroyAllWindows()
