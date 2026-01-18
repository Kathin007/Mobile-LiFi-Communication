from djitellopy import tello
import cv2
import time

tello = tello.Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()

while True:
    img = tello.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Drone Stream", img)

    key = cv2.waitKey(1) & 0xFF

    # Press 'c' to capture image
    if key == ord('c'):
        filename = f"capture_{time.time()}.jpg"
        cv2.imwrite(filename, img)
        print(f"Image saved: {filename}")

    # Press 'q' to quit
    elif key == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
