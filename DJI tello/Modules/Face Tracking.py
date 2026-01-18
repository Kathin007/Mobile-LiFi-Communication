import cv2
import numpy as np
from djitellopy import tello
import time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

width, height = 360, 240
forward_backward_range = [6200, 6800]  # safe distance (face area range)
pid = [0.4, 0.4, 0]
p_error = 0

def find_face(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, 1.2, 8)

    face_list_center = []
    face_list_area = []

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        face_list_center.append([cx, cy])
        face_list_area.append(area)

    if len(face_list_center) != 0:
        max_area_idx = face_list_area.index(max(face_list_area))
        return image, [face_list_center[max_area_idx], face_list_area[max_area_idx]]
    else:
        return image, [[0, 0], 0]


def track_face(drone, info, width, height, pid, p_error):
    area = info[1]
    x, y = info[0]

    # --- Horizontal rotation (Yaw) ---
    error_x = x - width // 2
    yaw_speed = pid[0] * error_x + pid[1] * (error_x - p_error)
    yaw_speed = int(np.clip(yaw_speed, -100, 100))

    # --- Forward / Backward ---
    fb_speed = 0
    if forward_backward_range[0] < area < forward_backward_range[1]:
        fb_speed = 0
    elif area > forward_backward_range[1]:
        fb_speed = -20
    elif area < forward_backward_range[0] and area != 0:
        fb_speed = 20

    # --- Up / Down ---
    ud_speed = 0
    if y != 0:  # only if a face is detected
        error_y = y - height // 2
        ud_speed = int(np.clip(-0.4 * error_y, -20, 20))  # negative because higher y = lower in frame

    # If no face detected
    if x == 0:
        yaw_speed, fb_speed, ud_speed = 0, 0, 0
        error_x = 0

    drone.send_rc_control(0, fb_speed, ud_speed, yaw_speed)
    return error_x


tello_drone = tello.Tello()
tello_drone.connect()

tello_drone.streamon()
tello_drone.takeoff()
tello_drone.send_rc_control(0, 0, 25, 0)  # small lift-off
time.sleep(1)

while True:
    frame = tello_drone.get_frame_read().frame
    frame = cv2.resize(frame, (width, height))
    frame, img_info = find_face(frame)
    p_error = track_face(tello_drone, img_info, width, height, pid, p_error)

    cv2.imshow("cam feed", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        tello_drone.streamoff()
        tello_drone.land()
        break
