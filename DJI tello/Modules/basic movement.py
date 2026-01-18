from djitellopy import Tello
from time import sleep

tello = Tello()
tello.connect()
print(f"Battery percentage: {tello.get_battery()}%")

if tello.get_battery() < 50:
    print("⚠️ Battery too low for flips. Please charge above 50%.")
else:
    tello.takeoff()
    tello.move_up(60)  # Ensure enough altitude
    sleep(2)

    tello.send_rc_control(0, -10, 0, 0)
    sleep(1)
    tello.send_rc_control(0, 0, 0, 0)

    tello.flip_back()
    sleep(3)
    tello.flip_forward()
    sleep(3)

    tello.send_rc_control(0, 30, 0, 0)
    sleep(1)
    tello.land()
