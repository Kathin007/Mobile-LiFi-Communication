from djitellopy import Tello

tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

#tello.takeoff()
#tello.move_up(50)
#tello.land()
