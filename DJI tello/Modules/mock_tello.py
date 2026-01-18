"""
mock_tello.py
A drop-in replacement for djitellopy.Tello.
It simulates:
1. Video Feed (from a file or webcam)
2. Telemetry (Battery, Height, TOF)
3. Movement (Updates a virtual X,Y,Z coordinate system)
"""
import time
import cv2
import numpy as np
import threading

class MockTello:
    def __init__(self, video_source=0): 
        # video_source: 0 for webcam, or "path/to/video.mp4"
        self.is_flying = False
        self.battery = 87
        self.height = 0
        self.tof = 100  # Time of Flight distance (cm)
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.temp = 44
        
        # Virtual coordinates for the 3D map
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0
        
        # Video Capture
        self.cap = cv2.VideoCapture(video_source)
        self.frame_grabbed = None
        self.running = True
        
        # Start a background thread to read video loop
        self.thread = threading.Thread(target=self._video_loop)
        self.thread.daemon = True
        self.thread.start()

    def _video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # Restart video if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.frame_grabbed = frame
            time.sleep(1/30) # Limit to ~30 FPS

    def connect(self):
        print("[SIM] Connecting to Virtual Drone...")
        time.sleep(0.5)
        print("[SIM] Connected! Battery: 87%")
        return True

    def streamon(self):
        print("[SIM] Video Stream Started")

    def streamoff(self):
        self.running = False
        self.cap.release()

    def get_frame_read(self):
        return self

    @property
    def frame(self):
        if self.frame_grabbed is not None:
            return self.frame_grabbed
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # --- Telemetry Simulation ---
    def get_battery(self):
        # Simulate slight drain
        drain = int((time.time() % 100) / 10)
        return max(0, 87 - drain)

    def get_height(self):
        # Add noise to simulate hovering
        return self.height + np.random.randint(-2, 3)

    def get_distance_tof(self):
        return self.tof + np.random.randint(-5, 5)
    
    def get_pitch(self): return np.random.randint(-2, 2)
    def get_roll(self): return np.random.randint(-2, 2)
    def get_yaw(self): return self.yaw

    def get_speed_x(self): return np.random.randint(-5, 5)
    def get_speed_y(self): return np.random.randint(-5, 5)
    def get_speed_z(self): return 0
    
    def get_temperature(self): return self.temp

    # --- Flight Controls (Updates Virtual Position) ---
    def takeoff(self):
        print("[SIM] 🚁 Takeoff Command Received")
        self.is_flying = True
        self.height = 80
        self.pos_z = 80
        time.sleep(1)

    def land(self):
        print("[SIM] 🛬 Land Command Received")
        self.is_flying = False
        self.height = 0
        self.pos_z = 0

    def move_left(self, x):
        print(f"[SIM] ⬅️ Move Left {x}cm")
        self.pos_x -= x

    def move_right(self, x):
        print(f"[SIM] ➡️ Move Right {x}cm")
        self.pos_x += x

    def move_up(self, x):
        self.pos_z += x
        self.height += x

    def move_down(self, x):
        self.pos_z -= x
        self.height -= x

    def move_forward(self, x):
        self.pos_y += x

    def move_back(self, x):
        self.pos_y -= x

    def rotate_clockwise(self, x):
        self.yaw += x

    def send_rc_control(self, lr, fb, ud, y):
        # Just update internal state for visualization
        pass