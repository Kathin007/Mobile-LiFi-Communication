"""
hackathon_dashboard.py
Runs the Hackathon Simulation using the video file at:
D:\\VS Code Programs\\DJI tello\\test_footage.mp4
"""
import sys
import cv2
import time
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox, 
                             QTextEdit, QGridLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# --- CONFIGURATION ---
# We use r"" to treat backslashes as literal characters
VIDEO_PATH = r"D:\VS Code Programs\DJI tello\test_footage.mp4"
MODEL_PATH = "yolov8n.pt" 

# Import Logic
from ultralytics import YOLO
from mock_tello import MockTello

class DroneDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroMinds - Autonomous Defense Drone Dashboard (SIMULATION)")
        self.setGeometry(100, 100, 1280, 720)
        self.apply_dark_theme()

        # --- Backend Initialization ---
        # Initialize MockTello with YOUR video path
        self.drone = MockTello(video_source=VIDEO_PATH)
        self.drone.connect()
        self.drone.streamon()
        
        try:
            self.model = YOLO(MODEL_PATH)
        except:
            print("Downloading YOLOv8n model...")
            self.model = YOLO("yolov8n.pt")

        self.ai_active = False

        # --- UI Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # LEFT PANEL: Video Feed
        left_layout = QVBoxLayout()
        self.video_label = QLabel("Loading Simulation Footage...")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #00ff00; background: #000;")
        self.video_label.setAlignment(Qt.AlignCenter)
        
        left_layout.addWidget(QLabel("🔭 LIVE OPTICAL FEED (SIMULATION)"))
        left_layout.addWidget(self.video_label)
        
        # Controls
        controls_group = QGroupBox("🎮 MISSION CONTROL")
        controls_layout = QHBoxLayout()
        
        self.btn_takeoff = QPushButton("🚁 TAKEOFF")
        self.btn_takeoff.clicked.connect(self.cmd_takeoff)
        self.btn_ai = QPushButton("🤖 ACTIVATE AI")
        self.btn_ai.clicked.connect(self.toggle_ai)
        self.btn_land = QPushButton("🛬 LAND")
        self.btn_land.clicked.connect(self.cmd_land)
        
        controls_layout.addWidget(self.btn_takeoff)
        controls_layout.addWidget(self.btn_ai)
        controls_layout.addWidget(self.btn_land)
        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)
        main_layout.addLayout(left_layout, 70)

        # RIGHT PANEL: Data & Logs
        right_layout = QVBoxLayout()
        
        # Telemetry
        telem_group = QGroupBox("📊 REAL-TIME TELEMETRY")
        telem_layout = QGridLayout()
        self.lbl_bat = QLabel("🔋 Battery: --%")
        self.lbl_h = QLabel("📏 Height: -- cm")
        self.lbl_tof = QLabel("📡 TOF: -- cm")
        
        telem_layout.addWidget(self.lbl_bat, 0, 0)
        telem_layout.addWidget(self.lbl_h, 0, 1)
        telem_layout.addWidget(self.lbl_tof, 1, 0)
        telem_group.setLayout(telem_layout)
        right_layout.addWidget(telem_group)

        # Logs
        log_group = QGroupBox("📡 SYSTEM LOGS")
        log_layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background: #111; color: #0f0; font-family: Consolas;")
        log_layout.addWidget(self.log_box)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        main_layout.addLayout(right_layout, 30)

        # Update Timer (30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(33)

        self.log(f"System Ready. Loaded footage: {VIDEO_PATH}")

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #fff; font-size: 12px; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; color: #aaa; font-weight: bold; }
            QPushButton { background: #333; color: #fff; border: 1px solid #555; padding: 10px; border-radius: 4px; }
            QPushButton:hover { background: #444; border: 1px solid #00ff00; }
        """)

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")

    def update_loop(self):
        # 1. Get Frame from Mock Drone
        frame = self.drone.frame.copy()
        if frame.shape[0] == 0: return # Skip if empty
        
        frame = cv2.resize(frame, (640, 480))
        
        # 2. Process AI
        if self.ai_active:
            results = self.model.predict(frame, conf=0.5, verbose=False)
            frame = results[0].plot()
            
            # Simple "Avoidance" Logic for Simulation
            for box in results[0].boxes:
                if int(box.cls[0]) == 0: # 0 is Person in YOLO
                    self.log("⚠️ OBSTACLE DETECTED! AI Executing Avoidance...")
                    cv2.putText(frame, "AVOIDING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 3. Update UI
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, w*ch, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

        self.lbl_bat.setText(f"🔋 Battery: {self.drone.get_battery()}%")
        self.lbl_h.setText(f"📏 Height: {self.drone.get_height()} cm")
        self.lbl_tof.setText(f"📡 TOF: {self.drone.get_distance_tof()} cm")

    def cmd_takeoff(self):
        self.drone.takeoff()
        self.log("Taking off...")

    def toggle_ai(self):
        self.ai_active = not self.ai_active
        self.log(f"AI Autopilot {'ENABLED' if self.ai_active else 'DISABLED'}")

    def cmd_land(self):
        self.drone.land()
        self.log("Landing initiated.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DroneDashboard()
    window.show()
    sys.exit(app.exec_())