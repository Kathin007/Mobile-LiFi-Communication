"""
phase2_defense_gui.py
PHASE 2: INTEGRATED AUTONOMOUS DEFENSE SYSTEM
- Combines 'Gap Seeker' AI (Phase 1) with 'Defense Dashboard' (GUI).
- Features: Safety Corridor HUD, Auto-Forward Logic, 3D Pathing, LiFi Logs.
"""

import sys
import cv2
import time
import json
import numpy as np
from datetime import datetime
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGridLayout,
                             QGroupBox, QProgressBar, QTextEdit, QTabWidget,
                             QLineEdit, QFileDialog)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QPainter, QPen
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# --- HARDWARE IMPORTS ---
try:
    from djitellopy import Tello
    from ultralytics import YOLO
    DRONE_AVAILABLE = True
except ImportError:
    DRONE_AVAILABLE = False
    print("[WARNING] djitellopy or ultralytics not found. Running in SIMULATION mode.")

# --- AI CONFIGURATION (FROM PHASE 1) ---
MODEL_PATH = "yolov8n.pt" 
CONF_THRESHOLD = 0.5
SAFETY_CORRIDOR_W = 0.30   # Center 30% width
SAFETY_CORRIDOR_H = 0.30   # Center 30% height
FORWARD_SPEED = 25         # Gap seeking speed
LATERAL_SPEED = 35         # Avoidance speed
MIN_MOVE_CM = 20           
MAX_MOVE_CM = 100          
FOCAL_LENGTH_PX = 550.0

# ==================== AI LOGIC THREAD ====================
class VideoThread(QThread):
    """
    The Brain: Handles Video + Phase 1 Gap Seeking Logic
    """
    frame_ready = pyqtSignal(np.ndarray)
    log_message = pyqtSignal(str)
    telemetry_update = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.ai_mode = False
        self.drone = None
        self.model = None
        
        # Flight State
        self.battery = 100
        self.height = 0
        self.tof = 100
        self.pos_x, self.pos_y, self.pos_z = 0, 0, 0
        
    def run(self):
        self.running = True
        
        # 1. Connect Drone
        if DRONE_AVAILABLE:
            try:
                self.drone = Tello()
                self.drone.connect()
                self.drone.streamon()
                self.log_message.emit("[INIT] Drone Connected Successfully.")
            except Exception as e:
                self.log_message.emit(f"[ERROR] Drone Connection Failed: {e}")
                self.drone = None
        
        # 2. Load AI
        try:
            self.model = YOLO(MODEL_PATH)
            self.log_message.emit("[INIT] YOLOv8 Neural Network Loaded.")
        except:
            self.log_message.emit("[ERROR] YOLO Model not found. AI Disabled.")

        # 3. Main Loop
        cap = cv2.VideoCapture(0) if not self.drone else None
        
        while self.running:
            # --- GET FRAME ---
            if self.drone:
                frame = self.drone.get_frame_read().frame
                self.battery = self.drone.get_battery()
                self.height = self.drone.get_height()
                self.tof = self.drone.get_distance_tof()
            elif cap:
                ret, frame = cap.read()
                if not ret: continue
                # Simulation Data
                self.battery = max(0, self.battery - 0.01)
                self.height = 100
            else:
                continue

            frame = cv2.resize(frame, (640, 480))
            h, w = frame.shape[:2]
            
            # --- PHASE 1 AI LOGIC ---
            decision = "HOVER"
            center_clear = True
            obstacle_detected = False
            
            # Run Detection
            if self.model:
                results = self.model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
                
                # Define Safety Corridor
                cor_min_x = int(w/2 - (w * SAFETY_CORRIDOR_W)/2)
                cor_max_x = int(w/2 + (w * SAFETY_CORRIDOR_W)/2)
                
                largest_obs = None
                max_area = 0
                
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls != 0: continue # Only Person (0)
                    
                    obstacle_detected = True
                    bx, by, bw, bh = box.xywh[0].cpu().numpy()
                    x1 = int(bx - bw/2)
                    x2 = int(bx + bw/2)
                    y1 = int(by - bh/2)
                    y2 = int(by + bh/2)
                    
                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Check Corridor Collision
                    if (x1 < cor_max_x) and (x2 > cor_min_x):
                        center_clear = False
                        area = bw * bh
                        if area > max_area:
                            max_area = area
                            largest_obs = (bx, by, bw, bh)

            # --- FLIGHT CONTROL ---
            if self.ai_mode and self.drone:
                if obstacle_detected:
                    if center_clear:
                        # GAP SEEKING
                        decision = "GAP FOUND -> FORWARD"
                        self.drone.send_rc_control(0, FORWARD_SPEED, 0, 0)
                        self.pos_y += 1 # Update virtual map
                    elif largest_obs:
                        # DYNAMIC AVOIDANCE
                        direction, dist = self.calculate_gap_move(largest_obs, (w, h))
                        decision = f"BLOCKED -> {direction.upper()}"
                        self.execute_avoid(direction)
                else:
                    # CLEAR SKY
                    decision = "CLEAR -> FORWARD"
                    self.drone.send_rc_control(0, FORWARD_SPEED, 0, 0)
                    self.pos_y += 1

            # --- HUD DRAWING ---
            self.draw_hud(frame, decision, center_clear)
            
            # Emit Data
            self.frame_ready.emit(frame)
            self.telemetry_update.emit({
                'bat': int(self.battery),
                'h': int(self.height),
                'tof': int(self.tof),
                'decision': decision,
                'pos': (self.pos_x, self.pos_y, self.pos_z)
            })
            
            time.sleep(0.05)

        if self.drone: self.drone.streamoff()
        if cap: cap.release()

    def calculate_gap_move(self, bbox, frame_size):
        """Phase 1 Logic: Find largest gap"""
        fx, fy, fw, fh = bbox
        im_w, im_h = frame_size
        gap_left = fx
        gap_right = im_w - (fx + fw)
        gap_top = fy
        gap_bottom = im_h - (fy + fh)
        
        gaps = {'left': gap_left, 'right': gap_right, 'up': gap_top, 'down': gap_bottom}
        best_dir = max(gaps, key=gaps.get)
        return best_dir, 0 # Distance not strictly needed for RC control

    def execute_avoid(self, direction):
        """Non-blocking RC commands"""
        lr, fb, ud, yaw = 0, 0, 0, 0
        if direction == 'left': lr = -LATERAL_SPEED; self.pos_x -= 1
        elif direction == 'right': lr = LATERAL_SPEED; self.pos_x += 1
        elif direction == 'up': ud = LATERAL_SPEED; self.pos_z += 1
        elif direction == 'down': ud = -LATERAL_SPEED; self.pos_z -= 1
        self.drone.send_rc_control(lr, fb, ud, yaw)

    def draw_hud(self, frame, decision, center_clear):
        """Draws the Safety Corridor"""
        h, w = frame.shape[:2]
        cw = int(w * SAFETY_CORRIDOR_W)
        ch = int(h * SAFETY_CORRIDOR_H)
        cx, cy = w // 2, h // 2
        color = (0, 255, 0) if center_clear else (0, 0, 255)
        
        # Virtual Tunnel
        cv2.rectangle(frame, (cx - cw//2, cy - ch//2), (cx + cw//2, cy + ch//2), color, 2)
        cv2.line(frame, (cx, cy-10), (cx, cy+10), (0, 255, 0), 1) # Crosshair
        cv2.line(frame, (cx-10, cy), (cx+10, cy), (0, 255, 0), 1)
        
        # Text Overlay
        cv2.putText(frame, f"AI: {decision}", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def toggle_ai(self):
        self.ai_mode = not self.ai_mode
        msg = "AI AUTOPILOT ENGAGED" if self.ai_mode else "MANUAL CONTROL"
        self.log_message.emit(f"[MODE] {msg}")
        if not self.ai_mode and self.drone:
            self.drone.send_rc_control(0,0,0,0) # Stop immediately

    def send_cmd(self, lr, fb, ud, yaw):
        if not self.ai_mode and self.drone:
            self.drone.send_rc_control(lr, fb, ud, yaw)

    def takeoff(self):
        if self.drone: self.drone.takeoff()
    def land(self):
        if self.drone: self.drone.land()

# ==================== 3D MAP WIDGET ====================
class FlightPathWidget(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 4), facecolor='#121212')
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.positions = deque(maxlen=50)
        self.setup_plot()
        
    def setup_plot(self):
        self.ax.set_facecolor('#121212')
        self.ax.grid(False)
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.set_xlabel('X', color='#00ff64'); self.ax.set_ylabel('Y', color='#00ff64'); self.ax.set_zlabel('Z', color='#00ff64')
        self.ax.tick_params(colors='#00ff64')
        
    def update_pos(self, x, y, z):
        self.positions.append((x, y, z))
        if len(self.positions) > 1:
            xs, ys, zs = zip(*self.positions)
            self.ax.clear()
            self.setup_plot()
            self.ax.plot(xs, ys, zs, color='#00ff64', linewidth=2)
            self.ax.scatter([x], [y], [z], color='red', s=50)
            self.draw()

# ==================== MAIN GUI ====================
class DefenseDroneGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroMinds - PHASE 2 INTEGRATED SYSTEM")
        self.setGeometry(50, 50, 1400, 900)
        self.set_theme()
        
        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # --- LEFT PANEL: VIDEO & CONTROLS ---
        left_layout = QVBoxLayout()
        
        # Video Feed
        self.video_label = QLabel("Initializing Optical Feed...")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #00ff64; background: #000;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)
        
        # Control Grid
        controls = QGroupBox("🎮 MISSION CONTROL")
        c_layout = QGridLayout()
        
        self.btn_takeoff = QPushButton("🚁 TAKEOFF")
        self.btn_land = QPushButton("🛬 LAND")
        self.btn_ai = QPushButton("🧠 AI MODE: OFF")
        self.btn_ai.setStyleSheet("color: #ff0000; border-color: #ff0000;")
        
        c_layout.addWidget(self.btn_takeoff, 0, 0)
        c_layout.addWidget(self.btn_land, 0, 1)
        c_layout.addWidget(self.btn_ai, 1, 0, 1, 2)
        
        controls.setLayout(c_layout)
        left_layout.addWidget(controls)
        layout.addLayout(left_layout, 65)
        
        # --- RIGHT PANEL: DATA & LiFi ---
        right_layout = QVBoxLayout()
        
        # Telemetry
        self.telem_group = QGroupBox("📊 LIVE TELEMETRY")
        t_layout = QGridLayout()
        self.lbl_bat = QLabel("🔋 BAT: --%")
        self.lbl_h = QLabel("📏 ALT: -- cm")
        self.lbl_act = QLabel("🤖 ACT: WAIT")
        t_layout.addWidget(self.lbl_bat, 0, 0)
        t_layout.addWidget(self.lbl_h, 0, 1)
        t_layout.addWidget(self.lbl_act, 1, 0, 1, 2)
        self.telem_group.setLayout(t_layout)
        right_layout.addWidget(self.telem_group)
        
        # 3D Map
        map_group = QGroupBox("🗺️ TACTICAL MAP")
        m_layout = QVBoxLayout()
        self.map_widget = FlightPathWidget()
        m_layout.addWidget(self.map_widget)
        map_group.setLayout(m_layout)
        right_layout.addWidget(map_group)
        
        # Logs
        log_group = QGroupBox("📜 SYSTEM LOGS")
        l_layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("font-family: Consolas; font-size: 10pt; color: #00ff64;")
        l_layout.addWidget(self.log_box)
        log_group.setLayout(l_layout)
        right_layout.addWidget(log_group)
        
        layout.addLayout(right_layout, 35)
        
        # --- LOGIC CONNECTION ---
        self.thread = VideoThread()
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.log_message.connect(self.log)
        self.thread.telemetry_update.connect(self.update_telemetry)
        self.thread.start()
        
        # Buttons
        self.btn_takeoff.clicked.connect(self.thread.takeoff)
        self.btn_land.clicked.connect(self.thread.land)
        self.btn_ai.clicked.connect(self.toggle_ai_ui)

    def set_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QGroupBox { border: 2px solid #005500; border-radius: 5px; margin-top: 10px; color: #00ff64; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #222; border: 1px solid #00ff64; color: #00ff64; padding: 10px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #00ff64; color: #000; }
            QLabel { color: #fff; font-size: 11pt; font-weight: bold; }
        """)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, w*ch, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_telemetry(self, data):
        self.lbl_bat.setText(f"🔋 BAT: {data['bat']}%")
        self.lbl_h.setText(f"📏 ALT: {data['h']} cm")
        self.lbl_act.setText(f"🤖 ACT: {data['decision']}")
        self.map_widget.update_pos(*data['pos'])

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")

    def toggle_ai_ui(self):
        self.thread.toggle_ai()
        if self.thread.ai_mode:
            self.btn_ai.setText("🧠 AI MODE: ON")
            self.btn_ai.setStyleSheet("background-color: #00ff64; color: #000;")
        else:
            self.btn_ai.setText("🧠 AI MODE: OFF")
            self.btn_ai.setStyleSheet("color: #ff0000; border-color: #ff0000;")

    # Keyboard Control (Manual Fallback)
    def keyPressEvent(self, event):
        if not self.thread.ai_mode:
            k = event.key()
            s = 50
            if k == Qt.Key_W: self.thread.send_cmd(0, s, 0, 0)
            elif k == Qt.Key_S: self.thread.send_cmd(0, -s, 0, 0)
            elif k == Qt.Key_A: self.thread.send_cmd(-s, 0, 0, 0)
            elif k == Qt.Key_D: self.thread.send_cmd(s, 0, 0, 0)
            elif k == Qt.Key_Up: self.thread.send_cmd(0, 0, s, 0)
            elif k == Qt.Key_Down: self.thread.send_cmd(0, 0, -s, 0)
            elif k == Qt.Key_Left: self.thread.send_cmd(0, 0, 0, -s)
            elif k == Qt.Key_Right: self.thread.send_cmd(0, 0, 0, s)
    
    def keyReleaseEvent(self, event):
        if not self.thread.ai_mode:
            self.thread.send_cmd(0,0,0,0)

    def closeEvent(self, event):
        self.thread.running = False
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DefenseDroneGUI()
    window.show()
    sys.exit(app.exec_())