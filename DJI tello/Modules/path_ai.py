"""
phase1_gap_seeker.py
INDUSTRY-GRADE AUTONOMOUS NAVIGATION CORE (FIXED)
Features:
1. 'Safety Corridor' Gap Seeking (Flies through holes in walls)
2. Dynamic Clearance Calculation (Math-based)
3. Fixed Keyboard Input & Graceful Exit
"""

import cv2
import time
import math
import numpy as np
from ultralytics import YOLO
from djitellopy import Tello
import KeyPressModule as kpm

# --- CONFIGURATION ---
MODEL_PATH = r"Models\yolov8n.pt"  # Ensure path matches your folder
CONF_THRESHOLD = 0.5
TARGET_CLASSES = [0]  # 0=Person

# FLIGHT PHYSICS
FORWARD_SPEED = 25         # Speed when gap is clear (cm/s)
LATERAL_SPEED = 35         # Speed when avoiding (cm/s)
MANUAL_SPEED = 50          # Speed for manual control
SAFETY_CORRIDOR_W = 0.30   # Center 30% width must be clear
SAFETY_CORRIDOR_H = 0.30   # Center 30% height
MIN_MOVE_CM = 20           
MAX_MOVE_CM = 100          
FOCAL_LENGTH_PX = 550.0    

# --- GLOBAL STATE ---
class DroneState:
    def __init__(self):
        self.ai_mode = False
        self.flying = False
        self.battery = 100
        self.current_pos = [0, 0, 0] # x, y, z

state = DroneState()

# --- HELPER: KEYBOARD INPUT (Fixed) ---
def get_keyboard_input():
    """
    Reads keypresses and returns [lr, fb, ud, yaw] values.
    Defined locally to avoid AttributeError.
    """
    lr, fb, ud, yaw = 0, 0, 0, 0
    speed = MANUAL_SPEED

    # Left / Right
    if kpm.get_key("LEFT"): lr = -speed
    elif kpm.get_key("RIGHT"): lr = speed

    # Forward / Backward
    if kpm.get_key("UP"): fb = speed
    elif kpm.get_key("DOWN"): fb = -speed

    # Up / Down
    if kpm.get_key("w"): ud = speed
    elif kpm.get_key("s"): ud = -speed

    # Yaw (Rotate)
    if kpm.get_key("a"): yaw = -speed
    elif kpm.get_key("d"): yaw = speed

    # Land
    if kpm.get_key("q"):
        return None  # Signal to land
        
    return [lr, fb, ud, yaw]

# --- HELPER: HUD & MATH ---
def draw_hud(frame, decision, obstacle_detected, center_clear):
    """Draws Industry-Grade HUD with Safety Corridor"""
    h, w = frame.shape[:2]
    
    # 1. Draw Safety Corridor (The "Tunnel")
    cw = int(w * SAFETY_CORRIDOR_W)
    ch = int(h * SAFETY_CORRIDOR_H)
    cx, cy = w // 2, h // 2
    
    # Color: Green=Go, Red=Blocked
    color = (0, 255, 0) if center_clear else (0, 0, 255)
    
    # Draw bracket style corners
    cv2.rectangle(frame, (cx - cw//2, cy - ch//2), (cx + cw//2, cy + ch//2), color, 2)
    cv2.circle(frame, (cx, cy), 5, color, -1)
    
    # 2. Status Banner
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    
    mode_text = "AI AUTOPILOT" if state.ai_mode else "MANUAL CONTROL"
    cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    status_text = f"BAT: {state.battery}% | ACT: {decision}"
    cv2.putText(frame, status_text, (w - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    
    if obstacle_detected:
        cv2.putText(frame, "OBSTACLE TRACKING", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def calculate_dynamic_move(bbox, frame_size):
    """Calculates best escape vector based on largest gap"""
    fx, fy, fw, fh = bbox
    im_w, im_h = frame_size
    
    gap_left = fx
    gap_right = im_w - (fx + fw)
    gap_top = fy
    gap_bottom = im_h - (fy + fh)
    
    gaps = {'left': gap_left, 'right': gap_right, 'up': gap_top, 'down': gap_bottom}
    best_dir = max(gaps, key=gaps.get)
    
    pixels_per_cm = max(1, fw / 50.0) 
    
    if best_dir == 'left':
        dist_px = (gap_right + fw/2) - (im_w * SAFETY_CORRIDOR_W) 
    elif best_dir == 'right':
        dist_px = (gap_left + fw/2) - (im_w * SAFETY_CORRIDOR_W)
    else:
        dist_px = 100 
        
    move_cm = int(max(MIN_MOVE_CM, min(dist_px / pixels_per_cm, MAX_MOVE_CM)))
    return best_dir, move_cm

# --- MAIN LOOP ---
def main():
    kpm.init()
    drone = None
    
    try:
        # 1. Initialize Hardware
        drone = Tello()
        drone.connect()
        drone.streamon()
        state.battery = drone.get_battery()
        print(f"[INIT] Drone Connected. Battery: {state.battery}%")
    except Exception as e:
        print(f"[ERROR] Connection Failed: {e}")
        return

    # 2. Initialize AI
    try:
        print("[INIT] Loading YOLO Brain...")
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return
    
    print("[READY] Press SPACE to Takeoff. M for AI Toggle. Q to Land.")
    
    try:
        while True:
            # --- SENSOR INPUT ---
            frame_read = drone.get_frame_read()
            if frame_read is None:
                time.sleep(0.1)
                continue

            img = frame_read.frame
            img = cv2.resize(img, (640, 480))
            h, w = img.shape[:2]
            
            # --- USER INPUT ---
            # Call local function instead of kpm module
            vals = get_keyboard_input() 
            
            # Special Keys
            if kpm.get_key("SPACE") and not state.flying:
                drone.takeoff()
                state.flying = True
                time.sleep(1)
            
            if kpm.get_key("q") or vals is None:
                print("[INFO] Landing...")
                if state.flying:
                    drone.land()
                break

            if kpm.get_key("m"):
                state.ai_mode = not state.ai_mode
                drone.send_rc_control(0, 0, 0, 0) # Stop drone on toggle
                print(f"[MODE] AI: {state.ai_mode}")
                time.sleep(0.3) 
                
            decision = "HOVER"
            center_clear = True
            obstacle_detected = False
            
            # --- AI LOGIC ---
            if state.ai_mode and state.flying:
                results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
                
                corridor_x_min = int(w/2 - (w * SAFETY_CORRIDOR_W)/2)
                corridor_x_max = int(w/2 + (w * SAFETY_CORRIDOR_W)/2)
                
                largest_obs = None
                max_area = 0
                
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls not in TARGET_CLASSES: continue
                    
                    obstacle_detected = True
                    bx, by, bw, bh = box.xywh[0].cpu().numpy()
                    x1 = int(bx - bw/2)
                    x2 = int(bx + bw/2)
                    y1 = int(by - bh/2)
                    y2 = int(by + bh/2)
                    
                    # Check Corridor Collision
                    if (x1 < corridor_x_max) and (x2 > corridor_x_min):
                        center_clear = False
                        area = bw * bh
                        if area > max_area:
                            max_area = area
                            largest_obs = (bx, by, bw, bh)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Decision Engine
                if obstacle_detected:
                    if center_clear:
                        decision = "GAP FOUND -> FORWARD"
                        drone.send_rc_control(0, FORWARD_SPEED, 0, 0)
                        state.current_pos[1] += (FORWARD_SPEED * 0.1)
                    
                    elif largest_obs:
                        direction, dist_cm = calculate_dynamic_move(largest_obs, (w, h))
                        decision = f"BLOCKED -> {direction.upper()}"
                        
                        if direction == 'left':
                            drone.send_rc_control(-LATERAL_SPEED, 0, 0, 0)
                            state.current_pos[0] -= dist_cm
                        elif direction == 'right':
                            drone.send_rc_control(LATERAL_SPEED, 0, 0, 0)
                            state.current_pos[0] += dist_cm
                        elif direction == 'up':
                            drone.send_rc_control(0, 0, LATERAL_SPEED, 0)
                            state.current_pos[2] += dist_cm
                        elif direction == 'down':
                            drone.send_rc_control(0, 0, -LATERAL_SPEED, 0)
                            state.current_pos[2] -= dist_cm
                else:
                    decision = "CLEAR -> FORWARD"
                    drone.send_rc_control(0, FORWARD_SPEED, 0, 0)
                    state.current_pos[1] += (FORWARD_SPEED * 0.1)

            elif state.flying and not state.ai_mode:
                # MANUAL MODE
                drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])
                decision = "MANUAL"
                
            # --- UPDATE UI ---
            draw_hud(img, decision, obstacle_detected, center_clear)
            cv2.imshow("Industry Grade Autonomous Core", img)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC key
                break

    except KeyboardInterrupt:
        print("[WARN] Keyboard Interrupt!")
    except Exception as e:
        print(f"[ERROR] Runtime Error: {e}")
    finally:
        print("[INFO] Closing resources...")
        if drone:
            try:
                drone.streamoff()
                drone.end()
            except:
                pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()