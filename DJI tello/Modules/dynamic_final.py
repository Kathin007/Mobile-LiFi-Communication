"""
final_dynamic_v2.py
Tello + YOLOv8 — Single-frame dynamic obstacle avoidance with professional HUD
- Single OpenCV window (no pygame window)
- KeyPressModule for background keyboard capture (manual controls work without pygame)
- Dynamic movement proportional to obstacle bbox & TOF (person-only)
- Backtracking + optional automatic backtrack when front clears
- Toggle AI/Manual with M, takeoff with SPACE, backtrack with B, capture C, land Q or ESC
"""

import os
import time
from time import sleep, time as now
import collections
import cv2
import math
import joblib
import pandas as pd
import numpy as np
from ultralytics import YOLO
import KeyPressModule as kpm

# Optional djitellopy import (TEST_MODE if missing)
try:
    from djitellopy import Tello, TelloException
except Exception:
    Tello = None
    TelloException = Exception

# ---------------- CONFIG ---------------- #
MODEL_PATH = r"Models\yolov8n.pt"        # YOLOv8 small model path (required)
AI_MODEL_PATH = r"Models\best_model.pkl" # optional ML model (not required for dynamic moves)
SAVE_DIR = r"data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Camera & frame
FRAME_W, FRAME_H = 640, 480
FRAME_SIZE = (FRAME_W, FRAME_H)

# Detection / Decision
CONF_THRESHOLD = 0.45
TARGET_CLASS_NAME = "person"     # only react to persons
CENTER_TOL = 0.18                # safe corridor half-width fraction (18%)
MIN_BBOX_H_PX = 80               # treat as obstacle if height in px >= this and centered
DECISION_INTERVAL = 0.10         # seconds between loop decisions (approx single-frame processing rate)

# Dynamic move math (tune these)
KNOWN_FOCAL_PX = 550.0           # focal length in px (approx for Tello). Calibrate for better accuracy
SAFETY_MARGIN = 1.15             # multiply computed cm by this
MIN_MOVE_CM = 8.0                # ignore tiny corrections < 8 cm
MAX_MOVE_CM = 150.0              # clamp max move
MIN_CLEARANCE_CM = 35.0          # desired clearance distance in front
VERTICAL_CAP_CM = 120            # don't ascend more than this additional cm

# Movement & safety
SPEED = 50                       # RC speed (manual)
MOVE_STEP_MAX_CM = 60.0          # prefer stepwise moves if computed > this
MIN_BATTERY_TO_MOVE = 15         # percent
EMERGENCY_BATT = 6

# Backtracking
ENABLE_BACKTRACK = True
BACKTRACK_MAX_STEPS = 10
BACKTRACK_CLEAR_FRAMES = 5

# Visual HUD
HUD_BG_ALPHA = 0.6

# TEST MODE (webcam) — set True if you don't have drone connected
TEST_MODE = False
USE_DRONE = (not TEST_MODE) and (Tello is not None)

# ---------------- HELPERS ---------------- #
def px_to_cm(px, depth_cm, focal_px=KNOWN_FOCAL_PX):
    """Convert length in pixels at given depth (cm) to centimeters."""
    return (px * depth_cm) / (focal_px + 1e-9)

def compute_candidate_moves(bbox_cxcywh, frame_size, depth_cm, focal_px=KNOWN_FOCAL_PX):
    """
    Compute candidate move distances (cm) for left/right/up/down/back to clear the center corridor.
    Returns dict {dir:cm} for valid directions (>MIN_MOVE_CM).
    """
    fw, fh = frame_size
    cx, cy, bw, bh = bbox_cxcywh

    corridor_half_x = CENTER_TOL * fw
    corridor_half_y = CENTER_TOL * fh
    left_bound = fw/2 - corridor_half_x
    right_bound = fw/2 + corridor_half_x
    top_bound = fh/2 - corridor_half_y
    bottom_bound = fh/2 + corridor_half_y

    left_x = cx - bw/2
    right_x = cx + bw/2
    top_y = cy - bh/2
    bottom_y = cy + bh/2

    moves_px = {}
    # Left: move left so right_x < right_bound
    delta_px_left = right_x - right_bound
    if delta_px_left > 0:
        moves_px['left'] = delta_px_left

    # Right: move right so left_x > left_bound
    delta_px_right = left_bound - left_x
    if delta_px_right > 0:
        moves_px['right'] = delta_px_right

    # Up: move up so bottom_y < top_bound (move drone up = increase drone altitude)
    delta_px_up = bottom_y - top_bound
    if delta_px_up > 0:
        moves_px['up'] = delta_px_up

    # Down: move down so top_y > bottom_bound
    delta_px_down = bottom_bound - top_y
    if delta_px_down > 0:
        moves_px['down'] = delta_px_down

    # Back: if depth less than desired clearance, back off by difference
    moves_cm = {}
    for d, px in moves_px.items():
        cm = px_to_cm(px, depth_cm, focal_px) * SAFETY_MARGIN
        if cm >= MIN_MOVE_CM:
            if d in ('up','down') and cm > VERTICAL_CAP_CM:
                cm = VERTICAL_CAP_CM
            if cm > MAX_MOVE_CM:
                cm = MAX_MOVE_CM
            moves_cm[d] = float(round(cm,1))

    # Back candidate
    if depth_cm < MIN_CLEARANCE_CM:
        back_cm = (MIN_CLEARANCE_CM - depth_cm) * SAFETY_MARGIN
        if back_cm >= MIN_MOVE_CM:
            moves_cm['back'] = float(round(min(back_cm, MAX_MOVE_CM), 1))

    return moves_cm

def choose_shortest_move(moves_cm):
    """Return (direction, cm) with minimum cm among moves_cm, or (None, None)."""
    if not moves_cm:
        return None, None
    dir_choice = min(moves_cm.items(), key=lambda kv: kv[1])
    return dir_choice  # (direction, cm)

def execute_move(drone, direction, cm):
    """
    Try high-level move_* first, fallback to RC pulses.
    Returns (success:bool, reason:str)
    """
    if drone is None:
        # simulate in TEST_MODE
        sleep(max(0.2, cm/40.0))
        return True, "simulated"
    try:
        # use integer cm for move_* commands
        cm_int = max(1, int(round(cm)))
        if direction == 'left' and hasattr(drone, 'move_left'):
            drone.move_left(cm_int)
        elif direction == 'right' and hasattr(drone, 'move_right'):
            drone.move_right(cm_int)
        elif direction == 'forward' and hasattr(drone, 'move_forward'):
            drone.move_forward(cm_int)
        elif direction == 'back' and hasattr(drone, 'move_back'):
            drone.move_back(cm_int)
        elif direction == 'up' and hasattr(drone, 'move_up'):
            drone.move_up(cm_int)
        elif direction == 'down' and hasattr(drone, 'move_down'):
            drone.move_down(cm_int)
        else:
            # fallback RC pulse
            mapping = {
                'left': (-SPEED, 0, 0, 0),
                'right': (SPEED, 0, 0, 0),
                'forward': (0, SPEED, 0, 0),
                'back': (0, -SPEED, 0, 0),
                'up': (0, 0, SPEED, 0),
                'down': (0, 0, -SPEED, 0),
            }
            if direction not in mapping:
                return False, "invalid_dir"
            lr, fb, ud, yaw = mapping[direction]
            dur = max(0.2, cm/40.0)
            drone.send_rc_control(lr, fb, ud, yaw)
            sleep(dur)
            drone.send_rc_control(0,0,0,0)
        return True, "moved"
    except Exception as e:
        return False, str(e)

def draw_hud(frame, mode_text, battery, height, tof, fps, obstacle_text=None, prediction_text=None):
    """
    Draw a modern semi-transparent HUD at top + crosshair + info lines.
    frame is modified in-place.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # top banner
    banner_h = 60
    cv2.rectangle(overlay, (0,0), (w, banner_h), (20,20,20), -1)
    cv2.addWeighted(overlay, HUD_BG_ALPHA, frame, 1-HUD_BG_ALPHA, 0, frame)

    # Mode / telemetry text
    mode_color = (0,200,0) if mode_text == "AI" else (0,160,255)
    left_text = f"[MODE: {mode_text}]  Battery: {battery}%  Height: {int(height)}cm  TOF: {int(tof)}cm"
    cv2.putText(frame, left_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

    # crosshair
    cx, cy = w//2, h//2
    ch_size = 10
    cv2.line(frame, (cx-ch_size, cy), (cx+ch_size, cy), (200,200,200), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy-ch_size), (cx, cy+ch_size), (200,200,200), 1, cv2.LINE_AA)

    # obstacle & prediction boxes
    if obstacle_text:
        cv2.putText(frame, obstacle_text, (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,220), 2, cv2.LINE_AA)
    if prediction_text:
        cv2.putText(frame, prediction_text, (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2, cv2.LINE_AA)

    return frame

# ---------------- MAIN ---------------- #
def main():
    kpm.init()  # start keyboard capture
    # Load models
    try:
        model_yolo = YOLO(MODEL_PATH)
    except Exception as e:
        print("[ERROR] Could not load YOLO model:", e)
        return

    # try to load optional AI model (not required)
    ai_model = None
    try:
        if os.path.exists(AI_MODEL_PATH):
            ai_model = joblib.load(AI_MODEL_PATH)
            print("[INFO] Optional AI movement model loaded.")
    except Exception as e:
        print("[WARN] Could not load AI model:", e)

    # connect to drone or webcam
    drone = None
    cap = None
    if USE_DRONE and Tello is not None:
        try:
            drone = Tello()
            drone.connect()
            print("[DRONE] Connected. Battery:", drone.get_battery())
            drone.streamon()
        except Exception as e:
            print("[WARN] Drone connect failed, switching to TEST_MODE webcam:", e)
            drone = None

    if drone is None:
        TEST = True
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        print("[INFO] TEST_MODE: using webcam")
    else:
        TEST = False

    # state
    ai_mode = False
    started = False
    backtrack_stack = []
    clear_front_frames = 0
    last_time = now()
    fps_count = 0
    fps = 0.0
    last_decision_time = 0

    window_name = "Tello Drone — Dynamic Obstacle Avoidance (AI/Manual HUD)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_W, FRAME_H)

    print("[READY] SPACE to takeoff (or start webcam), M to toggle AI/manual, B to backtrack, C capture, Q or ESC to land+quit.")

    try:
        while True:
            t0 = now()
            # --- grab frame ---
            if not TEST:
                frame = drone.get_frame_read().frame
            else:
                ret, frame = cap.read()
                if not ret:
                    continue
            frame = cv2.resize(frame, FRAME_SIZE)

            # FPS
            fps_count += 1
            if (now() - last_time) >= 1.0:
                fps = fps_count / (now() - last_time)
                last_time = now()
                fps_count = 0

            # keys: use KeyPressModule (global) and cv2 for ESC / char keys fallback
            key = cv2.waitKey(1) & 0xFF

            # start/takeoff: SPACE key
            if not started:
                # show ready overlay
                ready = frame.copy()
                cv2.putText(ready, "Press SPACE to start (takeoff). M toggle AI/manual. ESC or Q to quit.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 2, cv2.LINE_AA)
                cv2.imshow(window_name, ready)
                if kpm.get_key("SPACE") or key == ord(' '):
                    if not TEST:
                        try:
                            drone.takeoff()
                            sleep(1.2)
                            drone.send_rc_control(0,0,0,0)
                        except Exception as e:
                            print("[WARN] takeoff failed:", e)
                    started = True
                    print("[INFO] Started -> Manual mode by default.")
                    ai_mode = False
                    continue
                if key == 27 or kpm.get_key("q"):
                    print("[INFO] Exiting before start.")
                    break
                continue

            # Toggle AI/manual
            if kpm.get_key("m") or key == ord('m'):
                ai_mode = not ai_mode
                print(f"[MODE] {'AI' if ai_mode else 'MANUAL'} mode")
                sleep(0.2)  # debounce

            # Capture
            if kpm.get_key("c") or key == ord('c'):
                fname = os.path.join(SAVE_DIR, f"capture_{int(time.time())}.jpg")
                cv2.imwrite(fname, frame)
                print("[INFO] Saved", fname)

            # Backtrack manual trigger
            if kpm.get_key("b") or key == ord('b'):
                if backtrack_stack:
                    print("[BACKTRACK] Manual trigger -> reversing stack.")
                    # reverse all steps
                    while backtrack_stack:
                        mv, dist = backtrack_stack.pop()
                        inv = {'left':'right','right':'left','up':'down','down':'up','forward':'back','back':'forward'}.get(mv)
                        if inv:
                            ok, reason = execute_move(drone if not TEST else None, inv, dist)
                            print(f"[BACKTRACK] {inv} {dist}cm -> {ok} ({reason})")
                            sleep(0.25)
                else:
                    print("[BACKTRACK] Stack empty.")

            # Land / quit
            if kpm.get_key("q") or key == 27:
                print("[INFO] Landing and exiting.")
                if not TEST and drone:
                    try:
                        drone.land()
                    except:
                        pass
                break

            # Manual controls (always allow manual RC while in manual mode)
            if not ai_mode:
                lr = 0
                fb = 0
                ud = 0
                yaw = 0
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
                if kpm.get_key("d"):
                    yaw = SPEED
                elif kpm.get_key("a"):
                    yaw = -SPEED

                if not TEST and drone:
                    drone.send_rc_control(int(lr), int(fb), int(ud), int(yaw))
                # show HUD with manual status
                battery = drone.get_battery() if (not TEST and drone) else 100
                height = drone.get_height() if (not TEST and drone) else 0
                tof = drone.get_distance_tof() if (not TEST and drone) else 100
                draw = draw_hud(frame, "MANUAL", battery, height, tof, fps, obstacle_text=None, prediction_text=None)
                cv2.imshow(window_name, draw)
                sleep(DECISION_INTERVAL)
                continue

            # --- AI mode: single-frame detection + dynamic decision ---
            # We only make a decision at most every DECISION_INTERVAL seconds
            if (now() - last_decision_time) < DECISION_INTERVAL:
                # show HUD while waiting next decision
                battery = drone.get_battery() if (not TEST and drone) else 100
                height = drone.get_height() if (not TEST and drone) else 0
                tof = drone.get_distance_tof() if (not TEST and drone) else 100
                draw = draw_hud(frame, "AI", battery, height, tof, fps)
                cv2.imshow(window_name, draw)
                continue
            last_decision_time = now()

            # Run YOLO (single-frame)
            results = model_yolo.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes
            obstacle_text = None
            prediction_text = None
            moved_this_cycle = False

            # Choose the best person candidate (largest bbox)
            candidates = []
            if len(boxes) > 0:
                xywh = boxes.xywh.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                for (cxywh, cconf, ccls) in zip(xywh, confs, cls):
                    if cconf < CONF_THRESHOLD:
                        continue
                    name = model_yolo.names[int(ccls)]
                    if TARGET_CLASS_NAME and name != TARGET_CLASS_NAME:
                        continue
                    candidates.append((cxywh, cconf, name))
            if candidates:
                # pick largest bbox (area)
                candidates.sort(key=lambda t: (t[0][2]*t[0][3]), reverse=True)
                (cx, cy, bw, bh), conf_score, label_name = candidates[0]
                # draw bbox for candidate
                x1 = int(cx - bw/2); y1 = int(cy - bh/2); x2 = int(cx + bw/2); y2 = int(cy + bh/2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
                cv2.putText(frame, f"{label_name} {conf_score:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

                # approximate depth using TOF sensor if available
                if not TEST and drone:
                    try:
                        depth_cm = float(drone.get_distance_tof())
                        if math.isnan(depth_cm) or depth_cm <= 0:
                            depth_cm = 100.0
                    except Exception:
                        depth_cm = 100.0
                else:
                    # fallback estimate: derive depth from bbox height via focal assumption:
                    # depth ≈ (KNOWN_PERSON_HEIGHT_CM * focal_px) / bbox_h_px
                    # use typical person height ~170cm to approximate depth if no TOF
                    KNOWN_PERSON_H_CM = 170.0
                    depth_cm = (KNOWN_PERSON_H_CM * KNOWN_FOCAL_PX) / (bh + 1e-9)

                # consider as obstacle only if centered-ish and tall enough
                dx = abs(cx - FRAME_W/2) / FRAME_W
                dy = abs(cy - FRAME_H/2) / FRAME_H
                is_centered = (dx <= CENTER_TOL) and (dy <= CENTER_TOL)
                if bh >= MIN_BBOX_H_PX and is_centered:
                    obstacle_text = f"OBSTACLE: {label_name.upper()} (conf {conf_score:.2f})"
                    # compute candidates
                    moves_cm = compute_candidate_moves((cx, cy, bw, bh), (FRAME_W, FRAME_H), depth_cm)
                    direction, cm = choose_shortest_move(moves_cm)
                    if direction:
                        prediction_text = f"AI -> {direction.upper()} {cm}cm"
                        # safety checks (battery & try)
                        battery_pct = drone.get_battery() if (not TEST and drone) else 100
                        if battery_pct <= EMERGENCY_BATT:
                            print("[EMERGENCY] Battery critically low. Landing.")
                            if not TEST and drone:
                                drone.land()
                            break
                        if battery_pct < MIN_BATTERY_TO_MOVE:
                            print(f"[SAFETY] Battery {battery_pct}% < {MIN_BATTERY_TO_MOVE}% - skipping move")
                        else:
                            # stepwise movement for very large cm
                            step_cm = min(cm, MOVE_STEP_MAX_CM)
                            success, reason = execute_move(drone if not TEST else None, direction, step_cm)
                            if success:
                                moved_this_cycle = True
                                # push to backtrack stack
                                if ENABLE_BACKTRACK:
                                    backtrack_stack.append((direction, step_cm))
                                    if len(backtrack_stack) > BACKTRACK_MAX_STEPS:
                                        backtrack_stack = backtrack_stack[-BACKTRACK_MAX_STEPS:]
                            else:
                                print("[WARN] Move failed:", reason)
                    else:
                        prediction_text = "AI -> no valid short move"
                else:
                    obstacle_text = None

            # If no candidate person or not an obstacle: clear path - optionally auto-backtrack when cleared
            if not moved_this_cycle:
                # check front clear for auto-backtrack
                if ENABLE_BACKTRACK:
                    center_clear = True
                    # quick center check using single-frame function
                    try:
                        results_check = model_yolo.predict(frame, conf=CONF_THRESHOLD, verbose=False)
                        boxes_check = results_check[0].boxes
                        center_clear = True
                        if len(boxes_check) > 0:
                            xywh_c = boxes_check.xywh.cpu().numpy()
                            confs_c = boxes_check.conf.cpu().numpy()
                            clses_c = boxes_check.cls.cpu().numpy().astype(int)
                            for (cxywh, cconf, ccls) in zip(xywh_c, confs_c, clses_c):
                                if cconf < CONF_THRESHOLD:
                                    continue
                                name = model_yolo.names[int(ccls)]
                                if TARGET_CLASS_NAME and name != TARGET_CLASS_NAME:
                                    continue
                                xcc, ycc, wcc, hcc = cxywh
                                dxcc = abs(xcc - FRAME_W/2)/FRAME_W
                                dycc = abs(ycc - FRAME_H/2)/FRAME_H
                                if hcc >= MIN_BBOX_H_PX and dxcc <= CENTER_TOL and dycc <= CENTER_TOL:
                                    center_clear = False
                                    break
                    except Exception:
                        center_clear = True

                    if center_clear:
                        clear_front_frames += 1
                    else:
                        clear_front_frames = 0

                    if clear_front_frames >= BACKTRACK_CLEAR_FRAMES and backtrack_stack:
                        # perform backtracking
                        print("[BACKTRACK] Front clear -> performing backtrack")
                        while backtrack_stack:
                            mv, dist = backtrack_stack.pop()
                            inv = {'left':'right','right':'left','up':'down','down':'up','forward':'back','back':'forward'}.get(mv)
                            if inv:
                                ok, reason = execute_move(drone if not TEST else None, inv, dist)
                                print(f"[BACKTRACK] {inv} {dist}cm -> {ok} ({reason})")
                                sleep(0.25)
                        clear_front_frames = 0

            # HUD & display
            battery = drone.get_battery() if (not TEST and drone) else 100
            height = drone.get_height() if (not TEST and drone) else 0
            tof = drone.get_distance_tof() if (not TEST and drone) else (depth_cm if 'depth_cm' in locals() else 100)
            mode_label = "AI"
            draw = draw_hud(frame, mode_label, battery, height, tof, fps, obstacle_text=obstacle_text, prediction_text=prediction_text)
            cv2.imshow(window_name, draw)

            # small sleep to keep loop predictable
            elapsed = now() - t0
            if elapsed < DECISION_INTERVAL:
                sleep(max(0.0, DECISION_INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt -> landing & exit")
        if drone and not TEST:
            try:
                drone.land()
            except:
                pass

    finally:
        # cleanup
        if drone and not TEST:
            try:
                drone.streamoff()
            except:
                pass
            try:
                drone.end()
            except:
                pass
        if cap:
            try:
                cap.release()
            except:
                pass
        cv2.destroyAllWindows()
        print("[DONE] Program ended.")

if __name__ == "__main__":
    main()
