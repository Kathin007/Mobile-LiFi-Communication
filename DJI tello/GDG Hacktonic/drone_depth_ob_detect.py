"""
drone_depth_yolo_ui_final.py
DJI Tello + MiDaS_small + YOLOv8 + Manual Controls (Pygame KeyPressModule)
Polished UI for Hacktonic Demo:
- Left: RGB with YOLO detections
- Right: Depth from MiDaS (smoothed)
- HUD overlay: battery, height, status, FPS, YOLO count, movement
"""

import os
import time
import threading
from queue import Queue
from collections import deque
import cv2
import numpy as np
import torch
import warnings
import csv
from djitellopy import Tello
import KeyPressModule as kpm   # your working pygame module

import sys
sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "model")
YOLO_BEST = os.path.join(MODEL_DIR, "best.pt")
YOLO_DEFAULT = os.path.join(MODEL_DIR, "yolov8n.pt")
YOLO_MODEL = YOLO_BEST if os.path.exists(YOLO_BEST) else YOLO_DEFAULT
CSV_LOG = os.path.join(BASE_DIR, "telemetry.csv")

WIDTH, HEIGHT = 640, 480
TARGET_FPS = 20
DEPTH_W, DEPTH_H = 320, 240
FRAME_SKIP = 2
DEPTH_BUFFER = 5
SPEED = 36
YOLO_SKIP = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HUD
HUD_HEIGHT = 90
HUD_BG_COLOR = (15, 15, 15)
HUD_ALPHA = 0.55
SMOOTHING_ALPHA = 0.65

# Controls
TAKEOFF_KEY, LAND_KEY, EMER_KEY, CAP_KEY, EXIT_KEY = "SPACE", "q", "z", "c", "ESCAPE"

print(f"Model: {YOLO_MODEL} | Device: {DEVICE}")

# ---------------- LOAD MODELS ----------------
print("Loading MiDaS_small...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform
midas.to(DEVICE).eval()
print("✅ MiDaS loaded.")

try:
    from ultralytics import YOLO
    yolo = YOLO(YOLO_MODEL)
    print(f"✅ YOLO model loaded from {YOLO_MODEL}")
except Exception as e:
    print("⚠️ YOLO load failed:", e)
    yolo = None

# ---------------- GLOBALS ----------------
frame_q = Queue(maxsize=4)
display_q = Queue(maxsize=4)
depth_buf = deque(maxlen=DEPTH_BUFFER)
stop_event = threading.Event()
telemetry = {"battery": 0, "height": 0, "status": "Idle"}
tello = None

# FPS + connection
_cam_fps_time = time.time()
_cam_frame_count = 0
cam_fps_display = 0.0
_last_infer_fps = 0.0
frame_drop_counter = 0
frame_ok_counter = 0

# ---------------- CAMERA THREAD ----------------
def camera_reader():
    global tello, frame_drop_counter, frame_ok_counter
    try:
        tello = Tello()
        tello.connect()
        telemetry["battery"] = tello.get_battery() or 0
        print(f"Tello connected ✅ | Battery: {telemetry['battery']}%")
        tello.streamon()
        time.sleep(0.5)
    except Exception as e:
        print("⚠️ Tello connect failed:", e)
        stop_event.set()
        return

    fid = 0
    while not stop_event.is_set():
        try:
            frame = tello.get_frame_read().frame
            if frame is None:
                frame_drop_counter += 1
                time.sleep(0.01)
                continue
            frame_ok_counter += 1
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            if not frame_q.full():
                frame_q.put((frame.copy(), fid))
            fid += 1

            if fid % 30 == 0:
                try:
                    telemetry["battery"] = tello.get_battery() or telemetry["battery"]
                    telemetry["height"] = tello.get_height() or telemetry.get("height", 0)
                except Exception:
                    pass
            time.sleep(1.0 / TARGET_FPS)
        except Exception as e:
            frame_drop_counter += 1
            print("camera_reader error:", e)
            time.sleep(0.05)

    try: tello.streamoff()
    except: pass

# ---------------- DEPTH THREAD ----------------
def depth_processor():
    global _last_infer_fps, cam_fps_display, _cam_fps_time, _cam_frame_count
    frame_cnt = 0
    while not stop_event.is_set():
        if frame_q.empty():
            time.sleep(0.002)
            continue
        frame, fid = frame_q.get()
        frame_cnt += 1

        # Camera FPS
        _cam_frame_count += 1
        if _cam_frame_count >= 10:
            now = time.time()
            cam_fps_display = _cam_frame_count / (now - _cam_fps_time + 1e-6)
            _cam_frame_count = 0
            _cam_fps_time = now

        if frame_cnt % FRAME_SKIP != 0:
            depth_color = depth_buf[-1] if depth_buf else np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            if not display_q.full():
                display_q.put((frame, depth_color, None, fid))
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (DEPTH_W, DEPTH_H))
        input_batch = transform(small).to(DEVICE)

        t0 = time.time()
        with torch.no_grad():
            pred = midas(input_batch)
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(HEIGHT, WIDTH),
                                                   mode="bicubic", align_corners=False).squeeze()
            depth_np = pred.cpu().numpy()
        infer_time = time.time() - t0
        _last_infer_fps = 1.0 / max(infer_time, 1e-6)

        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        depth_buf.append(depth_color)
        avg_depth = np.mean(np.stack(list(depth_buf), axis=0), axis=0).astype(np.uint8)

        if not display_q.full():
            display_q.put((frame, avg_depth, depth_np, fid))

# ---------------- YOLO DETECTION ----------------
def run_yolo(frame):
    detections = []
    if yolo is None:
        return detections
    try:
        res = yolo.predict(source=frame, imgsz=640, device=0 if DEVICE.type == "cuda" else "cpu", verbose=False)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = res.names.get(cls, str(cls))
            detections.append((x1, y1, x2, y2, name, conf))
    except Exception:
        pass
    return detections

# ---------------- HUD HELPERS ----------------
def draw_transparent_rect(img, xy, color=(0,0,0), alpha=0.5):
    x, y, w, h = xy
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def hud_text(img, text, pos, size=0.55, color=(255,255,255), thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

def movement_label(lr, fb, ud, yaw):
    actions = []
    if ud > 0: actions.append("UP")
    elif ud < 0: actions.append("DOWN")
    if fb > 0: actions.append("FORWARD")
    elif fb < 0: actions.append("BACK")
    if lr > 0: actions.append("RIGHT")
    elif lr < 0: actions.append("LEFT")
    if yaw > 0: actions.append("YAW→")
    elif yaw < 0: actions.append("YAW←")
    return " | ".join(actions) if actions else "IDLE"

def signal_bar(img, x, y, quality):
    """Draws small 4-bar signal indicator"""
    bars = int(min(max(quality * 4, 0), 4))
    for i in range(4):
        color = (0,255,0) if i < bars else (70,70,70)
        cv2.rectangle(img, (x + i*10, y - i*4), (x + 8 + i*10, y), color, -1)

# ---------------- MAIN ----------------
def main():
    global tello, frame_drop_counter, frame_ok_counter
    kpm.init()
    cam_t = threading.Thread(target=camera_reader, daemon=True)
    depth_t = threading.Thread(target=depth_processor, daemon=True)
    cam_t.start()
    depth_t.start()

    cv2.namedWindow("MonoSight — RGB|Depth|HUD", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MonoSight — RGB|Depth|HUD", WIDTH*2, HEIGHT + HUD_HEIGHT)

    prev_cmd = np.zeros(4)
    print("Controls: SPACE=Takeoff | Q=Land | Z=Emergency | Arrows/WASD=Move | C=Capture | ESC=Exit")

    yolo_cache = []
    frame_id = 0

    while not stop_event.is_set():
        if display_q.empty():
            time.sleep(0.002)
            continue
        frame, depth_img, depth_np, fid = display_q.get()
        frame_id += 1

        # Run YOLO occasionally
        if yolo is not None and (frame_id % YOLO_SKIP == 0):
            yolo_cache = run_yolo(frame)

        # Draw YOLO
        rgb = frame.copy()
        for (x1, y1, x2, y2, cls, conf) in yolo_cache:
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (14, 202, 255), 2)
            label = f"{cls} {conf:.2f}"
            cv2.putText(rgb, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if depth_np is not None:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                patch = depth_np[max(0, cy-2):min(HEIGHT, cy+2), max(0, cx-2):min(WIDTH, cx+2)]
                if patch.size > 0:
                    rel_depth = np.median(patch)
                    cv2.putText(rgb, f"Rel:{rel_depth:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

        combined = np.hstack((rgb, depth_img))

        # HUD box and data
        draw_transparent_rect(combined, (8, 8, 400, 86), HUD_BG_COLOR, HUD_ALPHA)
        draw_transparent_rect(combined, (0, HEIGHT-18, WIDTH*2, 18), (30,30,30), 0.35)

        # FPS & telemetry
        cam_fps = cam_fps_display
        inf_fps = _last_infer_fps

        lr = fb = ud = yaw = 0
        if kpm.get_key("LEFT"): lr = -SPEED
        elif kpm.get_key("RIGHT"): lr = SPEED
        if kpm.get_key("UP"): fb = SPEED
        elif kpm.get_key("DOWN"): fb = -SPEED
        if kpm.get_key("w"): ud = SPEED
        elif kpm.get_key("s"): ud = -SPEED
        if kpm.get_key("a"): yaw = -SPEED
        elif kpm.get_key("d"): yaw = SPEED

        move_text = movement_label(lr, fb, ud, yaw)

        # Connection quality indicator
        total = frame_ok_counter + frame_drop_counter
        quality = frame_ok_counter / total if total > 10 else 1.0

        # HUD texts
        hud_x, hud_y = 16, 28
        txt1 = f"Battery: {telemetry.get('battery',0)}% | Height: {telemetry.get('height',0)}cm"
        txt2 = f"Status: {telemetry.get('status','Idle')} | FPS: {int(cam_fps):d}/{int(inf_fps):d} | YOLO: {len(yolo_cache)}"
        txt3 = f"Movement: {move_text}"
        txt4 = f"Signal: {int(quality*100)}% | Model: {os.path.basename(YOLO_MODEL)}"

        hud_text(combined, txt1, (hud_x, hud_y), 0.58, (220,220,220), 1)
        hud_text(combined, txt2, (hud_x, hud_y+22), 0.50, (200,200,200), 1)
        hud_text(combined, txt3, (hud_x, hud_y+44), 0.50, (190,230,180), 1)
        hud_text(combined, txt4, (hud_x, hud_y+66), 0.45, (160,160,255), 1)
        signal_bar(combined, hud_x+320, hud_y+60, quality)

        hud_text(combined, "SPACE=Takeoff | Q=Land | Z=Emergency | Arrows/WASD=Move | C=Capture | ESC=Exit",
                  (12, HEIGHT + (HUD_HEIGHT-18)//2 + 6), 0.45, (200,200,200), 1)

        cv2.imshow("MonoSight — RGB|Depth|HUD", combined)

        # One-shot actions
        if kpm.get_key(TAKEOFF_KEY):
            telemetry["status"] = "Flying"
            print("[KEY] Takeoff")
            try: tello.takeoff()
            except: pass
        if kpm.get_key(LAND_KEY):
            telemetry["status"] = "Landing"
            print("[KEY] Land")
            try: tello.land()
            except: pass
        if kpm.get_key(EMER_KEY):
            telemetry["status"] = "Emergency"
            print("[KEY] Emergency stop")
            stop_event.set()
            try: tello.emergency()
            except: pass
            break
        if kpm.get_key(CAP_KEY):
            fname = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, rgb)
            print("📸 Saved:", fname)
        if kpm.get_key(EXIT_KEY):
            print("ESC pressed — exiting.")
            stop_event.set()
            break

        # Smooth control blending
        target = np.array([lr, fb, ud, yaw], dtype=float)
        prev_cmd[:] = prev_cmd * (1 - SMOOTHING_ALPHA) + target * SMOOTHING_ALPHA
        try:
            tello.send_rc_control(int(prev_cmd[0]), int(prev_cmd[1]), int(prev_cmd[2]), int(prev_cmd[3]))
        except:
            pass

        #log_telemetry(fid)
        time.sleep(0.02)

    print("Shutting down...")
    stop_event.set()
    try:
        tello.streamoff()
        tello.end()
    except:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
