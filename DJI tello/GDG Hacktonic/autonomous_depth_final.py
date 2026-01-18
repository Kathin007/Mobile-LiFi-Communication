"""
autonomous_depth_final_graph.py
-------------------------------------------------
Final integrated cockpit (no joystick overlay)
Depth-only autonomous drone controller with:
- MiDaS depth estimation
- Single unified Pygame window (no cv2 window)
- Color-coded depth graph overlay
- Smooth manual/AI toggle
- Telemetry logging
"""

import cv2, torch, time, threading, csv, os, pygame
import numpy as np
from collections import deque
from djitellopy import Tello
import KeyPressModule as kpm

# --------------------------------------------
# CONFIGURATION
# --------------------------------------------
FRAME_W, FRAME_H = 640, 480
DEPTH_W, DEPTH_H = 320, 240
TARGET_FPS = 20
MOVE_STEP_MAX_CM = 60.0
MIN_CLEARANCE_CM = 40.0
MIN_MOVE_CM = 8.0
MAX_MOVE_CM = 120.0
SAFETY_MARGIN = 1.15
CENTER_RECT_W, CENTER_RECT_H = 0.30, 0.40
SIDE_ZONE_W, UP_ZONE_H, DOWN_ZONE_H = 0.30, 0.25, 0.25
SPEED = 50
DECISION_INTERVAL = 0.15
BACKTRACK_MAX = 12
BACKTRACK_CLEAR = 5
GRAPH_HISTORY = 100
LOG_FILE = "telemetry_depth_ai.csv"
WINDOW_SIZE = (1280, 520)

# --------------------------------------------
# LOAD MIDAS
# --------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform
midas.to(device).eval()
print("[INFO] MiDaS loaded on", device)

# --------------------------------------------
# DRONE CONNECTION
# --------------------------------------------
tello = Tello()
tello.connect()
print(f"[INFO] Connected to Tello | Battery: {tello.get_battery()}%")
tello.streamon()

# --------------------------------------------
# THREADS
# --------------------------------------------
frame_lock, depth_lock = threading.Lock(), threading.Lock()
latest_frame, latest_depth = None, None
running = True
depth_buf = deque(maxlen=4)
depth_history = deque(maxlen=GRAPH_HISTORY)

def camera_thread():
    global latest_frame
    while running:
        frame = tello.get_frame_read().frame
        if frame is not None:
            with frame_lock:
                latest_frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        time.sleep(1 / TARGET_FPS)

def depth_thread():
    global latest_frame, latest_depth
    while running:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (DEPTH_W, DEPTH_H))
        inp = transform(small).to(device)
        with torch.no_grad():
            pred = midas(inp)
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1),
                size=(FRAME_H, FRAME_W), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
        depth = (pred - pred.min()) / (np.ptp(pred) + 1e-9)
        depth = 1.0 - depth
        depth_buf.append(depth)
        avg = np.mean(np.stack(list(depth_buf), axis=0), axis=0)
        with depth_lock:
            latest_depth = avg.astype(np.float32)
        time.sleep(1 / TARGET_FPS)

# --------------------------------------------
# DEPTH HELPERS
# --------------------------------------------
def region_median(depth, x1, x2, y1, y2):
    h, w = depth.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return float(np.median(depth[y1:y2, x1:x2]))

def compute_regions(depth):
    h, w = depth.shape
    cx, cy = w // 2, h // 2
    cw, ch = int(w * CENTER_RECT_W), int(h * CENTER_RECT_H)
    sw, th, bh = int(w * SIDE_ZONE_W), int(h * UP_ZONE_H), int(h * DOWN_ZONE_H)
    center = region_median(depth, cx - cw//2, cx + cw//2, cy - ch//2, cy + ch//2)
    left = region_median(depth, cx - cw//2 - sw, cx - cw//2, cy - ch//2, cy + ch//2)
    right = region_median(depth, cx + cw//2, cx + cw//2 + sw, cy - ch//2, cy + ch//2)
    top = region_median(depth, cx - cw//2//2, cx + cw//2//2, 0, th)
    bottom = region_median(depth, cx - cw//2//2, cx + cw//2//2, h - bh, h)
    return center, left, right, top, bottom

def decide_move(depth):
    SCALE = 120.0
    c, l, r, t, b = [v * SCALE for v in compute_regions(depth)]
    if c < MIN_CLEARANCE_CM:
        back = min(MAX_MOVE_CM, (MIN_CLEARANCE_CM - c) * SAFETY_MARGIN * 1.2)
        return "back", round(back, 1), (c, l, r, t, b)
    diff_lr = r - l
    if abs(diff_lr) > 6:
        move = min(MOVE_STEP_MAX_CM, abs(diff_lr) * 0.6 * SAFETY_MARGIN)
        return ("right" if diff_lr > 0 else "left"), round(move, 1), (c, l, r, t, b)
    diff_v = b - t
    if diff_v > 10:
        return "up", round(min(60.0, diff_v * 0.5 * SAFETY_MARGIN), 1), (c, l, r, t, b)
    if diff_v < -10:
        return "down", round(min(60.0, abs(diff_v) * 0.5 * SAFETY_MARGIN), 1), (c, l, r, t, b)
    if c > MIN_CLEARANCE_CM + 30:
        fwd = min(40.0, (c - (MIN_CLEARANCE_CM + 30)) * 0.4)
        return "forward", round(min(fwd, MOVE_STEP_MAX_CM), 1), (c, l, r, t, b)
    return None, None, (c, l, r, t, b)

def execute_move(direction, cm):
    cm = int(round(cm))
    try:
        getattr(tello, f"move_{direction}")(cm)
        print(f"[AI] move_{direction} {cm}cm")
    except Exception:
        tello.send_rc_control(0, 0, 0, 0)

# --------------------------------------------
# LOGGING
# --------------------------------------------
def log_event(mode, direction, cm, battery, depths):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "mode", "direction", "distance_cm", "battery", "center", "left", "right", "top", "bottom"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), mode, direction or "-", cm or 0, battery, *[round(d, 2) for d in depths]])

# --------------------------------------------
# MAIN LOOP
# --------------------------------------------
def main():
    global running
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=depth_thread, daemon=True).start()
    kpm.init()

    ai_mode, started = False, False
    backtrack, move_in_progress = [], False
    last_move_time, clear_frames = 0.0, 0

    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Depth-AI Cockpit (Graph Integrated)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Segoe UI", 18)

    print("[INFO] Controls: SPACE=Takeoff | M=Toggle | Arrows=Move | W/S=Up/Down | A/D=Rotate | B=Backtrack | Q=Land")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                try: tello.land()
                except: pass
                pygame.quit()
                return

        if kpm.get_key("q") or kpm.get_key("ESCAPE"):
            print("[EXIT] Landing...")
            try: tello.land()
            except: pass
            running = False
            break

        if not started and kpm.get_key("SPACE"):
            tello.takeoff()
            started = True
            print("[TAKEOFF]")
            time.sleep(1)

        if kpm.get_key("m"):
            ai_mode = not ai_mode
            print("[MODE]", "AI" if ai_mode else "Manual")
            time.sleep(0.25)

        # manual movement
        lr = fb = ud = yaw = 0
        if kpm.get_key("LEFT"): lr = -SPEED
        elif kpm.get_key("RIGHT"): lr = SPEED
        if kpm.get_key("UP"): fb = SPEED
        elif kpm.get_key("DOWN"): fb = -SPEED
        if kpm.get_key("w"): ud = SPEED
        elif kpm.get_key("s"): ud = -SPEED
        if kpm.get_key("a"): yaw = -SPEED
        elif kpm.get_key("d"): yaw = SPEED
        tello.send_rc_control(lr, fb, ud, yaw)

        with frame_lock: frame = latest_frame.copy() if latest_frame is not None else None
        with depth_lock: depth = latest_depth.copy() if latest_depth is not None else None
        if frame is None or depth is None:
            pygame.display.flip()
            clock.tick(30)
            continue

        if ai_mode and started:
            if not move_in_progress or (time.time() - last_move_time) > DECISION_INTERVAL:
                direction, cm, depths = decide_move(depth)
                battery = tello.get_battery()
                depth_history.append(depths[0])
                if direction:
                    execute_move(direction, cm)
                    backtrack.append((direction, cm))
                    if len(backtrack) > BACKTRACK_MAX: backtrack = backtrack[-BACKTRACK_MAX:]
                    move_in_progress = True
                    last_move_time = time.time()
                    log_event("AI", direction, cm, battery, depths)
                else:
                    log_event("AI_idle", None, None, battery, depths)
        else:
            depths = compute_regions(depth)
            depth_history.append(depths[0])
            log_event("Manual", None, None, tello.get_battery(), depths)

        # Combine RGB + depth
        depth_vis = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        combo = np.hstack((frame, depth_vis))

        # Depth graph overlay
        if len(depth_history) > 2:
            graph_w, graph_h = 240, 100
            plot_img = np.zeros((graph_h, graph_w, 3), dtype=np.uint8)
            for i in range(1, len(depth_history)):
                val = depth_history[i]
                color = (0, 255, 0) if val > 0.7 else (0, 255, 255) if val > 0.4 else (0, 0, 255)
                x1 = int((i - 1) * graph_w / GRAPH_HISTORY)
                x2 = int(i * graph_w / GRAPH_HISTORY)
                y1 = graph_h - int(depth_history[i - 1] * graph_h)
                y2 = graph_h - int(val * graph_h)
                cv2.line(plot_img, (x1, y1), (x2, y2), color, 1)
            combo[10:10 + graph_h, combo.shape[1] - graph_w - 10:combo.shape[1] - 10] = plot_img

        frame_rgb = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        window.blit(surf, (0, 0))

        # HUD text
        color = (0, 255, 0) if ai_mode else (255, 215, 0)
        text = font.render(f"{'AI MODE' if ai_mode else 'MANUAL MODE'} | Battery: {tello.get_battery()}%", True, color)
        window.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    cv2.destroyAllWindows()
    tello.streamoff()
    print("[DONE] Program terminated.")
    print(f"[LOG] Telemetry saved to {LOG_FILE}")

# --------------------------------------------
if __name__ == "__main__":
    main()
