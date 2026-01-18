from djitellopy import Tello, TelloException
from ultralytics import YOLO
import cv2
import time
from time import sleep
import os
import joblib
import numpy as np
import pandas as pd
import warnings
import KeyPressModule as kpm
from sklearn.exceptions import InconsistentVersionWarning

# ------------------- SETTINGS ------------------- #
MODEL_PATH = r"Models\yolov8n.pt"
AI_MODEL_PATH = r"Models\best_model.pkl"
SAVE_PATH = r"data\captures"
CONF_THRESHOLD = 0.5
SPEED = 50  # Manual mode speed
TARGET_CLASS_NAME = "person"
CENTER_TOLERANCE = 0.22
MIN_BBOX_HEIGHT_PX = 100
MOVE_DISTANCE_CM = 25
DELAY_AFTER_MOVE = 2.5
FRAME_SAMPLE_N = 10
SAFE_MIN_HEIGHT_CM = 40
SAFE_MAX_HEIGHT_CM = 140
MIN_BATTERY_FOR_TAKEOFF = 30
PROBE_CLEAR_FRAMES = 3
PROBE_FRAME_DELAY = 0.08
MIN_CLASS_PROB = 0.0
# ------------------------------------------------ #

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
os.makedirs(SAVE_PATH, exist_ok=True)

FEATURE_NAMES = [
    "pitch_deg", "roll_deg", "yaw_deg", "tof_cm", "h_cm",
    "vgx_cm_s", "vgy_cm_s", "vgz_cm_s",
    "bbox_u_px", "bbox_v_px", "bbox_w_px", "bbox_h_px",
    "depth_cm", "drone_x_cm", "drone_y_cm"
]

# =============== YOLO DETECTION =============== #
def run_detect(model_yolo, frame, target_class_id):
    results = model_yolo.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()
    obstacle_detected = False
    bbox_u_px = bbox_v_px = bbox_w_px = bbox_h_px = 0.0
    conf_score = 0.0
    boxes = results[0].boxes
    if len(boxes) > 0:
        boxes_xywh = boxes.xywh.cpu().numpy()
        boxes_conf = boxes.conf.cpu().numpy()
        boxes_cls = boxes.cls.cpu().numpy().astype(int)
        candidates = [
            (xywh, conf) for xywh, conf, cls in zip(boxes_xywh, boxes_conf, boxes_cls)
            if conf >= CONF_THRESHOLD and (target_class_id is None or cls == target_class_id)
        ]
        if candidates:
            (bbox_u_px, bbox_v_px, bbox_w_px, bbox_h_px), conf_score = max(
                candidates, key=lambda c: c[0][2] * c[0][3]
            )
            h, w = annotated.shape[:2]
            dx = abs(bbox_u_px - w / 2) / w
            dy = abs(bbox_v_px - h / 2) / h
            if bbox_h_px >= MIN_BBOX_HEIGHT_PX and dx <= CENTER_TOLERANCE and dy <= CENTER_TOLERANCE:
                obstacle_detected = True
    return annotated, obstacle_detected, (bbox_u_px, bbox_v_px, bbox_w_px, bbox_h_px), conf_score

# =============== AI FEATURE BUILDING =============== #
def build_features(drone, bbox, depth_cm):
    bbox_u_px, bbox_v_px, bbox_w_px, bbox_h_px = bbox
    pitch_deg = drone.get_pitch()
    roll_deg = drone.get_roll()
    yaw_deg = drone.get_yaw()
    tof_cm = drone.get_distance_tof()
    h_cm = drone.get_height()
    vgx_cm_s = drone.get_speed_x()
    vgy_cm_s = drone.get_speed_y()
    vgz_cm_s = drone.get_speed_z()
    depth = tof_cm if depth_cm is None else depth_cm
    drone_x_cm = 0.0
    drone_y_cm = 0.0
    return [
        pitch_deg, roll_deg, yaw_deg, tof_cm, h_cm,
        vgx_cm_s, vgy_cm_s, vgz_cm_s,
        float(bbox_u_px), float(bbox_v_px), float(bbox_w_px), float(bbox_h_px),
        depth, drone_x_cm, drone_y_cm
    ]

# =============== SENSOR HEALTH CHECK =============== #
def sensors_healthy(drone):
    try:
        yaw = drone.get_yaw()
        pitch = drone.get_pitch()
        roll = drone.get_roll()
        h = drone.get_height()
        return all(v is not None for v in [yaw, pitch, roll, h])
    except:
        return False

# =============== SAFE ROTATION =============== #
def safe_rotate(drone, direction, deg):
    if not sensors_healthy(drone):
        print("[WARN] IMU/state not healthy; skipping rotation.")
        return False
    try:
        if direction == "ccw":
            drone.rotate_counter_clockwise(deg)
        else:
            drone.rotate_clockwise(deg)
        return True
    except TelloException as te:
        print(f"[WARN] Rotation {direction} {deg} failed: {te}")
        return False

# =============== PROBE SIDE =============== #
def probe_side_multi(drone, model_yolo, target_class_id, face_dir):
    if face_dir not in ("left", "right"):
        raise ValueError("probe_side_multi only for 'left'/'right'")
    first = "ccw" if face_dir == "left" else "cw"
    back = "cw" if face_dir == "left" else "ccw"
    if not safe_rotate(drone, first, 90):
        sleep(0.6)
        return False
    clear_count = 0
    try:
        for _ in range(PROBE_CLEAR_FRAMES):
            frame_probe = cv2.resize(drone.get_frame_read().frame, (640, 480))
            _, obs2, _, _ = run_detect(model_yolo, frame_probe, target_class_id)
            if not obs2:
                clear_count += 1
            else:
                clear_count = 0
            if clear_count >= PROBE_CLEAR_FRAMES:
                break
            sleep(PROBE_FRAME_DELAY)
    except Exception as e:
        print(f"[WARN] Probe detect ({face_dir}) failed: {e}")
        clear_count = 0
    finally:
        if not safe_rotate(drone, back, 90):
            print("[WARN] Restore heading failed; pausing 2s.")
            sleep(2.0)
        else:
            sleep(0.3)
    return clear_count >= PROBE_CLEAR_FRAMES

# =============== PROBE DOWN =============== #
def probe_down_safe(drone):
    try:
        h_cm_now = drone.get_height()
        if h_cm_now is None or h_cm_now <= SAFE_MIN_HEIGHT_CM:
            return False
        return True
    except Exception as e:
        print(f"[WARN] Probe height failed: {e}")
        return False

# =============== QUICK FRONT CHECK =============== #
def quick_front_check(model_yolo, frame, target_class_id):
    try:
        _, obs, _, _ = run_detect(model_yolo, frame, target_class_id)
        return not obs
    except:
        return True

# =============== TRY MOVE =============== #
def try_move(drone, direction):
    if not sensors_healthy(drone):
        print("[WARN] Sensors unhealthy; skipping move.")
        return False
    try:
        if direction == "left":
            drone.move_left(MOVE_DISTANCE_CM)
        elif direction == "right":
            drone.move_right(MOVE_DISTANCE_CM)
        elif direction == "down":
            h_cm_now = drone.get_height()
            if h_cm_now is None or h_cm_now <= SAFE_MIN_HEIGHT_CM:
                print(f"[WARN] Too low to descend; skip.")
                return False
            drone.move_down(MOVE_DISTANCE_CM)
        elif direction == "up":
            drone.move_up(MOVE_DISTANCE_CM)
        else:
            return False
        return True
    except TelloException as te:
        print(f"[WARN] Move '{direction} {MOVE_DISTANCE_CM}' rejected: {te}")
        return False

# =============== AI PREDICTION =============== #
def choose_label_excluding_top(ai_model, X):
    safe_labels = ("left", "right", "bottom")
    if hasattr(ai_model, "predict_proba") and hasattr(ai_model, "classes_"):
        proba = ai_model.predict_proba(X)[0]
        classes = np.array(ai_model.classes_)
        safe_probs = {lbl: float(proba[idx]) for idx, lbl in enumerate(classes) if lbl in safe_labels}
        if safe_probs:
            return max(safe_probs.items(), key=lambda kv: kv[1])[0]
        return "right"
    else:
        pred_label = str(ai_model.predict(X)[0]).lower()
        return pred_label if pred_label in safe_labels else "right"

def ranked_candidates(ai_model, X, primary_label):
    safe_labels = ("left", "right", "bottom")
    order = []
    if primary_label in safe_labels:
        order.append(primary_label)
    if hasattr(ai_model, "predict_proba") and hasattr(ai_model, "classes_"):
        proba = ai_model.predict_proba(X)[0]
        classes = np.array(ai_model.classes_)
        label2prob = {lbl: float(proba[idx]) for idx, lbl in enumerate(classes)}
        rest = [lbl for lbl in safe_labels if lbl not in order]
        rest_sorted = sorted(rest, key=lambda l: label2prob.get(l, -1.0), reverse=True)
        order.extend(rest_sorted)
    for lbl in safe_labels:
        if lbl not in order:
            order.append(lbl)
    return order

# =============== MANUAL MODE INPUT =============== #
def get_keyboard_input(drone):
    """Reads keypresses and returns movement values for manual mode."""
    lr, fb, ud, yaw = 0, 0, 0, 0

    try:
        # LEFT / RIGHT
        if kpm.get_key("RIGHT"):
            lr = SPEED
        elif kpm.get_key("LEFT"):
            lr = -SPEED

        # FORWARD / BACKWARD
        if kpm.get_key("UP"):
            fb = SPEED
        elif kpm.get_key("DOWN"):
            fb = -SPEED

        # UP / DOWN
        if kpm.get_key("w"):
            ud = SPEED
        elif kpm.get_key("s"):
            ud = -SPEED

        # YAW
        if kpm.get_key("a"):
            yaw = -SPEED
        elif kpm.get_key("d"):
            yaw = SPEED
    except (AttributeError, TypeError) as e:
        # Handle keyboard input errors gracefully
        pass

    return (lr, fb, ud, yaw)

# =============== MAIN FUNCTION =============== #
def main():
    kpm.init()

    drone = Tello()
    drone.connect()
    battery = drone.get_battery()
    print(f"[INFO] Battery: {battery}%")
    if battery is None or battery < MIN_BATTERY_FOR_TAKEOFF:
        print(f"[ABORT] Battery too low for reliable flight; charge above {MIN_BATTERY_FOR_TAKEOFF}% and retry.")
        return

    drone.streamon()
    print("[INFO] Video stream started")

    print("[INIT] Loading YOLO model...")
    model_yolo = YOLO(MODEL_PATH)
    print("[INFO] YOLO model loaded ✅")

    target_class_id = None
    for k, v in model_yolo.names.items():
        if v == TARGET_CLASS_NAME:
            target_class_id = int(k)
            break
    if target_class_id is None:
        print(f"[WARNING] Target class '{TARGET_CLASS_NAME}' not found in YOLO model!")

    try:
        ai_model = joblib.load(AI_MODEL_PATH)
        print("[INFO] AI model loaded ✅")
    except Exception as e:
        print(f"[ERROR] Failed to load AI model: {e}")
        ai_model = None

    try:
        print("[INFO] Taking off...")
        drone.takeoff()
        sleep(2)
    except TelloException as e:
        print(f"[ERROR] Takeoff failed: {e}")
        drone.streamoff()
        cv2.destroyAllWindows()
        return

    # Mode control
    ai_mode = True  # Start in AI mode
    last_move_time = 0.0
    move_in_progress = False
    frame_idx = 0

    print("\n" + "="*50)
    print("CONTROLS:")
    print("  M - Toggle between AI and Manual Mode")
    print("  C - Capture image")
    print("  Q - Land drone")
    print("  ESC - Emergency land and exit")
    print("\nMANUAL MODE CONTROLS:")
    print("  Arrow Keys - Left/Right/Forward/Back")
    print("  W/S - Up/Down")
    print("  A/D - Rotate Left/Right")
    print("\nNOTE: Make sure the Pygame window is focused for manual controls!")
    print("="*50 + "\n")

    try:
        while True:
            frame = drone.get_frame_read().frame
            frame = cv2.resize(frame, (640, 480))
            frame_idx += 1

            annotated_frame = frame.copy()
            obstacle_detected = False
            bbox = (0.0, 0.0, 0.0, 0.0)

            # Check for mode toggle
            try:
                if kpm.get_key("m"):
                    ai_mode = not ai_mode
                    mode_name = "AI MODE" if ai_mode else "MANUAL MODE"
                    print(f"\n[MODE] Switched to {mode_name}")
                    if not ai_mode:
                        print("[MANUAL] Now accepting keyboard controls. Focus the Pygame window!")
                    print()
                    sleep(0.3)  # Debounce

                # Check for land command
                if kpm.get_key("q"):
                    print("[INFO] Landing (Q pressed)...")
                    drone.land()
                    break
            except (AttributeError, TypeError):
                pass  # Ignore keyboard errors

            # ========== AI MODE ========== #
            if ai_mode:
                do_heavy = (frame_idx % FRAME_SAMPLE_N == 0)

                if do_heavy:
                    annotated_frame, obstacle_detected, bbox, conf_score = run_detect(
                        model_yolo, frame, target_class_id
                    )
                    if obstacle_detected:
                        print(f"[YOLO] Detected {TARGET_CLASS_NAME} (conf={conf_score:.2f}) Centered=True")

                if move_in_progress and (time.time() - last_move_time) <= DELAY_AFTER_MOVE:
                    remaining = max(0.0, DELAY_AFTER_MOVE - (time.time() - last_move_time))
                    cv2.putText(annotated_frame, f"Cooldown: {remaining:.1f}s", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                elif move_in_progress and (time.time() - last_move_time) > DELAY_AFTER_MOVE:
                    move_in_progress = False
                    print("[AI] Movement cooldown complete – resuming detection.")

                if do_heavy and ai_model and obstacle_detected and not move_in_progress:
                    print("[AI] Obstacle detected! Evaluating move...")

                    feats = build_features(drone, bbox, depth_cm=None)
                    X = pd.DataFrame([feats], columns=FEATURE_NAMES)
                    primary_label = choose_label_excluding_top(ai_model, X)

                    # Debug probabilities
                    if hasattr(ai_model, "predict_proba") and hasattr(ai_model, "classes_"):
                        proba = ai_model.predict_proba(X)[0]
                        classes = np.array(ai_model.classes_)
                        label2prob = {lbl: float(proba[idx]) for idx, lbl in enumerate(classes)}
                        print("[AI Debug] P(bottom)={:.3f} P(left)={:.3f} P(right)={:.3f}".format(
                            label2prob.get("bottom", -1.0),
                            label2prob.get("left", -1.0),
                            label2prob.get("right", -1.0)
                        ))

                    print(f"[AI] Primary label (no 'top'): {primary_label}")
                    candidates = ranked_candidates(ai_model, X, primary_label)
                    print(f"[AI] Candidates (no 'top'): {candidates}")

                    moved = False
                    for cand in candidates:
                        if moved:
                            break
                        direction = "down" if cand == "bottom" else cand

                        # Optional confidence gate
                        if MIN_CLASS_PROB > 0 and hasattr(ai_model, "predict_proba") and hasattr(ai_model, "classes_"):
                            proba = ai_model.predict_proba(X)[0]
                            classes = np.array(ai_model.classes_)
                            label2prob = {lbl: float(proba[idx]) for idx, lbl in enumerate(classes)}
                            cand_prob = label2prob.get(cand, 0.0)
                            if cand_prob < MIN_CLASS_PROB:
                                print(f"[AI] {cand} prob {cand_prob:.2f} below threshold; trying next.")
                                continue

                        if direction == "left":
                            clear = probe_side_multi(drone, model_yolo, target_class_id, "left")
                        elif direction == "right":
                            clear = probe_side_multi(drone, model_yolo, target_class_id, "right")
                        elif direction == "down":
                            clear = probe_down_safe(drone)
                        else:
                            clear = False

                        if clear:
                            frame_front = cv2.resize(drone.get_frame_read().frame, (640, 480))
                            if quick_front_check(model_yolo, frame_front, target_class_id):
                                moved = try_move(drone, direction)
                                if moved:
                                    print(f"[ACTION] Moved {direction.upper()} by {MOVE_DISTANCE_CM}cm ✅")
                                else:
                                    print(f"[INFO] Move {direction} attempted but not executed; trying next.")
                            else:
                                print("[PROBE] Front recheck blocked; trying next.")
                        else:
                            print(f"[PROBE] {direction.capitalize()} blocked; trying next.")

                    if not moved:
                        if sensors_healthy(drone):
                            h_cm_now = drone.get_height()
                            if h_cm_now is None or h_cm_now < SAFE_MAX_HEIGHT_CM:
                                print("[FALLBACK] All directions blocked; attempting UP once.")
                                moved = try_move(drone, "up")
                            else:
                                print("[FALLBACK] At/above safe max height; skip UP.")
                        else:
                            print("[FALLBACK] Sensors unhealthy; skip UP.")

                    if moved:
                        last_move_time = time.time()
                        move_in_progress = True
                        print(f"[COOLDOWN] Waiting {DELAY_AFTER_MOVE:.1f}s before next command...")
                    else:
                        print("[AI] No safe movement executed; holding position.")

            # ========== MANUAL MODE ========== #
            else:
                # Run YOLO detection for display only
                try:
                    annotated_frame, _, _, _ = run_detect(model_yolo, frame, target_class_id)
                except:
                    annotated_frame = frame.copy()
                
                # Get manual input and send to drone
                movement = get_keyboard_input(drone)
                if movement:
                    drone.send_rc_control(*movement)

            # Display mode indicator
            mode_text = "AI MODE ACTIVE" if ai_mode else "MANUAL MODE ACTIVE"
            mode_color = (0, 255, 0) if ai_mode else (0, 165, 255)
            cv2.putText(annotated_frame, mode_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            
            # Display toggle hint
            cv2.putText(annotated_frame, "Press 'M' to toggle mode", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show manual controls reminder when in manual mode
            if not ai_mode:
                cv2.putText(annotated_frame, "Use Arrow Keys + W/S/A/D to control", (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Tello Drone - AI + Manual Mode", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            # Capture image
            if key == ord('c'):
                filename = os.path.join(SAVE_PATH, f"capture_{int(time.time())}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[INFO] Image saved: {filename}")

            # Emergency exit
            if key == 27:  # ESC
                print("[INFO] Landing and closing...")
                try:
                    drone.land()
                except Exception as e:
                    print(f"[WARN] Land failed: {e}")
                break

            sleep(0.02)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt – landing safely...")
        try:
            drone.land()
        except Exception as e:
            print(f"[WARN] Land failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        try:
            drone.land()
        except:
            pass
    finally:
        try:
            drone.streamoff()
        except:
            pass
        cv2.destroyAllWindows()
        print("[INFO] Program ended safely ✅")

if __name__ == "__main__":
    main()