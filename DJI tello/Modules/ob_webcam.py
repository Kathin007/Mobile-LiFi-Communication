from ultralytics import YOLO
import cv2
import time
import joblib
import numpy as np
import pandas as pd
import warnings
import random
from sklearn.exceptions import InconsistentVersionWarning

# ------------------- SETTINGS ------------------- #
MODEL_PATH = r"Models\yolov8n.pt"
AI_MODEL_PATH = r"Models\best_model.pkl"
CONF_THRESHOLD = 0.5
TARGET_CLASS_NAME = "person"        # Change this if needed
CENTER_TOLERANCE = 0.18             # Tolerance for centered obstacle
MIN_BBOX_HEIGHT_PX = 120            # Min height to consider obstacle close

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ------------------- FEATURE NAMES ------------------- #
FEATURE_NAMES = [
    "pitch_deg", "roll_deg", "yaw_deg", "tof_cm", "h_cm",
    "vgx_cm_s", "vgy_cm_s", "vgz_cm_s",
    "bbox_u_px", "bbox_v_px", "bbox_w_px", "bbox_h_px",
    "depth_cm", "drone_x_cm", "drone_y_cm", "drone_z_cm"
]


# ------------------- MAIN PROGRAM ------------------- #
def main():
    print("[INIT] Loading YOLO model...")
    model_yolo = YOLO(MODEL_PATH)

    print("[INIT] Loading AI model...")
    try:
        ai_model = joblib.load(AI_MODEL_PATH)
        print("[INFO] Models loaded successfully!\n")
    except Exception as e:
        print(f"[ERROR] Could not load AI model: {e}")
        return

    # Identify target class ID
    target_class_id = None
    for k, v in model_yolo.names.items():
        if v == TARGET_CLASS_NAME:
            target_class_id = int(k)
            break
    if target_class_id is None:
        print(f"[WARNING] Target class '{TARGET_CLASS_NAME}' not found in model names.")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    print("[INFO] Webcam feed started.")
    print("[INFO] Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame capture failed.")
                break

            frame = cv2.resize(frame, (640, 480))
            results = model_yolo.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            annotated_frame = results[0].plot()

            # ---------- OBSTACLE DETECTION ---------- #
            obstacle_detected = False
            bbox_u_px = bbox_v_px = bbox_w_px = bbox_h_px = 0

            boxes = results[0].boxes
            if len(boxes) > 0:
                boxes_xywh = boxes.xywh.cpu().numpy()
                boxes_conf = boxes.conf.cpu().numpy()
                boxes_cls = boxes.cls.cpu().numpy().astype(int)

                candidates = []
                for (xywh, conf, cls) in zip(boxes_xywh, boxes_conf, boxes_cls):
                    if conf < CONF_THRESHOLD:
                        continue
                    if target_class_id is not None and cls != target_class_id:
                        continue
                    candidates.append((xywh, conf))

                if len(candidates) > 0:
                    largest = max(candidates, key=lambda c: c[0][2] * c[0][3])
                    (bbox_u_px, bbox_v_px, bbox_w_px, bbox_h_px), conf_score = largest

                    frame_h, frame_w = annotated_frame.shape[:2]
                    img_center_x = frame_w / 2
                    img_center_y = frame_h / 2
                    dx = abs(bbox_u_px - img_center_x) / frame_w
                    dy = abs(bbox_v_px - img_center_y) / frame_h

                    if bbox_h_px >= MIN_BBOX_HEIGHT_PX and dx <= CENTER_TOLERANCE and dy <= CENTER_TOLERANCE:
                        obstacle_detected = True

                    # Draw helper visuals
                    x1 = int(bbox_u_px - bbox_w_px / 2)
                    y1 = int(bbox_v_px - bbox_h_px / 2)
                    x2 = int(bbox_u_px + bbox_w_px / 2)
                    y2 = int(bbox_v_px + bbox_h_px / 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(annotated_frame, (int(bbox_u_px), int(bbox_v_px)), 5, (0, 255, 0), -1)

                    # Draw tolerance box around center
                    cx1 = int(img_center_x - CENTER_TOLERANCE * frame_w)
                    cy1 = int(img_center_y - CENTER_TOLERANCE * frame_h)
                    cx2 = int(img_center_x + CENTER_TOLERANCE * frame_w)
                    cy2 = int(img_center_y + CENTER_TOLERANCE * frame_h)
                    cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

                    print(f"[YOLO] {TARGET_CLASS_NAME}: conf={conf_score:.2f}, centered={obstacle_detected}")

            # ---------- AI DECISION ---------- #
            ai_decision = "None"
            if obstacle_detected:
                print("[AI] Obstacle detected! Evaluating movement...")

                # Simulated telemetry values
                pitch_deg = random.uniform(-5, 5)
                roll_deg = random.uniform(-5, 5)
                yaw_deg = random.uniform(-180, 180)
                tof_cm = random.uniform(30, 200)
                h_cm = random.uniform(50, 150)
                vgx_cm_s = random.uniform(-20, 20)
                vgy_cm_s = random.uniform(-20, 20)
                vgz_cm_s = random.uniform(-10, 10)
                depth_cm = tof_cm
                drone_x_cm = random.uniform(-10, 10)
                drone_y_cm = random.uniform(-10, 10)
                drone_z_cm = random.uniform(-10, 10)

                features = [
                    pitch_deg, roll_deg, yaw_deg, tof_cm, h_cm,
                    vgx_cm_s, vgy_cm_s, vgz_cm_s,
                    bbox_u_px, bbox_v_px, bbox_w_px, bbox_h_px,
                    depth_cm, drone_x_cm, drone_y_cm, drone_z_cm
                ]

                X = pd.DataFrame([features], columns=FEATURE_NAMES)

                try:
                    prediction = ai_model.predict(X)[0]
                    ai_decision = str(prediction).lower()
                    print(f"[AI Decision] Move: {ai_decision.upper()}")

                    if ai_decision == "left":
                        print("[ACTION] Simulated: Move LEFT ⬅️")
                    elif ai_decision == "right":
                        print("[ACTION] Simulated: Move RIGHT ➡️")
                    else:
                        print(f"[ACTION] Simulated: Stay/Unknown ({ai_decision})")

                except Exception as e:
                    print(f"[ERROR] Prediction failed: {e}")

            # ---------- DISPLAY OVERLAYS ---------- #
            if obstacle_detected:
                cv2.putText(annotated_frame, "⚠️ CENTERED OBSTACLE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

            if ai_decision != "None":
                cv2.putText(annotated_frame, f"AI Decision: {ai_decision.upper()}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow("YOLO + AI Vision Test", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quitting...")
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — exiting...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Program ended safely ✅")


# ------------------- RUN ------------------- #
if __name__ == "__main__":
    main()
