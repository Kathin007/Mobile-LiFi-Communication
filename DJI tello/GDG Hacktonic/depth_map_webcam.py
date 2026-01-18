import sys
import cv2
import torch
import numpy as np
import warnings
from torchvision.transforms import Compose
from queue import Queue
import threading
from time import time

# -------- OPTIONAL: fix Unicode output on Windows -------- #
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

# -------- OPENCV OPTIMIZATION -------- #
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# -------- DEVICE SETUP -------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# -------- LOAD LIGHTWEIGHT MiDaS -------- #
print("Loading MiDaS_small model for real-time depth...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
midas.to(device).eval()
print("✅ MiDaS_small loaded successfully.")

# -------- CAMERA + DEPTH THREADING -------- #
frame_queue = Queue(maxsize=2)
depth_queue = Queue(maxsize=2)
stop_signal = False

# 5-frame smoothing buffer
depth_buffer = []
BUFFER_SIZE = 5

def camera_reader():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return
    while not stop_signal:
        ret, frame = cap.read()
        if ret and not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

def depth_processor():
    global depth_buffer
    frame_id = 0
    while not stop_signal:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        frame_id += 1

        # Skip every other frame for speed
        if frame_id % 2 != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (320, 240))  # downscale for speed
        input_batch = transform(rgb).to(device)

        with torch.no_grad():
            start = time()
            pred = midas(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(480, 640),  # upsample to match display
                mode="bicubic",
                align_corners=False
            ).squeeze()
            depth_np = pred.cpu().numpy()
            fps = 1.0 / (time() - start + 1e-6)

        # Normalize and colorize
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)

        # ---- 5-Frame Temporal Smoothing ---- #
        depth_buffer.append(depth_color)
        if len(depth_buffer) > BUFFER_SIZE:
            depth_buffer.pop(0)
        avg_depth = np.mean(depth_buffer, axis=0).astype(np.uint8)

        combined = np.hstack((frame, avg_depth))
        cv2.putText(combined, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if not depth_queue.full():
            depth_queue.put(combined)

# -------- START THREADS -------- #
camera_thread = threading.Thread(target=camera_reader, daemon=True)
depth_thread = threading.Thread(target=depth_processor, daemon=True)
camera_thread.start()
depth_thread.start()

print("🎥 Running Dual Feed: [Left = Original | Right = Smoothed Depth]")
print("Press 'q' to quit.")

# -------- MAIN DISPLAY LOOP -------- #
while True:
    if not depth_queue.empty():
        frame = depth_queue.get()
        cv2.imshow("MonoSight Depth (Optimized)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_signal = True
        break

cv2.destroyAllWindows()
print("🛑 Exiting MonoSight...")
