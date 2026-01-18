import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque
import plotly.graph_objects as go

# Try to import YOLO, fallback to simulation if not present
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# ==================== AI DECISION ENGINE ====================
class AIDecisionEngine:
    def __init__(self):
        if HAS_YOLO:
            # Loads small model for speed; will download on first run
            self.model = YOLO('yolov8n.pt') 
        self.safe_zone_threshold = 0.25 # 25% from center

    def process_frame(self, frame, ai_active):
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        decision = "STABLE"
        detection_data = []

        if not ai_active:
            return frame, "MANUAL OVERRIDE", []

        if HAS_YOLO:
            results = self.model(frame, conf=0.4, verbose=False)
            for box in results[0].boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                conf = float(box.conf[0])
                
                # Draw bounding box
                color = (0, 255, 102) if label != 'person' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Logic for Decision
                obj_center_x = (x1 + x2) // 2
                if abs(obj_center_x - center_x) < (w * self.safe_zone_threshold):
                    decision = "EVADE: " + ("LEFT" if obj_center_x > center_x else "RIGHT")
                
                detection_data.append(f"{label.upper()} DETECTED")
        else:
            # Simulated Detection if YOLO is missing
            if np.random.random() > 0.95:
                decision = "EVADE: OBSTACLE DETECTED"
                cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 255), 2)
                detection_data.append("SIM_OBSTACLE")

        return frame, decision, detection_data

# ==================== UTILS ====================
MORSE_CODE_DICT = {'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----', ' ': '/'}
REVERSE_MORSE = {v: k for k, v in MORSE_CODE_DICT.items()}

def encode_morse(text): return ' '.join(MORSE_CODE_DICT.get(c.upper(), '?') for c in text)
def decode_morse(morse):
    try:
        return ' '.join([''.join([REVERSE_MORSE.get(l, '') for l in w.split()]) for w in morse.split(' / ')])
    except: return "INVALID"

# ==================== STYLING ====================
def apply_tactical_theme():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500&family=JetBrains+Mono:wght@200;400&display=swap');
        :root { --primary: #00FF66; --bg: #050505; --card-bg: #0D0D0D; }
        .stApp { background-color: var(--bg); color: #E0E0E0; font-family: 'Space Grotesk', sans-serif; }
        .terminal-text { font-family: 'JetBrains Mono', monospace; color: var(--primary); font-size: 0.85rem; }
        .hud-card { background: var(--card-bg); border: 1px solid #1A1A1A; padding: 20px; position: relative; margin-bottom: 1rem; }
        .hud-card::before { content: ""; position: absolute; top: -1px; left: -1px; width: 10px; height: 10px; border-top: 2px solid var(--primary); border-left: 2px solid var(--primary); }
        [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: var(--primary) !important; }
        .stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important; }
        </style>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    st.set_page_config(page_title="NEUROMINDS COMMAND", layout="wide")
    apply_tactical_theme()
    
    # Init Session State
    if 'ai_log' not in st.session_state: st.session_state.ai_log = deque(maxlen=10)
    if 'engine' not in st.session_state: st.session_state.engine = AIDecisionEngine()

    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color:#00FF66;'>NEUROMINDS v4.1</h2>", unsafe_allow_html=True)
        st.divider()
        ai_toggle = st.toggle("AI OVERRIDE / DETECTION", value=True)
        feed_toggle = st.toggle("OPTICAL FEED", value=False)
        if st.button("💾 EXPORT MISSION DATA"):
            st.download_button("Download Log", "\n".join(st.session_state.ai_log), "mission.txt")

    tab1, tab2, tab3 = st.tabs(["[ 01 COMMAND ]", "[ 02 ANALYTICS ]", "[ 03 SECURE COMM ]"])

    with tab1:
        col_vid, col_stats = st.columns([3, 1])
        
        with col_vid:
            st.markdown('<div class="hud-card">', unsafe_allow_html=True)
            video_placeholder = st.empty()
            
            if feed_toggle:
                cap = cv2.VideoCapture(0)
                while feed_toggle:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    
                    # AI Processing
                    processed_frame, decision, detections = st.session_state.engine.process_frame(frame, ai_toggle)
                    
                    # Update Log if something new happens
                    if decision != "STABLE" and decision != "MANUAL OVERRIDE":
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.ai_log.appendleft(f"[{timestamp}] {decision}")

                    # Crosshair Overlay
                    h, w, _ = processed_frame.shape
                    cv2.line(processed_frame, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 102), 1)
                    cv2.line(processed_frame, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 102), 1)
                    
                    video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                    time.sleep(0.01)
                cap.release()
            else:
                video_placeholder.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1000")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_stats:
            st.markdown('<div class="hud-card">', unsafe_allow_html=True)
            st.write("AI DECISION LOG")
            for entry in list(st.session_state.ai_log):
                st.markdown(f'<p class="terminal-text" style="margin:0;">{entry}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="hud-card">', unsafe_allow_html=True)
            st.metric("BATT", "91%", "-3%")
            st.metric("ALT", "22.1M")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="hud-card">', unsafe_allow_html=True)
        st.write("MISSION TRAJECTORY (3D)")
        # Real-time trajectory simulation
        z = np.linspace(0, 10, 100)
        x = np.sin(z) * z
        y = np.cos(z) * z
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='#00FF66', width=5))])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        # (Morse code section remains the same as your functional version)
        st.markdown("### LiFi REAL-TIME ENCRYPTION")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="hud-card">', unsafe_allow_html=True)
            txt = st.text_input("PLAIN TEXT ENCODER")
            if txt: st.code(encode_morse(txt))
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="hud-card">', unsafe_allow_html=True)
            m_in = st.text_input("MORSE DECODER")
            if m_in: st.success(decode_morse(m_in))
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()