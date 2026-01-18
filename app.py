import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Try to import YOLO, fallback to simulation if not present
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# RTC Configuration for STUN servers (helps connection through firewalls)
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ==================== AI DECISION ENGINE ====================
class AIDecisionEngine:
    def __init__(self):
        if HAS_YOLO:
            self.model = YOLO('yolov8n.pt') 
        self.safe_zone_threshold = 0.25 

    def process_frame(self, frame, ai_active):
        h, w, _ = frame.shape
        center_x = w // 2
        decision = "STABLE"
        
        if not ai_active:
            return frame, "MANUAL OVERRIDE"

        if HAS_YOLO:
            results = self.model(frame, conf=0.4, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls[0])]
                
                # Visuals
                color = (0, 255, 102) if label != 'person' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Logic
                obj_center_x = (x1 + x2) // 2
                if abs(obj_center_x - center_x) < (w * self.safe_zone_threshold):
                    decision = "EVADE: " + ("LEFT" if obj_center_x > center_x else "RIGHT")
        
        # Crosshair Overlay
        cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 102), 1)
        cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 102), 1)
        
        return frame, decision

# ==================== WEBRTC TRANSFORMER ====================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.engine = AIDecisionEngine()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror for user comfort
        
        # Access streamlit state to check if AI is toggled
        ai_active = st.session_state.get("ai_active", True)
        
        processed_img, decision = self.engine.process_frame(img, ai_active)
        
        # Log decision to session state via a callback or shared queue if needed, 
        # but for simplicity we'll just display it on the frame
        cv2.putText(processed_img, f"STATUS: {decision}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 102), 2)
        
        return processed_img

# ==================== UTILS & STYLING ====================
MORSE_CODE_DICT = {'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..', '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----', ' ': '/'}
REVERSE_MORSE = {v: k for k, v in MORSE_CODE_DICT.items()}

def encode_morse(text): return ' '.join(MORSE_CODE_DICT.get(c.upper(), '?') for c in text)
def decode_morse(morse):
    try: return ' '.join([''.join([REVERSE_MORSE.get(l, '') for l in w.split()]) for w in morse.split(' / ')])
    except: return "INVALID"

def apply_tactical_theme():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500&family=JetBrains+Mono:wght@200;400&display=swap');
        :root { --primary: #00FF66; --bg: #050505; --card-bg: #0D0D0D; }
        .stApp { background-color: var(--bg); color: #E0E0E0; font-family: 'Space Grotesk', sans-serif; }
        .hud-card { background: var(--card-bg); border: 1px solid #1A1A1A; padding: 20px; position: relative; margin-bottom: 1rem; }
        .hud-card::before { content: ""; position: absolute; top: -1px; left: -1px; width: 10px; height: 10px; border-top: 2px solid var(--primary); border-left: 2px solid var(--primary); }
        </style>
    """, unsafe_allow_html=True)

# ==================== MAIN APP ====================
def main():
    st.set_page_config(page_title="NEUROMINDS COMMAND", layout="wide")
    apply_tactical_theme()
    
    if 'ai_log' not in st.session_state: st.session_state.ai_log = deque(maxlen=10)

    with st.sidebar:
        st.markdown("<h2 style='color:#00FF66;'>NEUROMINDS v4.2</h2>", unsafe_allow_html=True)
        st.session_state.ai_active = st.toggle("AI OVERRIDE / DETECTION", value=True)
        st.divider()
        st.info("Browser Feed: Uses WebRTC to process your local camera on the cloud.")

    tab1, tab2, tab3 = st.tabs(["[ 01 COMMAND ]", "[ 02 ANALYTICS ]", "[ 03 SECURE COMM ]"])

    with tab1:
        col_vid, col_stats = st.columns([3, 1])
        with col_vid:
            st.markdown('<div class="hud-card">', unsafe_allow_html=True)
            # Fetch camera from browser
            webrtc_streamer(
                key="drone-feed",
                video_transformer_factory=VideoProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False}
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_stats:
            st.markdown('<div class="hud-card">', unsafe_allow_html=True)
            st.metric("BATT", "91%", "-3%")
            st.metric("ALT", "22.1M")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="hud-card">', unsafe_allow_html=True)
        z = np.linspace(0, 10, 100)
        fig = go.Figure(data=[go.Scatter3d(x=np.sin(z)*z, y=np.cos(z)*z, z=z, mode='lines', line=dict(color='#00FF66', width=5))])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
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