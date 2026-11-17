"""Streamlit Web GUI & Dashboard for Eye Blink Detection System."""

import tempfile
import time
from pathlib import Path
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config.settings import EyeBlinkConfig
from src.video_capture import VideoCaptureManager
from src.face_mesh import FaceMeshDetector
from src.eye_detector import EyeDetector
from src.blink_counter import BlinkCounter
from src.audio_alert import AudioAlertManager
from src.display_overlay import DisplayOverlay

# Page configuration
st.set_page_config(
    page_title="Eye Blink Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("👁️ Eye Blink Detection & Drowsiness Monitoring System")
st.markdown(
    "A real-time Computer Vision system implementing facial landmark analysis, "
    "**Eye Aspect Ratio (EAR)** computation, and prolonged eye-closure alerting."
)

# Sidebar settings
st.sidebar.header("⚙️ System Configuration")

input_mode = st.sidebar.radio("Input Source", ["Live Webcam", "Upload Video File"])
ear_threshold = st.sidebar.slider("EAR Threshold", min_value=0.10, max_value=0.35, value=0.22, step=0.01)
prolonged_sec = st.sidebar.slider("Prolonged Closure Limit (sec)", min_value=0.5, max_value=4.0, value=1.5, step=0.1)
min_frames = st.sidebar.number_input("Min Blink Frames", min_value=1, max_value=5, value=2)
max_frames = st.sidebar.number_input("Max Blink Frames", min_value=5, max_value=30, value=15)
auto_calibrate = st.sidebar.checkbox("Auto-Calibrate Baseline", value=False)
enable_audio = st.sidebar.checkbox("Enable Audio Beep Alert", value=True)

# Initialize configuration
cfg = EyeBlinkConfig()
cfg.ear_threshold = ear_threshold
cfg.prolonged_closure_sec = prolonged_sec
cfg.min_blink_frames = int(min_frames)
cfg.max_blink_frames = int(max_frames)
cfg.auto_calibrate = auto_calibrate
cfg.enable_audio_alert = enable_audio

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
metric_blinks = col1.empty()
metric_bpm = col2.empty()
metric_ear = col3.empty()
metric_state = col4.empty()

alert_placeholder = st.empty()
video_placeholder = st.empty()
chart_placeholder = st.empty()

# Prepare video source
video_source = None
if input_mode == "Live Webcam":
    video_source = 0
else:
    uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_source = tfile.name

start_btn = st.sidebar.button("▶️ Start Monitoring", use_container_width=True)
stop_btn = st.sidebar.button("⏹️ Stop Monitoring", use_container_width=True)

if start_btn and video_source is not None:
    video_mgr = VideoCaptureManager(source=video_source)
    if not video_mgr.start():
        st.error(f"Failed to open video source: {video_source}")
    else:
        face_detector = FaceMeshDetector(
            left_eye_indices=cfg.left_eye_indices,
            right_eye_indices=cfg.right_eye_indices,
            left_contour_indices=cfg.left_eye_contour,
            right_contour_indices=cfg.right_eye_contour,
        )

        eye_detector = EyeDetector(
            default_threshold=cfg.ear_threshold,
            auto_calibrate=cfg.auto_calibrate,
            calibration_frames=cfg.calibration_frames,
            calibration_factor=cfg.calibration_factor,
        )

        blink_counter = BlinkCounter(
            min_blink_frames=cfg.min_blink_frames,
            max_blink_frames=cfg.max_blink_frames,
            prolonged_closure_sec=cfg.prolonged_closure_sec,
            target_fps=cfg.target_fps,
        )

        audio_alert = AudioAlertManager(enabled=cfg.enable_audio_alert)
        overlay = DisplayOverlay(config=cfg)

        ear_history = []
        timestamps = []
        start_time = time.time()

        while True:
            ret, frame = video_mgr.read_frame()
            if not ret or frame is None:
                break

            current_time = time.time()
            fps = video_mgr.get_fps()

            face_res = face_detector.process(frame)
            eye_res = None

            if face_res.face_detected and face_res.left_eye_pts is not None and face_res.right_eye_pts is not None:
                eye_res = eye_detector.process_eyes(
                    left_points=face_res.left_eye_pts,
                    right_points=face_res.right_eye_pts,
                    blink_score=face_res.blink_score,
                )
                status = blink_counter.update(
                    is_closed=eye_res.is_closed,
                    current_ear=eye_res.avg_ear,
                    current_time=current_time,
                )
                ear_history.append(eye_res.avg_ear)
            else:
                status = blink_counter.update(
                    is_closed=False,
                    current_ear=0.0,
                    current_time=current_time,
                )
                ear_history.append(0.0)

            timestamps.append(round(current_time - start_time, 1))

            # Trigger prolonged closure alert
            if status.is_prolonged_closure:
                audio_alert.trigger_alert()
                alert_placeholder.error(
                    f"⚠️ **DROWSINESS ALERT!** Eyes closed for {status.current_closure_duration_sec:.1f}s!"
                )
            else:
                alert_placeholder.empty()

            # Update metrics
            metric_blinks.metric("Total Blinks", status.total_blinks)
            metric_bpm.metric("Blink Rate (BPM)", f"{status.blinks_per_minute:.0f}")
            metric_ear.metric(
                "Current EAR",
                f"{eye_res.avg_ear:.2f}" if eye_res else "N/A",
                delta=f"Threshold: {eye_detector.threshold:.2f}",
            )
            state_label = eye_res.eye_state if eye_res else "NO FACE"
            metric_state.metric("Eye State", state_label)

            # Render HUD
            calib_progress = (
                len(eye_detector.calibration_buffer) / eye_detector.calibration_frames
                if not eye_detector.is_calibrated
                else 1.0
            )
            annotated = overlay.draw_hud(
                frame=frame,
                eye_result=eye_res,
                blink_status=status,
                fps=fps,
                is_calibrated=eye_detector.is_calibrated,
                calib_progress=min(1.0, calib_progress),
                backend_name=face_detector.backend,
            )

            # Display frame in Streamlit
            rgb_view = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_view, channels="RGB", use_container_width=True)

            # Update rolling EAR plot (last 60 frames)
            if len(ear_history) % 5 == 0:
                recent_ear = ear_history[-60:]
                recent_ts = timestamps[-60:]
                df = pd.DataFrame({"Time (s)": recent_ts, "EAR": recent_ear, "Threshold": [eye_detector.threshold] * len(recent_ear)})
                chart_placeholder.line_chart(df.set_index("Time (s)"))

        video_mgr.stop()
        st.success(f"Session finished! Total blinks: {blink_counter.total_blinks}")
