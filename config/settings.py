"""Configuration settings for Eye Blink Detection System."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EyeBlinkConfig:
    # ---------------- Video / Input Settings ----------------
    video_source: str | int = 0  # 0 for webcam or path to video file
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30

    # ---------------- Eye Landmark Indices (MediaPipe Face Mesh) ----------------
    # Left eye landmarks (standard 6 canonical points: [P1, P2, P3, P4, P5, P6])
    # P1 (outer corner), P2 (top-outer), P3 (top-inner), P4 (inner corner), P5 (bottom-inner), P6 (bottom-outer)
    left_eye_indices: list[int] = field(
        default_factory=lambda: [33, 160, 158, 133, 153, 144]
    )

    # Right eye landmarks (canonical 6 points)
    right_eye_indices: list[int] = field(
        default_factory=lambda: [362, 385, 387, 263, 373, 380]
    )

    # Full eye contours for visual overlay rendering
    left_eye_contour: list[int] = field(
        default_factory=lambda: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    )
    right_eye_contour: list[int] = field(
        default_factory=lambda: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    )

    # ---------------- Detection & Blink Thresholds ----------------
    # Eye Aspect Ratio (EAR) default threshold below which eye is considered closed
    ear_threshold: float = 0.22

    # Number of consecutive frames the eye must be below threshold to record a blink
    min_blink_frames: int = 2

    # Maximum frames below threshold to count as normal blink (beyond this is prolonged closure)
    max_blink_frames: int = 15

    # Prolonged closure duration in seconds (Drowsiness / Fatigue warning)
    prolonged_closure_sec: float = 1.5

    # ---------------- Calibration Settings ----------------
    auto_calibrate: bool = True
    calibration_frames: int = 90  # ~3 seconds at 30fps to learn user's baseline open EAR
    calibration_factor: float = 0.75  # Threshold = baseline_open_ear * 0.75

    # ---------------- Audio Alert Settings ----------------
    enable_audio_alert: bool = True
    alert_beep_freq: int = 1200   # Hertz (Windows winsound)
    alert_beep_duration: int = 350 # Milliseconds

    # ---------------- Logging & Output ----------------
    log_dir: Path = field(default_factory=lambda: Path("data/logs"))
    save_session_csv: bool = True

    # ---------------- UI Color Scheme (BGR) ----------------
    color_normal: tuple[int, int, int] = (0, 255, 0)      # Green
    color_closed: tuple[int, int, int] = (0, 165, 255)    # Orange
    color_warning: tuple[int, int, int] = (0, 0, 255)     # Red
    color_text: tuple[int, int, int] = (255, 255, 255)    # White
    color_panel_bg: tuple[int, int, int] = (30, 30, 30)   # Dark Gray
