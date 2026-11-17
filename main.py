"""Main entry point for real-time Eye Blink Detection System."""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
from config.settings import EyeBlinkConfig
from src.video_capture import VideoCaptureManager
from src.face_mesh import FaceMeshDetector
from src.eye_detector import EyeDetector
from src.blink_counter import BlinkCounter
from src.audio_alert import AudioAlertManager
from src.display_overlay import DisplayOverlay
from src.evaluator import BlinkSessionLogger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eye Blink Detection System (OpenCV & Computer Vision)")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (e.g. 0) or path to video file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.22,
        help="Initial Eye Aspect Ratio (EAR) threshold (default: 0.22)",
    )
    parser.add_argument(
        "--prolonged-sec",
        type=float,
        default=1.5,
        help="Prolonged closure threshold in seconds for drowsiness warning (default: 1.5s)",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable automatic open-eye baseline calibration on startup",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio beep alert on prolonged closure",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window (for automated testing / benchmark)",
    )
    return parser.parse_args()


def run_app():
    args = parse_arguments()

    # Load configuration
    cfg = EyeBlinkConfig()
    cfg.video_source = int(args.source) if args.source.isdigit() else args.source
    cfg.ear_threshold = args.threshold
    cfg.prolonged_closure_sec = args.prolonged_sec
    cfg.auto_calibrate = not args.no_calibration
    cfg.enable_audio_alert = not args.no_audio

    print("=" * 65)
    print("      EYE BLINK DETECTION & DROWSINESS MONITORING SYSTEM")
    print("=" * 65)
    print(f" Source:            {cfg.video_source}")
    print(f" Initial Threshold: {cfg.ear_threshold}")
    print(f" Prolonged Limit:   {cfg.prolonged_closure_sec}s")
    print(f" Auto-Calibration:  {cfg.auto_calibrate}")
    print(f" Audio Alert:       {cfg.enable_audio_alert}")
    print("-" * 65)
    print(" Controls:")
    print("   [SPACE] / [p] - Pause / Resume the video stream")
    print("   [q] / [ESC]   - Quit and export session summary")
    print("   [r]           - Reset blink count and metrics")
    print("   [c]           - Trigger open-eye re-calibration")
    print("   [+] / [-]     - Adjust EAR threshold (+0.01 / -0.01)")
    print("=" * 65)

    # Initialize modules
    video_mgr = VideoCaptureManager(source=cfg.video_source, width=cfg.frame_width, height=cfg.frame_height)
    if not video_mgr.start():
        print(f"[ERROR] Failed to open video source: {cfg.video_source}")
        return

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

    audio_alert = AudioAlertManager(
        enabled=cfg.enable_audio_alert,
        frequency=cfg.alert_beep_freq,
        duration_ms=cfg.alert_beep_duration,
    )

    overlay = DisplayOverlay(config=cfg)
    logger = BlinkSessionLogger(output_dir=cfg.log_dir)

    window_name = "Eye Blink Detection System - [Press SPACE to Pause, Q to Exit]"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, cfg.frame_width, cfg.frame_height)

    prolonged_incidents = 0
    prev_prolonged_state = False
    prev_blink_count = 0
    is_paused = False
    last_frame = None

    try:
        while True:
            # Handle Paused State
            if is_paused:
                if not args.headless and last_frame is not None:
                    paused_view = last_frame.copy()
                    h, w, _ = paused_view.shape
                    # Semi-transparent dark overlay
                    dark_box = paused_view.copy()
                    cv2.rectangle(dark_box, (w // 2 - 220, h // 2 - 40), (w // 2 + 220, h // 2 + 40), (0, 0, 0), -1)
                    cv2.addWeighted(dark_box, 0.7, paused_view, 0.3, 0, paused_view)
                    cv2.rectangle(paused_view, (w // 2 - 220, h // 2 - 40), (w // 2 + 220, h // 2 + 40), (0, 255, 255), 2)
                    cv2.putText(
                        paused_view,
                        "PAUSED - Press SPACE or 'P' to Resume",
                        (w // 2 - 200, h // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow(window_name, paused_view)

                key = cv2.waitKey(30) & 0xFF
                if key in [ord("p"), ord("P"), 32]:  # Toggle unpause
                    is_paused = False
                    print("[INFO] Resumed.")
                elif key in [ord("q"), ord("Q"), 27]:
                    break
                continue

            ret, frame = video_mgr.read_frame()
            if not ret or frame is None:
                # Video ended or camera disconnected
                break

            current_time = time.time()
            fps = video_mgr.get_fps()

            # 1. Face & Landmark Detection (MediaPipe Tasks / Legacy / Haar)
            face_res = face_detector.process(frame)

            # 2. Eye State & EAR Analysis
            eye_res = None
            if face_res.face_detected and face_res.left_eye_pts is not None and face_res.right_eye_pts is not None:
                eye_res = eye_detector.process_eyes(
                    left_points=face_res.left_eye_pts,
                    right_points=face_res.right_eye_pts,
                    blink_score=face_res.blink_score,
                )
                cfg.ear_threshold = eye_detector.threshold

                # 3. Blink Tracking & Closure Duration
                status = blink_counter.update(
                    is_closed=eye_res.is_closed,
                    current_ear=eye_res.avg_ear,
                    current_time=current_time,
                )
            else:
                # No face detected in this frame
                status = blink_counter.update(
                    is_closed=False,
                    current_ear=0.0,
                    current_time=current_time,
                )

            # Check if a new blink occurred
            if status.total_blinks > prev_blink_count:
                prev_blink_count = status.total_blinks
                if status.last_blink is not None:
                    logger.log_blink(status.last_blink)
                    print(f"[BLINK DETECTED] Blink #{status.total_blinks} | Duration: {status.last_blink.duration_sec}s | Min EAR: {status.last_blink.min_ear}")

            # Prolonged closure alerting
            if status.is_prolonged_closure:
                audio_alert.trigger_alert()
                if not prev_prolonged_state:
                    prolonged_incidents += 1
                    prev_prolonged_state = True
            else:
                prev_prolonged_state = False

            # 4. Display & HUD Rendering
            if not args.headless:
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

                last_frame = annotated.copy()
                cv2.imshow(window_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:  # Quit
                    break
                elif key in [ord("p"), ord("P"), 32]:  # Spacebar or 'p' -> Pause
                    is_paused = True
                    print("[INFO] Paused. Press SPACE or 'p' to resume.")
                elif key in [ord("r"), ord("R")]:  # Reset
                    blink_counter.reset()
                    prev_blink_count = 0
                    print("[INFO] Blink counters reset.")
                elif key in [ord("c"), ord("C")]:  # Recalibrate
                    eye_detector.reset_calibration()
                    print("[INFO] Re-calibration started. Please look at the camera with eyes open.")
                elif key in [ord("+"), ord("=")]:
                    cfg.ear_threshold = round(cfg.ear_threshold + 0.01, 3)
                    eye_detector.threshold = cfg.ear_threshold
                    print(f"[CONFIG] EAR Threshold adjusted to: {cfg.ear_threshold}")
                elif key in [ord("-"), ord("_")]:
                    cfg.ear_threshold = round(max(0.05, cfg.ear_threshold - 0.01), 3)
                    eye_detector.threshold = cfg.ear_threshold
                    print(f"[CONFIG] EAR Threshold adjusted to: {cfg.ear_threshold}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        video_mgr.stop()
        if not args.headless:
            cv2.destroyAllWindows()

        # Export session metrics
        summary = logger.export_summary(
            total_blinks=blink_counter.total_blinks,
            prolonged_closures=prolonged_incidents,
        )
        print("\n" + "=" * 65)
        print("                        SESSION SUMMARY")
        print("=" * 65)
        for k, v in summary.items():
            print(f" {k.replace('_', ' ').capitalize():<24}: {v}")
        print("=" * 65)


if __name__ == "__main__":
    run_app()
