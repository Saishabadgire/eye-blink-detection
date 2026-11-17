"""Synthetic blinking video generator for offline testing and verification."""

import json
from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_blink_video(
    output_path: Path | str = "data/sample_videos/synthetic_blinks.mp4",
    duration_sec: float = 10.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> dict:
    """
    Generates a synthetic video containing a face with animated blinking eyes.
    Includes normal blinks and one prolonged eye closure.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_file), fourcc, float(fps), (width, height))

    total_frames = int(duration_sec * fps)

    # Define blink schedules (in seconds): [start_sec, duration_sec]
    blink_schedule = [
        {"type": "blink", "start": 2.0, "duration": 0.20},  # ~6 frames
        {"type": "blink", "start": 4.0, "duration": 0.17},  # ~5 frames
        {"type": "blink", "start": 6.0, "duration": 0.23},  # ~7 frames
        {"type": "prolonged", "start": 7.5, "duration": 2.0},  # 60 frames (Drowsiness)
    ]

    face_cx, face_cy = width // 2, height // 2
    face_w, face_h = 240, 300

    left_eye_center = (face_cx - 50, face_cy - 40)
    right_eye_center = (face_cx + 50, face_cy - 40)
    eye_radius_x = 28
    eye_radius_y_open = 14
    eye_radius_y_closed = 2

    ground_truth_blinks = []

    for frame_idx in range(total_frames):
        current_time = frame_idx / fps
        frame = np.ones((height, width, 3), dtype=np.uint8) * 235

        # Check if eyes should be closed
        is_closed = False
        for event in blink_schedule:
            if event["start"] <= current_time <= (event["start"] + event["duration"]):
                is_closed = True
                break

        # Draw Face Oval
        cv2.ellipse(frame, (face_cx, face_cy), (face_w // 2, face_h // 2), 0, 0, 360, (200, 180, 160), -1)
        cv2.ellipse(frame, (face_cx, face_cy), (face_w // 2, face_h // 2), 0, 0, 360, (120, 100, 80), 2)

        # Eyebrows
        cv2.line(frame, (left_eye_center[0] - 25, left_eye_center[1] - 20), (left_eye_center[0] + 25, left_eye_center[1] - 22), (60, 40, 20), 4)
        cv2.line(frame, (right_eye_center[0] - 25, right_eye_center[1] - 22), (right_eye_center[0] + 25, right_eye_center[1] - 20), (60, 40, 20), 4)

        # Eyes
        curr_ry = eye_radius_y_closed if is_closed else eye_radius_y_open
        for eye_cx, eye_cy in [left_eye_center, right_eye_center]:
            # Sclera (White)
            cv2.ellipse(frame, (eye_cx, eye_cy), (eye_radius_x, curr_ry), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(frame, (eye_cx, eye_cy), (eye_radius_x, curr_ry), 0, 0, 360, (0, 0, 0), 2)
            # Pupil & Iris
            if not is_closed:
                cv2.circle(frame, (eye_cx, eye_cy), 7, (80, 40, 10), -1)
                cv2.circle(frame, (eye_cx, eye_cy), 3, (0, 0, 0), -1)
            else:
                # Eyelid crease
                cv2.line(frame, (eye_cx - eye_radius_x, eye_cy), (eye_cx + eye_radius_x, eye_cy), (60, 40, 20), 2)

        # Nose & Mouth
        cv2.line(frame, (face_cx, face_cy - 10), (face_cx - 5, face_cy + 25), (140, 120, 100), 2)
        cv2.line(frame, (face_cx - 5, face_cy + 25), (face_cx + 10, face_cy + 25), (140, 120, 100), 2)
        cv2.ellipse(frame, (face_cx, face_cy + 65), (35, 12), 0, 0, 180, (80, 60, 160), 3)

        out.write(frame)

    out.release()

    ground_truth = {
        "video_path": str(out_file),
        "duration_sec": duration_sec,
        "fps": fps,
        "blink_timestamps": [ev["start"] for ev in blink_schedule if ev["type"] == "blink"],
        "prolonged_closure_start": 7.5,
    }

    gt_file = out_file.with_suffix(".json")
    with open(gt_file, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"[SUCCESS] Synthetic test video created at: {out_file}")
    print(f"[SUCCESS] Ground truth metadata written to: {gt_file}")
    return ground_truth


if __name__ == "__main__":
    generate_synthetic_blink_video()
