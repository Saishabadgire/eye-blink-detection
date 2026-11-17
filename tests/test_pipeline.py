"""End-to-end pipeline simulation test for Eye Blink Detection System."""

import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eye_detector import EyeDetector
from src.blink_counter import BlinkCounter
from src.evaluator import evaluate_against_ground_truth


def create_synthetic_eye(is_closed: bool, cx: float = 200, cy: float = 200) -> np.ndarray:
    """Generate 6 canonical landmark points for an eye."""
    rx = 20.0
    ry = 2.0 if is_closed else 7.0  # Closed EAR ~ 0.10, Open EAR ~ 0.35
    return np.array([
        [cx - rx, cy],
        [cx - rx / 2, cy - ry],
        [cx + rx / 2, cy - ry],
        [cx + rx, cy],
        [cx + rx / 2, cy + ry],
        [cx - rx / 2, cy + ry],
    ], dtype=np.float32)


def test_simulation_pipeline():
    """Simulates a 15-second video stream with 3 blinks and 1 prolonged eye closure."""
    fps = 30
    total_frames = 15 * fps
    dt = 1.0 / fps

    detector = EyeDetector(default_threshold=0.22, auto_calibrate=False)
    counter = BlinkCounter(min_blink_frames=2, max_blink_frames=12, prolonged_closure_sec=1.5)

    # Ground truth schedule: blinks at t=3.0s, t=6.0s, t=9.0s; prolonged closure at t=12.0s
    schedule = [
        {"type": "blink", "start": 3.0, "duration": 0.20},      # ~6 frames
        {"type": "blink", "start": 6.0, "duration": 0.17},      # ~5 frames
        {"type": "blink", "start": 9.0, "duration": 0.23},      # ~7 frames
        {"type": "prolonged", "start": 12.0, "duration": 2.2},  # 66 frames (> 1.5s)
    ]

    detected_blink_times = []
    prolonged_alerts = 0
    in_prolonged = False
    prev_blinks = 0

    for frame_idx in range(total_frames):
        current_time = frame_idx * dt

        # Determine if eyes are closed at current timestamp
        is_closed_ground_truth = any(
            ev["start"] <= current_time <= (ev["start"] + ev["duration"])
            for ev in schedule
        )

        left_pts = create_synthetic_eye(is_closed=is_closed_ground_truth, cx=150, cy=150)
        right_pts = create_synthetic_eye(is_closed=is_closed_ground_truth, cx=250, cy=150)

        eye_res = detector.process_eyes(left_pts, right_pts)
        status = counter.update(
            is_closed=eye_res.is_closed,
            current_ear=eye_res.avg_ear,
            current_time=current_time,
        )

        if status.total_blinks > prev_blinks:
            prev_blinks = status.total_blinks
            detected_blink_times.append(current_time)

        if status.is_prolonged_closure:
            if not in_prolonged:
                prolonged_alerts += 1
                in_prolonged = True
        else:
            in_prolonged = False

    gt_blinks = [ev["start"] for ev in schedule if ev["type"] == "blink"]
    metrics = evaluate_against_ground_truth(detected_blink_times, gt_blinks, tolerance_sec=0.4)

    print("\n" + "=" * 60)
    print("        END-TO-END SIMULATION PIPELINE TEST")
    print("=" * 60)
    print(f" Total Frames Simulated : {total_frames}")
    print(f" Ground Truth Blinks    : {len(gt_blinks)} (at {gt_blinks})")
    print(f" Detected Blinks        : {len(detected_blink_times)} (at {[round(t, 2) for t in detected_blink_times]})")
    print(f" Prolonged Closures     : {prolonged_alerts} (Expected: 1)")
    print("-" * 60)
    print(f" Precision              : {metrics['precision']:.2%}")
    print(f" Recall                 : {metrics['recall']:.2%}")
    print(f" F1-Score               : {metrics['f1_score']:.2%}")
    print("=" * 60)

    assert metrics["precision"] == 1.0, f"Expected 100% precision, got {metrics['precision']}"
    assert metrics["recall"] == 1.0, f"Expected 100% recall, got {metrics['recall']}"
    assert prolonged_alerts == 1, f"Expected 1 prolonged closure alert, got {prolonged_alerts}"
    print("[PASSED] All pipeline assertions verified successfully!\n")


if __name__ == "__main__":
    test_simulation_pipeline()
