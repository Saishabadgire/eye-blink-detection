"""Offline evaluation and benchmark script for Eye Blink Detection System."""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config.settings import EyeBlinkConfig
from src.video_capture import VideoCaptureManager
from src.face_mesh import FaceMeshDetector
from src.eye_detector import EyeDetector
from src.blink_counter import BlinkCounter
from src.evaluator import evaluate_against_ground_truth


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Eye Blink Detection on a recorded video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to ground truth JSON file (optional)")
    parser.add_argument("--threshold", type=float, default=0.22, help="EAR threshold")
    parser.add_argument("--save-plot", action="store_true", help="Save EAR time series plot to image")
    return parser.parse_args()


def run_evaluation():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return

    cfg = EyeBlinkConfig()
    cfg.ear_threshold = args.threshold
    cfg.video_source = str(video_path)
    cfg.auto_calibrate = False  # Static threshold for reproducible benchmarking

    video_mgr = VideoCaptureManager(source=str(video_path))
    if not video_mgr.start():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    face_detector = FaceMeshDetector(
        left_eye_indices=cfg.left_eye_indices,
        right_eye_indices=cfg.right_eye_indices,
        left_contour_indices=cfg.left_eye_contour,
        right_contour_indices=cfg.right_eye_contour,
    )

    eye_detector = EyeDetector(default_threshold=cfg.ear_threshold, auto_calibrate=False)
    blink_counter = BlinkCounter(
        min_blink_frames=cfg.min_blink_frames,
        max_blink_frames=cfg.max_blink_frames,
        prolonged_closure_sec=cfg.prolonged_closure_sec,
    )

    ear_values = []
    frame_indices = []
    timestamps = []
    fps_video = video_mgr.cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_num = 0
    start_eval_time = time.time()
    detected_blink_timestamps = []

    prev_blink_count = 0
    prolonged_count = 0
    in_prolonged = False

    print(f"\n[INFO] Running evaluation on: {video_path.name}")
    print(f"[INFO] Video FPS: {fps_video:.1f} | Engine: {face_detector.backend}")

    while True:
        ret, frame = video_mgr.read_frame()
        if not ret or frame is None:
            break

        current_time = frame_num / fps_video
        timestamps.append(current_time)
        frame_indices.append(frame_num)

        face_res = face_detector.process(frame)
        if face_res.face_detected and face_res.left_eye_pts is not None and face_res.right_eye_pts is not None:
            eye_res = eye_detector.process_eyes(face_res.left_eye_pts, face_res.right_eye_pts)
            ear_values.append(eye_res.avg_ear)
            status = blink_counter.update(
                is_closed=eye_res.is_closed,
                current_ear=eye_res.avg_ear,
                current_time=current_time,
            )
        else:
            ear_values.append(0.0)
            status = blink_counter.update(
                is_closed=False,
                current_ear=0.0,
                current_time=current_time,
            )

        if status.total_blinks > prev_blink_count:
            prev_blink_count = status.total_blinks
            detected_blink_timestamps.append(current_time)

        if status.is_prolonged_closure:
            if not in_prolonged:
                prolonged_count += 1
                in_prolonged = True
        else:
            in_prolonged = False

        frame_num += 1

    video_mgr.stop()
    eval_elapsed = time.time() - start_eval_time
    proc_fps = frame_num / eval_elapsed if eval_elapsed > 0 else 0.0

    print("\n" + "=" * 65)
    print("                     EVALUATION RESULTS")
    print("=" * 65)
    print(f" Total Frames Processed : {frame_num}")
    print(f" Video Duration (s)     : {timestamps[-1]:.2f}s" if timestamps else "0s")
    print(f" Processing Speed       : {proc_fps:.1f} FPS")
    print(f" Detected Blinks        : {blink_counter.total_blinks}")
    print(f" Detected Blink Times(s): {[round(t, 2) for t in detected_blink_timestamps]}")
    print(f" Prolonged Closures     : {prolonged_count}")

    # Ground truth evaluation if file provided
    gt_path = Path(args.ground_truth) if args.ground_truth else video_path.with_suffix(".json")
    if gt_path.exists():
        with open(gt_path, "r") as f:
            gt_data = json.load(f)

        gt_blinks = gt_data.get("blink_timestamps", [])
        metrics = evaluate_against_ground_truth(detected_blink_timestamps, gt_blinks)

        print("-" * 65)
        print(" ACCURACY METRICS (VS GROUND TRUTH):")
        print(f" Ground Truth Blinks    : {metrics['ground_truth_count']}")
        print(f" True Positives (TP)    : {metrics['true_positives']}")
        print(f" False Positives (FP)   : {metrics['false_positives']}")
        print(f" False Negatives (FN)   : {metrics['false_negatives']}")
        print(f" Precision              : {metrics['precision']:.2%}")
        print(f" Recall                 : {metrics['recall']:.2%}")
        print(f" F1-Score               : {metrics['f1_score']:.2%}")
        print(f" Blink Count Error      : {metrics['count_error']}")

    print("=" * 65)

    if args.save_plot and ear_values:
        plot_path = video_path.parent / f"{video_path.stem}_ear_plot.png"
        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, ear_values, label="EAR", color="blue", linewidth=1.5)
        plt.axhline(y=cfg.ear_threshold, color="red", linestyle="--", label=f"Threshold ({cfg.ear_threshold})")
        for bt in detected_blink_timestamps:
            plt.axvline(x=bt, color="green", alpha=0.6, linestyle=":", label="Detected Blink" if bt == detected_blink_timestamps[0] else "")
        plt.title(f"EAR Signal & Blink Detections: {video_path.name}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Eye Aspect Ratio (EAR)")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(plot_path), dpi=150)
        print(f"[INFO] EAR plot saved to: {plot_path}")


if __name__ == "__main__":
    run_evaluation()
