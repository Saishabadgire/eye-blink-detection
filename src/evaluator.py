"""Testing, logging, and performance evaluation module."""

import csv
import time
from pathlib import Path
from dataclasses import asdict
from src.blink_counter import BlinkEvent


class BlinkSessionLogger:
    """Logs blink detection events and session metrics to CSV files."""

    def __init__(self, output_dir: Path | str = "data/logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"blink_session_{self.session_id}.csv"
        self.start_time = time.time()
        self.events_logged = 0

        self._init_csv()

    def _init_csv(self):
        with open(self.log_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "timestamp_iso", "relative_time_sec", "duration_frames", "duration_sec", "min_ear"])

    def log_blink(self, event: BlinkEvent):
        """Append a blink event to the session CSV."""
        self.events_logged += 1
        rel_time = round(event.timestamp - self.start_time, 3)
        iso_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp))

        with open(self.log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.events_logged,
                iso_str,
                rel_time,
                event.duration_frames,
                event.duration_sec,
                event.min_ear,
            ])

    def export_summary(self, total_blinks: int, prolonged_closures: int) -> dict:
        """Compute summary statistics for the session."""
        total_time = max(0.1, time.time() - self.start_time)
        bpm = (total_blinks / total_time) * 60.0

        summary = {
            "session_id": self.session_id,
            "session_duration_sec": round(total_time, 2),
            "total_blinks_detected": total_blinks,
            "average_bpm": round(bpm, 2),
            "prolonged_closures": prolonged_closures,
            "csv_log_path": str(self.log_file),
        }
        return summary


def evaluate_against_ground_truth(detected_timestamps: list[float], ground_truth_timestamps: list[float], tolerance_sec: float = 0.5) -> dict:
    """
    Evaluate blink detection accuracy against ground truth annotations.
    Matches detected blinks to ground truth within tolerance window.
    """
    matched_gt = set()
    matched_det = set()

    for d_idx, d_t in enumerate(detected_timestamps):
        best_match = None
        min_diff = float("inf")
        for g_idx, g_t in enumerate(ground_truth_timestamps):
            if g_idx in matched_gt:
                continue
            diff = abs(d_t - g_t)
            if diff <= tolerance_sec and diff < min_diff:
                min_diff = diff
                best_match = g_idx

        if best_match is not None:
            matched_gt.add(best_match)
            matched_det.add(d_idx)

    tp = len(matched_gt)
    fp = len(detected_timestamps) - len(matched_det)
    fn = len(ground_truth_timestamps) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    count_error = abs(len(detected_timestamps) - len(ground_truth_timestamps))

    return {
        "ground_truth_count": len(ground_truth_timestamps),
        "detected_count": len(detected_timestamps),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "count_error": count_error,
    }
