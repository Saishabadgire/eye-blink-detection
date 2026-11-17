"""Eye state and Eye Aspect Ratio (EAR) computation module."""

from dataclasses import dataclass
import numpy as np


@dataclass
class EyeStateResult:
    left_ear: float
    right_ear: float
    avg_ear: float
    is_closed: bool
    eye_state: str  # 'OPEN', 'CLOSED'
    left_eye_points: np.ndarray
    right_eye_points: np.ndarray
    blink_score: float | None = None


def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) for 6 canonical eye landmarks:
    p1: outer corner, p2: top-outer, p3: top-inner,
    p4: inner corner, p5: bottom-inner, p6: bottom-outer

    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    """
    if eye_points is None or len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points[:6]

    # Compute Euclidean distances between the vertical landmarks
    dist_vertical_1 = float(np.linalg.norm(p2 - p6))
    dist_vertical_2 = float(np.linalg.norm(p3 - p5))

    # Compute Euclidean distance between horizontal landmarks
    dist_horizontal = float(np.linalg.norm(p1 - p4))

    if dist_horizontal < 1e-6:
        return 0.0

    ear = (dist_vertical_1 + dist_vertical_2) / (2.0 * dist_horizontal)
    return float(ear)


class EyeDetector:
    """Detects eye state using EAR and blendshapes with dynamic calibration."""

    def __init__(
        self,
        default_threshold: float = 0.22,
        auto_calibrate: bool = True,
        calibration_frames: int = 90,
        calibration_factor: float = 0.75,
    ):
        self.threshold = default_threshold
        self.auto_calibrate = auto_calibrate
        self.calibration_frames = calibration_frames
        self.calibration_factor = calibration_factor

        self.calibration_buffer: list[float] = []
        self.is_calibrated = not auto_calibrate
        self.baseline_open_ear = 0.0

    def calibrate(self, avg_ear: float) -> bool:
        """Add sample to calibration buffer. Returns True when calibration completes."""
        if self.is_calibrated:
            return True

        if avg_ear > 0.15:  # Avoid including closed eyes during calibration
            self.calibration_buffer.append(avg_ear)

        if len(self.calibration_buffer) >= self.calibration_frames:
            # Baseline is the median of open-eye measurements
            self.baseline_open_ear = float(np.median(self.calibration_buffer))
            self.threshold = round(self.baseline_open_ear * self.calibration_factor, 3)
            self.is_calibrated = True
            print(f"[CALIBRATION] Baseline Open EAR: {self.baseline_open_ear:.3f} | Threshold set to: {self.threshold:.3f}")
            return True

        return False

    def reset_calibration(self):
        """Reset calibration and restart baseline learning."""
        self.calibration_buffer.clear()
        self.is_calibrated = not self.auto_calibrate
        self.baseline_open_ear = 0.0

    def process_eyes(
        self,
        left_points: np.ndarray,
        right_points: np.ndarray,
        blink_score: float | None = None,
    ) -> EyeStateResult:
        """Calculate EAR for both eyes and evaluate open/closed state."""
        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)

        # Average EAR across both eyes
        if left_ear > 0 and right_ear > 0:
            avg_ear = (left_ear + right_ear) / 2.0
        elif left_ear > 0:
            avg_ear = left_ear
        else:
            avg_ear = right_ear

        # Update calibration if needed
        if not self.is_calibrated:
            self.calibrate(avg_ear)

        # Eye is closed if EAR drops below threshold OR if blendshape blink score indicates closed (> 0.50)
        is_closed_by_ear = avg_ear < self.threshold if avg_ear > 0 else False
        is_closed_by_blendshape = (blink_score is not None and blink_score > 0.50)

        is_closed = is_closed_by_ear or is_closed_by_blendshape
        state_str = "CLOSED" if is_closed else "OPEN"

        return EyeStateResult(
            left_ear=round(left_ear, 3),
            right_ear=round(right_ear, 3),
            avg_ear=round(avg_ear, 3),
            is_closed=is_closed,
            eye_state=state_str,
            left_eye_points=left_points,
            right_eye_points=right_points,
            blink_score=round(blink_score, 2) if blink_score is not None else None,
        )
