"""Display overlay and heads-up display (HUD) rendering for OpenCV."""

import cv2
import numpy as np
from src.eye_detector import EyeStateResult
from src.blink_counter import BlinkStatus
from config.settings import EyeBlinkConfig


class DisplayOverlay:
    """Renders real-time HUD overlays, eye landmarks, graphs, and alert banners."""

    def __init__(self, config: EyeBlinkConfig):
        self.config = config

    def draw_hud(
        self,
        frame: np.ndarray,
        eye_result: EyeStateResult | None,
        blink_status: BlinkStatus,
        fps: float,
        is_calibrated: bool,
        calib_progress: float,
        backend_name: str,
    ) -> np.ndarray:
        """Render complete HUD onto the frame."""
        h, w, _ = frame.shape
        annotated = frame.copy()

        # 1. Draw Eye Contours and Landmarks
        if eye_result is not None:
            self._draw_eye_landmarks(annotated, eye_result)

        # 2. Draw Top Control & Stats Panel
        self._draw_top_panel(annotated, eye_result, blink_status, fps, backend_name)

        # 3. Draw EAR Progress Gauge
        if eye_result is not None:
            self._draw_ear_gauge(annotated, eye_result.avg_ear)

        # 4. Draw Calibration Overlay if active
        if not is_calibrated:
            self._draw_calibration_banner(annotated, calib_progress)

        # 5. Draw Prolonged Eye Closure / Drowsiness Alert
        if blink_status.is_prolonged_closure:
            self._draw_warning_banner(annotated, blink_status.current_closure_duration_sec)

        return annotated

    def _draw_eye_landmarks(self, frame: np.ndarray, eye_result: EyeStateResult):
        """Draw landmark points and polygon around eyes."""
        color = self.config.color_normal if not eye_result.is_closed else self.config.color_closed

        for pts in [eye_result.left_eye_points, eye_result.right_eye_points]:
            if pts is not None and len(pts) > 0:
                # Draw polygon around eye
                cv2.polylines(frame, [pts.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=1)
                # Draw landmark points
                for p in pts:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

    def _draw_top_panel(
        self,
        frame: np.ndarray,
        eye_result: EyeStateResult | None,
        blink_status: BlinkStatus,
        fps: float,
        backend_name: str,
    ):
        """Draw transparent dark banner on top with organized telemetry."""
        w = frame.shape[1]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Eye State Tag
        if eye_result is not None:
            state_color = self.config.color_closed if eye_result.is_closed else self.config.color_normal
            state_text = f"EYES: {eye_result.eye_state}"
            ear_text = f"EAR: {eye_result.avg_ear:.2f} (Thresh: {self.config.ear_threshold:.2f})"
        else:
            state_color = (0, 165, 255)
            state_text = "NO FACE DETECTED"
            ear_text = "EAR: N/A"

        cv2.putText(frame, state_text, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, state_color, 2)
        cv2.putText(frame, ear_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Blink Metrics (Middle)
        blink_text = f"Blinks: {blink_status.total_blinks}"
        bpm_text = f"Rate: {blink_status.blinks_per_minute:.0f} BPM"
        cv2.putText(frame, blink_text, (w // 2 - 60, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, bpm_text, (w // 2 - 60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # System Info (Right)
        fps_text = f"FPS: {fps:.0f}"
        backend_text = f"Engine: {backend_name}"
        cv2.putText(frame, fps_text, (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, backend_text, (w - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def _draw_ear_gauge(self, frame: np.ndarray, avg_ear: float):
        """Draw dynamic EAR bar gauge on the left side."""
        gauge_x, gauge_y, gauge_w, gauge_h = 20, 110, 16, 120
        # Background bar
        cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h), (50, 50, 50), -1)

        # Calculate fill ratio (0.0 to 0.4 EAR mapped to 0 to 120 pixels)
        max_ear = 0.40
        ratio = min(1.0, max(0.0, avg_ear / max_ear))
        fill_h = int(ratio * gauge_h)
        gauge_color = self.config.color_closed if avg_ear < self.config.ear_threshold else self.config.color_normal

        # Filled bar (from bottom up)
        cv2.rectangle(
            frame,
            (gauge_x, gauge_y + gauge_h - fill_h),
            (gauge_x + gauge_w, gauge_y + gauge_h),
            gauge_color,
            -1,
        )

        # Threshold indicator line
        thresh_ratio = min(1.0, self.config.ear_threshold / max_ear)
        thresh_y = int(gauge_y + gauge_h - (thresh_ratio * gauge_h))
        cv2.line(frame, (gauge_x - 4, thresh_y), (gauge_x + gauge_w + 4, thresh_y), (0, 0, 255), 2)

    def _draw_calibration_banner(self, frame: np.ndarray, progress: float):
        """Show auto-calibration progress banner."""
        h, w, _ = frame.shape
        bar_w = 300
        bar_h = 20
        bx = (w - bar_w) // 2
        by = h - 60

        cv2.rectangle(frame, (bx - 10, by - 30), (bx + bar_w + 10, by + bar_h + 10), (0, 0, 0), -1)
        pct = int(progress * 100)
        cv2.putText(
            frame,
            f"Calibrating Eyes... {pct}% (Keep eyes open)",
            (bx, by - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        # Progress bar
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), -1)
        cv2.rectangle(frame, (bx, by), (bx + int(bar_w * progress), by + bar_h), (0, 255, 0), -1)

    def _draw_warning_banner(self, frame: np.ndarray, duration_sec: float):
        """Display an eye-catching red alert banner across screen."""
        h, w, _ = frame.shape
        banner_h = 70
        by = (h // 2) - (banner_h // 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, by), (w, by + banner_h), (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border
        cv2.rectangle(frame, (0, by), (w, by + banner_h), (0, 0, 255), 3)

        alert_msg = f"! DROWSINESS ALERT: EYES CLOSED FOR {duration_sec:.1f}s !"
        cv2.putText(
            frame,
            alert_msg,
            (w // 2 - 280, by + 45),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
