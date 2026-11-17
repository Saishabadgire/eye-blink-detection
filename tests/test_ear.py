"""Unit tests for Eye Aspect Ratio calculation and blink tracking logic."""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eye_detector import calculate_ear, EyeDetector
from src.blink_counter import BlinkCounter


class TestEyeBlinkDetection(unittest.TestCase):

    def setUp(self):
        # Synthetic open eye coordinates: width = 40, height = 14
        # P1=(0,0), P2=(10,-7), P3=(30,-7), P4=(40,0), P5=(30,7), P6=(10,7)
        self.open_eye_points = np.array([
            [0.0, 0.0],
            [10.0, -7.0],
            [30.0, -7.0],
            [40.0, 0.0],
            [30.0, 7.0],
            [10.0, 7.0],
        ], dtype=np.float32)

        # Synthetic closed eye coordinates: width = 40, height = 2
        self.closed_eye_points = np.array([
            [0.0, 0.0],
            [10.0, -1.0],
            [30.0, -1.0],
            [40.0, 0.0],
            [30.0, 1.0],
            [10.0, 1.0],
        ], dtype=np.float32)

    def test_calculate_ear_open(self):
        ear = calculate_ear(self.open_eye_points)
        # Vertical 1 = 14, Vertical 2 = 14, Horizontal = 40 -> EAR = (14+14)/(2*40) = 28/80 = 0.35
        self.assertAlmostEqual(ear, 0.35, places=2)
        self.assertGreater(ear, 0.25)

    def test_calculate_ear_closed(self):
        ear = calculate_ear(self.closed_eye_points)
        # Vertical 1 = 2, Vertical 2 = 2, Horizontal = 40 -> EAR = (2+2)/80 = 0.05
        self.assertAlmostEqual(ear, 0.05, places=2)
        self.assertLess(ear, 0.15)

    def test_eye_detector_classification(self):
        detector = EyeDetector(default_threshold=0.22, auto_calibrate=False)
        open_res = detector.process_eyes(self.open_eye_points, self.open_eye_points)
        self.assertFalse(open_res.is_closed)
        self.assertEqual(open_res.eye_state, "OPEN")

        closed_res = detector.process_eyes(self.closed_eye_points, self.closed_eye_points)
        self.assertTrue(closed_res.is_closed)
        self.assertEqual(closed_res.eye_state, "CLOSED")

    def test_blink_counter_single_blink(self):
        counter = BlinkCounter(min_blink_frames=2, max_blink_frames=10, prolonged_closure_sec=1.5)

        # Open state (frames 1-3)
        t = 0.0
        for _ in range(3):
            counter.update(is_closed=False, current_ear=0.35, current_time=t)
            t += 0.033
        self.assertEqual(counter.total_blinks, 0)

        # Closed state (3 frames: ~0.1s blink)
        for _ in range(3):
            counter.update(is_closed=True, current_ear=0.08, current_time=t)
            t += 0.033
        self.assertEqual(counter.total_blinks, 0)

        # Back to Open state -> Blink should be counted!
        status = counter.update(is_closed=False, current_ear=0.35, current_time=t)
        self.assertEqual(status.total_blinks, 1)
        self.assertIsNotNone(status.last_blink)

    def test_prolonged_closure_warning(self):
        counter = BlinkCounter(prolonged_closure_sec=1.0)
        t = 0.0

        # Close eyes for 0.5s -> should not be prolonged yet
        while t < 0.5:
            status = counter.update(is_closed=True, current_ear=0.05, current_time=t)
            t += 0.1
        self.assertFalse(status.is_prolonged_closure)

        # Keep eyes closed past 1.0s -> prolonged closure alert must trigger!
        while t <= 1.2:
            status = counter.update(is_closed=True, current_ear=0.05, current_time=t)
            t += 0.1
        self.assertTrue(status.is_prolonged_closure)


if __name__ == "__main__":
    unittest.main()
