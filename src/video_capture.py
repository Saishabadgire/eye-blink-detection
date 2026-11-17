"""Video capture and camera management module."""

import time
import cv2
import numpy as np


class VideoCaptureManager:
    """Manages webcam feeds and video file streams with FPS estimation."""

    def __init__(self, source: str | int = 0, width: int = 640, height: int = 480):
        self.source = source
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None

        self.prev_time = 0.0
        self.current_fps = 0.0
        self.frame_count = 0

    def start(self) -> bool:
        """Initialize and open the video stream."""
        if isinstance(self.source, str) and self.source.isdigit():
            src = int(self.source)
        else:
            src = self.source

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            return False

        # Attempt to set camera resolution
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.prev_time = time.time()
        self.frame_count = 0
        return True

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read a frame and compute rolling FPS."""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        now = time.time()
        elapsed = now - self.prev_time
        if elapsed > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / elapsed)
        self.prev_time = now
        self.frame_count += 1

        return True, frame

    def get_fps(self) -> float:
        """Return estimated current frames per second."""
        return max(1.0, round(self.current_fps, 1))

    def stop(self):
        """Release the capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
