"""Blink counting and prolonged eye-closure duration tracking module."""

from dataclasses import dataclass
import time
from collections import deque


@dataclass
class BlinkEvent:
    timestamp: float
    duration_frames: int
    duration_sec: float
    min_ear: float


@dataclass
class BlinkStatus:
    total_blinks: int
    consecutive_closed_frames: int
    current_closure_duration_sec: float
    is_prolonged_closure: bool
    blinks_per_minute: float
    last_blink: BlinkEvent | None


class BlinkCounter:
    """Tracks blink events and closure durations using a finite state tracker."""

    def __init__(
        self,
        min_blink_frames: int = 2,
        max_blink_frames: int = 15,
        prolonged_closure_sec: float = 1.5,
        target_fps: float = 30.0,
    ):
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames
        self.prolonged_closure_sec = prolonged_closure_sec
        self.target_fps = target_fps

        self.total_blinks = 0
        self.consecutive_closed_frames = 0
        self.closure_start_time: float | None = None
        self.min_ear_during_closure = 1.0

        # Store recent blink timestamps for Blinks Per Minute (BPM) rolling rate
        self.blink_history: deque[float] = deque()
        self.all_blink_events: list[BlinkEvent] = []
        self.last_blink_event: BlinkEvent | None = None

    def update(self, is_closed: bool, current_ear: float, current_time: float | None = None) -> BlinkStatus:
        """Update tracker with the current frame's eye state."""
        if current_time is None:
            current_time = time.time()

        is_prolonged = False
        closure_duration_sec = 0.0

        if is_closed:
            self.consecutive_closed_frames += 1
            if self.closure_start_time is None:
                self.closure_start_time = current_time
                self.min_ear_during_closure = current_ear
            else:
                self.min_ear_during_closure = min(self.min_ear_during_closure, current_ear)

            closure_duration_sec = current_time - self.closure_start_time

            # Check if prolonged closure threshold is exceeded
            if closure_duration_sec >= self.prolonged_closure_sec:
                is_prolonged = True

        else:
            # Transition: eyes were closed, now open
            if self.consecutive_closed_frames > 0:
                closure_duration = (
                    current_time - self.closure_start_time
                    if self.closure_start_time is not None
                    else (self.consecutive_closed_frames / self.target_fps)
                )

                # Validate whether this was a genuine blink
                if self.min_blink_frames <= self.consecutive_closed_frames <= self.max_blink_frames:
                    self.total_blinks += 1
                    event = BlinkEvent(
                        timestamp=current_time,
                        duration_frames=self.consecutive_closed_frames,
                        duration_sec=round(closure_duration, 3),
                        min_ear=round(self.min_ear_during_closure, 3),
                    )
                    self.all_blink_events.append(event)
                    self.last_blink_event = event
                    self.blink_history.append(current_time)

                # Reset closure tracker
                self.consecutive_closed_frames = 0
                self.closure_start_time = None
                self.min_ear_during_closure = 1.0

        # Purge blinks older than 60 seconds to compute current BPM
        cutoff = current_time - 60.0
        while self.blink_history and self.blink_history[0] < cutoff:
            self.blink_history.popleft()

        bpm = float(len(self.blink_history))

        return BlinkStatus(
            total_blinks=self.total_blinks,
            consecutive_closed_frames=self.consecutive_closed_frames,
            current_closure_duration_sec=round(closure_duration_sec, 2),
            is_prolonged_closure=is_prolonged,
            blinks_per_minute=bpm,
            last_blink=self.last_blink_event,
        )

    def reset(self):
        """Reset all counters and state."""
        self.total_blinks = 0
        self.consecutive_closed_frames = 0
        self.closure_start_time = None
        self.min_ear_during_closure = 1.0
        self.blink_history.clear()
        self.all_blink_events.clear()
        self.last_blink_event = None
