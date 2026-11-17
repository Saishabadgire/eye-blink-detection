"""Audio alert module for prolonged eye closure warning."""

import sys
import threading
import time


class AudioAlertManager:
    """Manages asynchronous sound alerts without blocking the video processing loop."""

    def __init__(self, enabled: bool = True, frequency: int = 1200, duration_ms: int = 350, cooldown_sec: float = 0.5):
        self.enabled = enabled
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.cooldown_sec = cooldown_sec
        self.last_played_time = 0.0
        self._lock = threading.Lock()
        self._is_playing = False

    def trigger_alert(self):
        """Trigger an alert sound asynchronously if cooldown has elapsed."""
        if not self.enabled:
            return

        now = time.time()
        with self._lock:
            if self._is_playing or (now - self.last_played_time < self.cooldown_sec):
                return
            self._is_playing = True
            self.last_played_time = now

        thread = threading.Thread(target=self._play_sound, daemon=True)
        thread.start()

    def _play_sound(self):
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(self.frequency, self.duration_ms)
            else:
                # Terminal bell fallback on other OS
                print("\a", end="", flush=True)
        except Exception:
            pass
        finally:
            with self._lock:
                self._is_playing = False
