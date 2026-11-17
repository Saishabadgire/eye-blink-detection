"""Face landmark extraction module using MediaPipe Tasks FaceLandmarker with legacy & Haar fallbacks."""

from dataclasses import dataclass
from pathlib import Path
import urllib.request
import cv2
import numpy as np


@dataclass
class FaceDetectionResult:
    face_detected: bool
    left_eye_pts: np.ndarray | None
    right_eye_pts: np.ndarray | None
    left_contour_pts: np.ndarray | None
    right_contour_pts: np.ndarray | None
    face_bbox: tuple[int, int, int, int] | None  # (x, y, w, h)
    all_landmarks: np.ndarray | None
    blink_score: float | None = None  # MediaPipe blendshape blink confidence (0.0=open, 1.0=closed)
    backend: str = "OpenCV-Haar"


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


class FaceMeshDetector:
    """Detects facial landmarks and extracts eye coordinates using MediaPipe Tasks or fallback."""

    def __init__(
        self,
        left_eye_indices: list[int],
        right_eye_indices: list[int],
        left_contour_indices: list[int],
        right_contour_indices: list[int],
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.left_eye_indices = left_eye_indices
        self.right_eye_indices = right_eye_indices
        self.left_contour_indices = left_contour_indices
        self.right_contour_indices = right_contour_indices

        self.backend = "OpenCV-Haar"
        self.tasks_landmarker = None
        self.legacy_face_mesh = None
        self.mp = None

        # 1. Try MediaPipe Tasks API (modern mediapipe >= 0.10.x / 1.x)
        self._init_mediapipe_tasks()

        # 2. If Tasks API not available, try legacy solutions.face_mesh
        if self.backend == "OpenCV-Haar":
            self._init_mediapipe_legacy(max_num_faces, refine_landmarks, min_detection_confidence, min_tracking_confidence)

        # 3. Pre-load Haar cascades for extreme fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def _ensure_model_file(self) -> Path | None:
        """Locates or downloads the face_landmarker.task model file."""
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "face_landmarker.task"

        if not model_path.exists() or model_path.stat().st_size < 100000:
            print("[INFO] Downloading MediaPipe Face Landmarker model (3.7 MB)...")
            try:
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print(f"[SUCCESS] Model saved to {model_path}")
            except Exception as e:
                print(f"[WARNING] Could not download model: {e}")
                return None
        return model_path

    def _init_mediapipe_tasks(self):
        """Initialize modern MediaPipe Tasks FaceLandmarker."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = self._ensure_model_file()
            if model_path is None or not model_path.exists():
                return

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                num_faces=1,
            )
            self.tasks_landmarker = vision.FaceLandmarker.create_from_options(options)
            self.mp = mp
            self.backend = "MediaPipe-Tasks"
            print("[INFO] Initialized MediaPipe Tasks FaceLandmarker (High Precision).")
        except Exception as e:
            print(f"[DEBUG] MediaPipe Tasks init fallback: {e}")

    def _init_mediapipe_legacy(self, max_num_faces, refine_landmarks, min_detection_conf, min_tracking_conf):
        """Initialize legacy MediaPipe FaceMesh (for older versions)."""
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.legacy_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=max_num_faces,
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=min_detection_conf,
                    min_tracking_confidence=min_tracking_conf,
                )
                self.backend = "MediaPipe-Legacy"
                print("[INFO] Initialized MediaPipe Legacy FaceMesh.")
        except Exception:
            pass

    def process(self, frame: np.ndarray) -> FaceDetectionResult:
        """Process a BGR video frame and extract eye landmarks."""
        h, w, _ = frame.shape

        if self.backend == "MediaPipe-Tasks" and self.tasks_landmarker is not None:
            return self._process_tasks(frame, h, w)
        elif self.backend == "MediaPipe-Legacy" and self.legacy_face_mesh is not None:
            return self._process_legacy(frame, h, w)

        return self._process_haar_fallback(frame, h, w)

    def _process_tasks(self, frame: np.ndarray, h: int, w: int) -> FaceDetectionResult:
        """Inference with MediaPipe Tasks FaceLandmarker."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.tasks_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return FaceDetectionResult(
                face_detected=False,
                left_eye_pts=None,
                right_eye_pts=None,
                left_contour_pts=None,
                right_contour_pts=None,
                face_bbox=None,
                all_landmarks=None,
                blink_score=None,
                backend="MediaPipe-Tasks",
            )

        landmarks = result.face_landmarks[0]
        landmark_coords = np.array(
            [[int(p.x * w), int(p.y * h)] for p in landmarks],
            dtype=np.int32,
        )

        # Compute face bounding box
        x_min, y_min = np.min(landmark_coords, axis=0)
        x_max, y_max = np.max(landmark_coords, axis=0)
        face_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        left_pts = landmark_coords[self.left_eye_indices]
        right_pts = landmark_coords[self.right_eye_indices]
        left_contour = landmark_coords[self.left_contour_indices]
        right_contour = landmark_coords[self.right_contour_indices]

        # Extract blendshape blink score if available
        blink_score = None
        if result.face_blendshapes:
            categories = {cat.category_name: cat.score for cat in result.face_blendshapes[0]}
            left_blink = categories.get("eyeBlinkLeft", 0.0)
            right_blink = categories.get("eyeBlinkRight", 0.0)
            blink_score = float((left_blink + right_blink) / 2.0)

        return FaceDetectionResult(
            face_detected=True,
            left_eye_pts=left_pts,
            right_eye_pts=right_pts,
            left_contour_pts=left_contour,
            right_contour_pts=right_contour,
            face_bbox=face_bbox,
            all_landmarks=landmark_coords,
            blink_score=blink_score,
            backend="MediaPipe-Tasks",
        )

    def _process_legacy(self, frame: np.ndarray, h: int, w: int) -> FaceDetectionResult:
        """Inference with legacy MediaPipe FaceMesh."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.legacy_face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return FaceDetectionResult(
                face_detected=False,
                left_eye_pts=None,
                right_eye_pts=None,
                left_contour_pts=None,
                right_contour_pts=None,
                face_bbox=None,
                all_landmarks=None,
                backend="MediaPipe-Legacy",
            )

        landmarks = results.multi_face_landmarks[0].landmark
        landmark_coords = np.array(
            [[int(p.x * w), int(p.y * h)] for p in landmarks],
            dtype=np.int32,
        )

        x_min, y_min = np.min(landmark_coords, axis=0)
        x_max, y_max = np.max(landmark_coords, axis=0)
        face_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        left_pts = landmark_coords[self.left_eye_indices]
        right_pts = landmark_coords[self.right_eye_indices]
        left_contour = landmark_coords[self.left_contour_indices]
        right_contour = landmark_coords[self.right_contour_indices]

        return FaceDetectionResult(
            face_detected=True,
            left_eye_pts=left_pts,
            right_eye_pts=right_pts,
            left_contour_pts=left_contour,
            right_contour_pts=right_contour,
            face_bbox=face_bbox,
            all_landmarks=landmark_coords,
            backend="MediaPipe-Legacy",
        )

    def _process_haar_fallback(self, frame: np.ndarray, h: int, w: int) -> FaceDetectionResult:
        """Fallback detector when MediaPipe is unavailable."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            return FaceDetectionResult(
                face_detected=False,
                left_eye_pts=None,
                right_eye_pts=None,
                left_contour_pts=None,
                right_contour_pts=None,
                face_bbox=None,
                all_landmarks=None,
                backend="OpenCV-Haar",
            )

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]

        # Estimate eye locations from face geometry
        eye_y = fy + int(fh * 0.35)
        left_cx = fx + int(fw * 0.3)
        right_cx = fx + int(fw * 0.7)
        ew = int(fw * 0.15)
        eh = int(fh * 0.08)

        left_pts = self._box_to_canonical_points(left_cx - ew, eye_y - eh, ew * 2, eh * 2)
        right_pts = self._box_to_canonical_points(right_cx - ew, eye_y - eh, ew * 2, eh * 2)

        return FaceDetectionResult(
            face_detected=True,
            left_eye_pts=left_pts,
            right_eye_pts=right_pts,
            left_contour_pts=left_pts,
            right_contour_pts=right_pts,
            face_bbox=(fx, fy, fw, fh),
            all_landmarks=None,
            backend="OpenCV-Haar",
        )

    def _box_to_canonical_points(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        cx = x + w // 2
        cy = y + h // 2
        rx = w // 2
        ry = h // 3
        return np.array([
            [cx - rx, cy],
            [cx - rx // 2, cy - ry],
            [cx + rx // 2, cy - ry],
            [cx + rx, cy],
            [cx + rx // 2, cy + ry],
            [cx - rx // 2, cy + ry],
        ], dtype=np.int32)
