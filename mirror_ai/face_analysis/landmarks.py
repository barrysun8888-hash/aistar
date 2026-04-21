"""MediaPipe FaceMesh wrapper for 468-point facial landmark extraction."""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarks:
    """Extract 468 3D facial landmarks using MediaPipe FaceMesh."""

    # Key landmark indices for different facial regions
    LEFT_EYE = list(range(362, 382))
    RIGHT_EYE = list(range(133, 153))
    LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282]
    RIGHT_EYEBROW = [107, 66, 69, 103, 70, 46, 53, 52]
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 292, 325, 319, 403, 270]
    NOSE = [1, 2, 98, 327, 4, 5, 6, 168, 195, 197]
    FACE_OVAL = list(range(10, 127))

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 landmarks from an image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            Landmarks array (468, 3) with x, y, z coordinates, or None if no face detected
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

    def get_face_rect(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get bounding box of face from landmarks.

        Returns:
            (x_min, y_min, x_max, y_max) in pixel coordinates
        """
        h, w = image_shape[:2]
        pts = landmarks[:, :2] * np.array([w, h])
        x_min, y_min = pts.min(axis=0).astype(int)
        x_max, y_max = pts.max(axis=0).astype(int)
        return x_min, y_min, x_max, y_max

    def get_region(self, landmarks: np.ndarray, indices: List[int]) -> np.ndarray:
        """Extract landmarks for a specific facial region."""
        return landmarks[indices]

    def compute_eye_distance(self, landmarks: np.ndarray) -> float:
        """Compute normalized inter-eye distance."""
        left_eye_center = landmarks[self.LEFT_EYE].mean(axis=0)
        right_eye_center = landmarks[self.RIGHT_EYE].mean(axis=0)
        distance = np.linalg.norm(left_eye_center - right_eye_center)
        face_width = self._compute_face_width(landmarks)
        return distance / face_width if face_width > 0 else 0

    def _compute_face_width(self, landmarks: np.ndarray) -> float:
        """Helper to compute face width."""
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        return np.linalg.norm(left_cheek - right_cheek)

    def compute_nose_ratio(self, landmarks: np.ndarray) -> float:
        """Compute nose width to height ratio."""
        nose_bridge = landmarks[1]
        nose_tip = landmarks[6]
        nose_width_left = landmarks[329]
        nose_width_right = landmarks[327]
        nose_height = np.linalg.norm(nose_bridge - nose_tip)
        nose_width = np.linalg.norm(nose_width_left - nose_width_right)
        return nose_width / nose_height if nose_height > 0 else 0

    def compute_lip_ratio(self, landmarks: np.ndarray) -> float:
        """Compute lip width to height ratio."""
        lip_left = landmarks[61]
        lip_right = landmarks[291]
        lip_upper = landmarks[13]
        lip_lower = landmarks[14]
        lip_width = np.linalg.norm(lip_left - lip_right)
        lip_height = np.linalg.norm(lip_upper - lip_lower)
        return lip_height / lip_width if lip_width > 0 else 0

    def extract_features(self, image: np.ndarray) -> dict:
        """Extract all basic features in one pass.

        Returns:
            Dictionary containing landmarks and basic measurements
        """
        landmarks = self.extract(image)
        if landmarks is None:
            return {"landmarks": None, "error": "No face detected"}

        return {
            "landmarks": landmarks,
            "eye_distance": self.compute_eye_distance(landmarks),
            "face_width": self._compute_face_width(landmarks),
            "nose_ratio": self.compute_nose_ratio(landmarks),
            "lip_ratio": self.compute_lip_ratio(landmarks),
        }
