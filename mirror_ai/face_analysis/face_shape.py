"""Face shape classification based on geometric features."""

import numpy as np
from typing import Tuple


# Face shape types
FACE_SHAPES = ["oval", "round", "square", "heart", "diamond"]


class FaceShapeClassifier:
    """Classify face shape based on landmark geometry."""

    def classify(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> str:
        """Classify face shape from landmarks.

        Args:
            landmarks: (468, 3) array of facial landmarks
            image_shape: (H, W, C) of source image

        Returns:
            Face shape type: "oval", "round", "square", "heart", or "diamond"
        """
        h, w = image_shape[:2]
        pts = landmarks[:, :2] * np.array([w, h])

        # Key measurements
        face_length = self._get_face_length(pts)
        face_width = self._get_face_width(pts)
        jaw_width = self._get_jaw_width(pts)
        forehead_width = self._get_forehead_width(pts)
        cheek_width = self._get_cheek_width(pts)

        # Ratios
        length_width_ratio = face_length / face_width if face_width > 0 else 0
        jaw_face_ratio = jaw_width / face_width if face_width > 0 else 0
        forehead_cheek_ratio = forehead_width / cheek_width if cheek_width > 0 else 0

        # Classification based on ratios
        if jaw_face_ratio < 0.85:
            if forehead_cheek_ratio > 1.1:
                return "heart"
            elif forehead_cheek_ratio < 0.95:
                return "diamond"
            else:
                return "oval"
        elif jaw_face_ratio > 0.95:
            if length_width_ratio < 1.3:
                return "round"
            else:
                return "square"
        else:
            if length_width_ratio > 1.5:
                return "oval"
            elif length_width_ratio < 1.2:
                return "round"
            else:
                return "square"

    def _get_face_length(self, pts: np.ndarray) -> float:
        """Forehead to chin distance."""
        forehead_center = (pts[10] + pts[151]) / 2
        chin = pts[8]
        return np.linalg.norm(forehead_center - chin)

    def _get_face_width(self, pts: np.ndarray) -> float:
        """Width at cheekbones."""
        left_cheek = pts[123]
        right_cheek = pts[352]
        return np.linalg.norm(left_cheek - right_cheek)

    def _get_jaw_width(self, pts: np.ndarray) -> float:
        """Width at jaw level."""
        left_jaw = pts[136]
        right_jaw = pts[366]
        return np.linalg.norm(left_jaw - right_jaw)

    def _get_forehead_width(self, pts: np.ndarray) -> float:
        """Width at forehead."""
        left_forehead = pts[336]
        right_forehead = pts[107]
        return np.linalg.norm(left_forehead - right_forehead)

    def _get_cheek_width(self, pts: np.ndarray) -> float:
        """Width at cheeks."""
        left_cheek = pts[234]
        right_cheek = pts[454]
        return np.linalg.norm(left_cheek - right_cheek)

    def get_golden_ratio_analysis(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> dict:
        """Analyze deviation from golden ratio.

        The golden ratio (1.618) appears in ideal facial proportions.
        Returns analysis of how close each proportion is to the ideal.
        """
        h, w = image_shape[:2]
        pts = landmarks[:, :2] * np.array([w, h])

        # Key ratios
        face_length = self._get_face_length(pts)
        face_width = self._get_face_width(pts)

        # Various golden ratio checks
        ratios = {
            "face_length_to_width": face_length / face_width if face_width > 0 else 0,
            "ideal_golden_ratio": 1.618,
        }

        # Compute deviation from golden ratio
        deviation = abs(ratios["face_length_to_width"] - ratios["ideal_golden_ratio"])
        ratios["deviation_from_golden"] = deviation
        ratios["is_ideal_proportion"] = deviation < 0.2

        return ratios
