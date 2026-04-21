"""Expression and micro-expression analysis."""

import numpy as np
from typing import Dict


class ExpressionAnalyzer:
    """Analyze facial expressions and dynamic qualities."""

    # Expression types
    EXPRESSIONS = ["neutral", "happy", "sad", "surprised", "fearful", "angry", "disgusted"]

    def analyze(self, landmarks: np.ndarray) -> Dict:
        """Analyze expression characteristics from landmarks.

        Returns:
            Expression analysis including smile curve, eye expression, etc.
        """
        smile_curve = self._compute_smile_curve(landmarks)
        eye_expression = self._compute_eye_expression(landmarks)
        brow_position = self._compute_brow_position(landmarks)

        # Overall expression classification
        expression = self._classify_expression(smile_curve, eye_expression, brow_position)

        return {
            "smile_curve": smile_curve,  # positive = upturned corners
            "eye_expression": eye_expression,  # positive = wider/more open
            "brow_position": brow_position,  # positive = raised
            "classified_expression": expression,
            "expression_intensity": self._compute_expression_intensity(landmarks),
        }

    def _compute_smile_curve(self, landmarks: np.ndarray) -> float:
        """Compute smile curve from mouth corners.

        Positive = upturned smile (happier)
        Negative = downturned smile
        """
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mouth_center = landmarks[13]

        # Corner height relative to center
        left_height = left_corner[1] - mouth_center[1]
        right_height = right_corner[1] - mouth_center[1]
        avg_height = (left_height + right_height) / 2

        # Normalize by face width
        face_width = np.linalg.norm(landmarks[234] - landmarks[454])
        return avg_height / face_width if face_width > 0 else 0

    def _compute_eye_expression(self, landmarks: np.ndarray) -> float:
        """Compute eye openness and expression strength.

        Higher = more open/expressive eyes
        """
        left_eye_height = self._eye_openness(landmarks, "left")
        right_eye_height = self._eye_openness(landmarks, "right")
        return (left_eye_height + right_eye_height) / 2

    def _eye_openness(self, landmarks: np.ndarray, side: str) -> float:
        """Compute eye openness for one side."""
        if side == "left":
            upper = landmarks[386]
            lower = landmarks[374]
            corner_left = landmarks[362]
            corner_right = landmarks[263]
        else:
            upper = landmarks[159]
            lower = landmarks[145]
            corner_left = landmarks[33]
            corner_right = landmarks[133]

        openness = np.linalg.norm(upper - lower)
        eye_width = np.linalg.norm(corner_left - corner_right)
        return openness / eye_width if eye_width > 0 else 0

    def _compute_brow_position(self, landmarks: np.ndarray) -> float:
        """Compute brow position relative to eye.

        Higher brows = more surprised/expressive
        """
        left_brow_center = landmarks[300]
        right_brow_center = landmarks[70]
        left_eye_center = landmarks[(386 + 374) // 2]
        right_eye_center = landmarks[(159 + 145) // 2]

        brow_height_left = left_brow_center[1] - left_eye_center[1]
        brow_height_right = right_brow_center[1] - right_eye_center[1]
        avg_height = (brow_height_left + brow_height_right) / 2

        # Normalize
        face_height = np.linalg.norm(landmarks[10] - landmarks[152])
        return avg_height / face_height if face_height > 0 else 0

    def _classify_expression(
        self, smile_curve: float, eye_expression: float, brow_position: float
    ) -> str:
        """Classify overall expression type."""
        # Simple rule-based classification
        if smile_curve > 0.02 and eye_expression > 0.3:
            return "happy"
        elif smile_curve < -0.01 and brow_position < 0:
            return "sad"
        elif brow_position > 0.03 and eye_expression > 0.35:
            return "surprised"
        elif smile_curve < -0.01 and brow_position < -0.02:
            return "angry"
        else:
            return "neutral"

    def _compute_expression_intensity(self, landmarks: np.ndarray) -> float:
        """Compute how strong/dynamic the expression is.

        Returns 0-1 score of expression expressiveness.
        """
        # Based on asymmetry between left and right sides
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        mouth_asymmetry = abs(left_mouth[0] - right_mouth[0])

        left_eye = landmarks[386]
        right_eye = landmarks[159]
        eye_asymmetry = abs(left_eye[1] - right_eye[1])

        # Lower asymmetry = more symmetrical = less expressive
        avg_asymmetry = (mouth_asymmetry + eye_asymmetry) / 2

        # Normalize to 0-1 range
        intensity = min(1.0, avg_asymmetry / 10.0)
        return intensity
