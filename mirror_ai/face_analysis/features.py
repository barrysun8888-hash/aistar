"""Extract detailed facial features: eyes, nose, lips, eyebrows."""

import numpy as np
from typing import Dict, Tuple


# Eye landmark indices
LEFT_EYE_OUTER = [362, 380, 381, 374, 373, 390, 249, 263]
RIGHT_EYE_OUTER = [33, 160, 161, 173, 133, 7, 163, 153]
LEFT_EYE_INNER = [362, 382, 381, 380, 374, 373, 390, 389]
RIGHT_EYE_INNER = [33, 133, 160, 161, 173, 133, 7, 155]


class FeatureExtractor:
    """Extract detailed features from facial landmarks."""

    def extract_eye_features(self, landmarks: np.ndarray) -> Dict:
        """Analyze eye shape and characteristics.

        Returns eye type: almond, round, hooded, monolid, upturned, downturned
        """
        left_eye = landmarks[LEFT_EYE_OUTER + LEFT_EYE_INNER]
        right_eye = landmarks[RIGHT_EYE_OUTER + RIGHT_EYE_INNER]

        left_aspect = self._eye_aspect_ratio(left_eye)
        right_aspect = self._eye_aspect_ratio(right_eye)
        avg_aspect = (left_aspect + right_aspect) / 2

        # Eye angle (tilt)
        left_angle = self._eye_angle(landmarks[LEFT_EYE_OUTER])
        right_angle = self._eye_angle(landmarks[RIGHT_EYE_OUTER])

        eye_type = self._classify_eye_type(avg_aspect, left_angle, right_angle)

        return {
            "type": eye_type,
            "aspect_ratio": avg_aspect,
            "left_angle": left_angle,
            "right_angle": right_angle,
            "is_symmetric": abs(left_angle - right_angle) < 5,
        }

    def _eye_aspect_ratio(self, eye_pts: np.ndarray) -> float:
        """Compute eye aspect ratio (width / height)."""
        width = np.linalg.norm(eye_pts[0] - eye_pts[4])
        heights = [
            np.linalg.norm(eye_pts[1] - eye_pts[5]),
            np.linalg.norm(eye_pts[2] - eye_pts[6]),
            np.linalg.norm(eye_pts[3] - eye_pts[7]),
        ]
        avg_height = np.mean(heights)
        return width / avg_height if avg_height > 0 else 0

    def _eye_angle(self, eye_pts: np.ndarray) -> float:
        """Compute eye tilt angle in degrees."""
        left_corner = eye_pts[0]
        right_corner = eye_pts[4]
        dy = right_corner[1] - left_corner[1]
        dx = right_corner[0] - left_corner[0]
        return np.degrees(np.arctan2(dy, dx))

    def _classify_eye_type(self, aspect_ratio: float, left_angle: float, right_angle: float) -> str:
        """Classify eye shape based on aspect ratio and angle."""
        avg_angle = (left_angle + right_angle) / 2

        if aspect_ratio > 2.5:
            if avg_angle > 5:
                return "upturned"
            elif avg_angle < -5:
                return "downturned"
            else:
                return "almond"
        elif aspect_ratio < 1.8:
            return "round"
        else:
            if avg_angle > 3:
                return "upturned"
            elif avg_angle < -3:
                return "downturned"
            return "almond"

    def extract_nose_features(self, landmarks: np.ndarray) -> Dict:
        """Analyze nose shape.

        Returns nose type based on bridge, tip, and wing characteristics.
        """
        nose_tip = landmarks[4]
        nose_bridge_top = landmarks[10]
        nose_bridge_bottom = landmarks[2]
        nose_left_wing = landmarks[329]
        nose_right_wing = landmarks[327]
        nose_bottom_left = landmarks[98]
        nose_bottom_right = landmarks[327]

        # Bridge straightness
        bridge_vector = nose_bridge_bottom - nose_bridge_top
        bridge_straightness = abs(bridge_vector[0]) / np.linalg.norm(bridge_vector) if np.linalg.norm(bridge_vector) > 0 else 0

        # Tip definition
        tip_angle = self._tip_angle(landmarks)

        # Width
        width = np.linalg.norm(nose_left_wing - nose_right_wing)
        bridge_length = np.linalg.norm(nose_tip - nose_bridge_top)
        width_ratio = width / bridge_length if bridge_length > 0 else 0

        # Classification
        if width_ratio < 0.4:
            nose_type = "narrow"
        elif width_ratio > 0.6:
            nose_type = "wide"
        else:
            nose_type = "medium"

        if bridge_straightness < 0.2:
            nose_type += "_straight"
        else:
            nose_type += "_curved"

        return {
            "type": nose_type,
            "width_ratio": width_ratio,
            "bridge_straightness": bridge_straightness,
            "tip_angle": tip_angle,
        }

    def _tip_angle(self, landmarks: np.ndarray) -> float:
        """Compute nose tip angle."""
        nose_tip = landmarks[4]
        left_ala = landmarks[98]
        right_ala = landmarks[327]
        vector_left = left_ala - nose_tip
        vector_right = right_ala - nose_tip
        angle = np.degrees(np.arctan2(vector_right[1] - vector_left[1], vector_right[0] - vector_left[0]))
        return angle

    def extract_lip_features(self, landmarks: np.ndarray) -> Dict:
        """Analyze lip shape.

        Returns lip type based on fullness and shape.
        """
        upper_lip_center = landmarks[13]
        lower_lip_center = landmarks[14]
        lip_left = landmarks[61]
        lip_right = landmarks[291]

        # Fullness ratio
        height = np.linalg.norm(upper_lip_center - lower_lip_center)
        width = np.linalg.norm(lip_left - lip_right)
        fullness_ratio = height / width if width > 0 else 0

        # Cupid's bow (upper lip shape)
        cupid_bow = self._cupid_bow(landmarks)

        # Corner angle
        corner_angle = self._corner_angle(landmarks)

        # Classification
        if fullness_ratio > 0.35:
            fullness = "full"
        elif fullness_ratio < 0.25:
            fullness = "thin"
        else:
            fullness = "medium"

        return {
            "fullness": fullness,
            "fullness_ratio": fullness_ratio,
            "cupid_bow": cupid_bow,
            "corner_angle": corner_angle,
        }

    def _cupid_bow(self, landmarks: np.ndarray) -> str:
        """Classify upper lip shape."""
        upper_lip = landmarks[185]
        left_peak = landmarks[39]
        right_peak = landmarks[269]
        center = landmarks[13]

        left_slope = abs(upper_lip[1] - left_peak[1])
        right_slope = abs(upper_lip[1] - right_peak[1])

        if abs(left_slope - right_slope) < 2:
            return "defined"
        elif left_slope > right_slope:
            return "left_heavy"
        return "right_heavy"

    def _corner_angle(self, landmarks: np.ndarray) -> float:
        """Compute lip corner angle."""
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        center = landmarks[13]
        left_angle = np.degrees(np.arctan2(center[1] - left_corner[1], center[0] - left_corner[0]))
        right_angle = np.degrees(np.arctan2(center[1] - right_corner[1], center[0] - right_corner[0]))
        return (left_angle + right_angle) / 2

    def extract_eyebrow_features(self, landmarks: np.ndarray) -> Dict:
        """Analyze eyebrow shape.

        Returns eyebrow type based on arch position and thickness.
        """
        left_brow = landmarks[[336, 296, 334, 293, 300, 276, 283, 282]]
        right_brow = landmarks[[107, 66, 69, 103, 70, 46, 53, 52]]

        left_arch_pos = self._arch_position(left_brow)
        right_arch_pos = self._arch_position(right_brow)

        left_thickness = self._brow_thickness(left_brow)
        right_thickness = self._brow_thickness(right_brow)

        # Classification
        avg_arch = (left_arch_pos + right_arch_pos) / 2
        if avg_arch < 0.35:
            arch_type = "low"
        elif avg_arch > 0.55:
            arch_type = "high"
        else:
            arch_type = "medium"

        avg_thickness = (left_thickness + right_thickness) / 2
        if avg_thickness > 3:
            thickness = "thick"
        elif avg_thickness < 1.5:
            thickness = "thin"
        else:
            thickness = "medium"

        return {
            "arch_type": arch_type,
            "thickness": thickness,
            "left_arch_pos": left_arch_pos,
            "right_arch_pos": right_arch_pos,
        }

    def _arch_position(self, brow_pts: np.ndarray) -> float:
        """Compute normalized arch position (0 = inner, 1 = outer)."""
        brow_length = np.linalg.norm(brow_pts[0] - brow_pts[-1])
        arch_pt = brow_pts[3]
        start_pt = brow_pts[0]
        end_pt = brow_pts[-1]
        total_length = np.linalg.norm(end_pt - start_pt)
        if total_length == 0:
            return 0.5
        arch_position = np.linalg.norm(arch_pt - start_pt) / total_length
        return arch_position

    def _brow_thickness(self, brow_pts: np.ndarray) -> float:
        """Compute average eyebrow thickness."""
        thicknesses = []
        for i in range(len(brow_pts) - 1):
            thicknesses.append(np.linalg.norm(brow_pts[i] - brow_pts[i + 1]))
        return np.mean(thicknesses)

    def extract_all(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> Dict:
        """Extract all facial features.

        Returns comprehensive feature dictionary.
        """
        return {
            "eyes": self.extract_eye_features(landmarks),
            "nose": self.extract_nose_features(landmarks),
            "lips": self.extract_lip_features(landmarks),
            "eyebrows": self.extract_eyebrow_features(landmarks),
        }
