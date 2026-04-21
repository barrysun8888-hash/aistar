"""Style space mapping for multi-celebrity similarity visualization."""

import numpy as np
from typing import Dict, List, Tuple, Optional


class StyleSpaceMapper:
    """Map faces into a 4D style space for visualization and recommendation."""

    # Style space dimensions
    DIM_CONTOUR = "contour_hardness"  # 轮廓硬度: 柔和 ←→ 锋利
    DIM_VOLUME = "feature_volume"  # 五官量感: 清淡 ←→ 浓艳
    DIM_TEMPERATURE = "temperament"  # 气质温度: 亲和 ←→ 疏离
    DIM_AGE = "age_perception"  # 年龄感知: 幼态 ←→ 成熟

    DIM_LABELS = {
        DIM_CONTOUR: "轮廓硬度",
        DIM_VOLUME: "五官量感",
        DIM_TEMPERATURE: "气质温度",
        DIM_AGE: "年龄感知",
    }

    DIM_RANGES = {
        DIM_CONTOUR: (0, 1),  # 0=soft, 1=sharp
        DIM_VOLUME: (0, 1),  # 0=subtle, 1=bold
        DIM_TEMPERATURE: (0, 1),  # 0=approachable, 1=distant
        DIM_AGE: (0, 1),  # 0=youthful, 1=mature
    }

    def __init__(self):
        self._celebrity_positions: Dict[str, np.ndarray] = {}
        self._celebrity_styles: Dict[str, List[str]] = {}

    def set_celebrity_style(
        self,
        celebrity_id: str,
        position: Tuple[float, float, float, float],
        style_tags: Optional[List[str]] = None,
    ) -> None:
        """Set a celebrity's position in style space.

        Args:
            celebrity_id: Unique identifier
            position: (contour, volume, temperature, age) values 0-1
            style_tags: Associated style tags
        """
        self._celebrity_positions[celebrity_id] = np.array(position)
        self._celebrity_styles[celebrity_id] = style_tags or []

    def compute_user_position(
        self,
        landmarks: np.ndarray,
        face_shape: str,
        eye_type: str,
        features: Dict,
    ) -> np.ndarray:
        """Compute user's position in style space from facial features.

        Args:
            landmarks: 468-landmark array
            face_shape: One of oval/round/square/heart/diamond
            eye_type: One of almond/round/upturned/downturned
            features: Dict with eyes, nose, lips, eyebrows features

        Returns:
            4D position vector
        """
        # Contour hardness (face shape based)
        contour = self._compute_contour_hardness(landmarks, face_shape)

        # Feature volume (eye/lip fullness)
        volume = self._compute_feature_volume(features)

        # Temperament (smile curve, expression)
        temperature = self._compute_temperament(features)

        # Age perception (facial proportions)
        age = self._compute_age_perception(landmarks)

        return np.array([contour, volume, temperature, age])

    def _compute_contour_hardness(self, landmarks: np.ndarray, face_shape: str) -> float:
        """Compute contour hardness score."""
        shape_scores = {
            "round": 0.2,
            "oval": 0.4,
            "diamond": 0.6,
            "heart": 0.7,
            "square": 0.9,
        }
        base = shape_scores.get(face_shape, 0.5)

        # Adjust by jaw sharpness
        jaw_angle = self._compute_jaw_angle(landmarks)
        jaw_score = min(1, jaw_angle / 90)

        return (base + jaw_score) / 2

    def _compute_jaw_angle(self, landmarks: np.ndarray) -> float:
        """Compute jaw angle in degrees."""
        left_jaw = landmarks[136][:2]
        right_jaw = landmarks[366][:2]
        chin = landmarks[8][:2]

        left_angle = np.degrees(np.arctan2(chin[1] - left_jaw[1], chin[0] - left_jaw[0]))
        right_angle = np.degrees(np.arctan2(chin[1] - right_jaw[1], chin[0] - right_jaw[0]))

        return abs(left_angle - right_angle)

    def _compute_feature_volume(self, features: Dict) -> float:
        """Compute feature volume (boldness of features)."""
        scores = []

        # Eye fullness
        if features.get("eyes", {}).get("type") in ["round", "almond"]:
            scores.append(0.6)
        else:
            scores.append(0.4)

        # Lip fullness
        lip_fullness = features.get("lips", {}).get("fullness", "medium")
        fullness_scores = {"thin": 0.2, "medium": 0.5, "full": 0.8}
        scores.append(fullness_scores.get(lip_fullness, 0.5))

        # Nose width
        nose_type = features.get("nose", {}).get("type", "medium")
        if "wide" in nose_type:
            scores.append(0.7)
        elif "narrow" in nose_type:
            scores.append(0.3)
        else:
            scores.append(0.5)

        return np.mean(scores)

    def _compute_temperament(self, features: Dict) -> float:
        """Compute temperament score (approachable vs distant)."""
        # Based on expression and smile
        smile_curve = features.get("expression", {}).get("smile_curve", 0)
        eye_openness = features.get("expression", {}).get("eye_expression", 0.3)

        # Upturned smile + open eyes = approachable
        approachable_score = (smile_curve + eye_openness) / 2

        return np.clip(approachable_score, 0, 1)

    def _compute_age_perception(self, landmarks: np.ndarray) -> float:
        """Compute age perception score."""
        # Forehead to face ratio
        forehead_height = np.linalg.norm(landmarks[10] - landmarks[151])
        face_height = np.linalg.norm(landmarks[10] - landmarks[8])
        forehead_ratio = forehead_height / (face_height + 1e-8)

        # Eye to eyebrow distance (larger = older)
        eye_center = (landmarks[386] + landmarks[159]) / 2
        brow_center = (landmarks[300] + landmarks[70]) / 2
        brow_eye_dist = np.linalg.norm(brow_center - eye_center) / face_height

        # Combine (rough heuristic)
        age_score = (forehead_ratio * 0.5 + brow_eye_dist * 2 * 0.5)

        return np.clip(age_score, 0, 1)

    def compute_similarity_to_celebrity(
        self,
        user_position: np.ndarray,
        celebrity_id: str,
    ) -> float:
        """Compute similarity in style space (inverse distance).

        Args:
            user_position: User's 4D position
            celebrity_id: Target celebrity ID

        Returns:
            Similarity score 0-1
        """
        if celebrity_id not in self._celebrity_positions:
            return 0.0

        celeb_pos = self._celebrity_positions[celebrity_id]
        distance = np.linalg.norm(user_position - celeb_pos)

        # Convert distance to similarity (0 distance = 1, max distance ~3 = 0)
        similarity = 1 - min(1, distance / np.sqrt(4))
        return similarity

    def find_nearest_celebrities(
        self,
        user_position: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find nearest celebrities in style space.

        Returns:
            List of (celebrity_id, similarity) tuples
        """
        similarities = []
        for celeb_id in self._celebrity_positions:
            sim = self.compute_similarity_to_celebrity(user_position, celeb_id)
            similarities.append((celeb_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def get_style_path(
        self,
        user_position: np.ndarray,
        target_celebrity_id: str,
    ) -> Dict:
        """Get path from user to target celebrity in style space.

        Returns:
            Dictionary with path information and recommendations
        """
        if target_celebrity_id not in self._celebrity_positions:
            return {"error": "Celebrity not found"}

        target_pos = self._celebrity_positions[target_celebrity_id]
        diff = target_pos - user_position

        # Dimensions that need most change
        dim_changes = {
            self.DIM_LABELS[dim]: round(float(diff[i]), 2)
            for i, dim in enumerate([self.DIM_CONTOUR, self.DIM_VOLUME, self.DIM_TEMPERATURE, self.DIM_AGE])
        }

        # Sort by magnitude of change needed
        sorted_dims = sorted(dim_changes.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "current_position": user_position.tolist(),
            "target_position": target_pos.tolist(),
            "changes_needed": dict(sorted_dims),
            "primary_change": sorted_dims[0][0] if sorted_dims else None,
        }

    def get_style_space_visualization_data(self) -> Dict:
        """Get data for 2D visualization (project to 2D).

        Returns:
            Dictionary with celebrity positions for plotting
        """
        positions = []
        for celeb_id, pos in self._celebrity_positions.items():
            # Project 4D to 2D using first two dimensions
            positions.append({
                "id": celeb_id,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "styles": self._celebrity_styles.get(celeb_id, []),
            })

        return {
            "celebrities": positions,
            "x_label": self.DIM_LABELS[self.DIM_CONTOUR],
            "y_label": self.DIM_LABELS[self.DIM_VOLUME],
        }
