"""Multi-dimensional similarity calculation between user and celebrities."""

import numpy as np
from typing import Dict, List, Tuple, Optional


class MultiDimSimilarity:
    """Compute multi-dimensional similarity breakdown (bone structure, eyes, nose, lips, overall)."""

    # Landmark indices for each dimension
    BONE_LANDMARKS = [10, 151, 234, 454, 136, 366]  # Face contour points
    EYE_LANDMARKS = list(range(362, 382)) + list(range(133, 153))
    NOSE_LANDMARKS = [1, 2, 4, 5, 6, 98, 168, 195, 327, 329]
    LIP_LANDMARKS = [61, 291, 13, 14, 185, 39, 269, 78, 308]

    def __init__(self):
        self._dim_weights = {
            "bone": 0.30,  # 骨相
            "eyes": 0.25,  # 眼型
            "nose": 0.15,  # 鼻型
            "lips": 0.15,  # 唇形
            "overall": 0.15,  # 整体气质
        }

    def compute_dim_similarity(
        self,
        user_landmarks: np.ndarray,
        celeb_landmarks: np.ndarray,
    ) -> Dict[str, float]:
        """Compute similarity for each facial dimension.

        Args:
            user_landmarks: User's 468-landmark array
            celeb_landmarks: Celebrity's 468-landmark array

        Returns:
            Dictionary with per-dimension similarities (0-1)
        """
        return {
            "bone": self._bone_similarity(user_landmarks, celeb_landmarks),
            "eyes": self._eye_similarity(user_landmarks, celeb_landmarks),
            "nose": self._nose_similarity(user_landmarks, celeb_landmarks),
            "lips": self._lip_similarity(user_landmarks, celeb_landmarks),
        }

    def _normalize_landmarks(self, landmarks: np.ndarray, ref_landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks to same scale and position."""
        # Use face width as reference scale
        user_width = np.linalg.norm(landmarks[234] - landmarks[454])
        ref_width = np.linalg.norm(ref_landmarks[234] - ref_landmarks[454])

        if user_width < 1e-6 or ref_width < 1e-6:
            return landmarks

        scale = ref_width / user_width
        normalized = landmarks.copy()
        normalized[:, :2] *= scale

        # Align centers
        user_center = landmarks[:, :2].mean(axis=0)
        ref_center = ref_landmarks[:, :2].mean(axis=0)
        offset = ref_center - user_center * scale
        normalized[:, :2] += offset

        return normalized

    def _bone_similarity(self, user: np.ndarray, celeb: np.ndarray) -> float:
        """Compute jawline/cheekbone similarity."""
        user_pts = user[self.BONE_LANDMARKS][:, :2]
        celeb_pts = self._normalize_landmarks(user, celeb)[self.BONE_LANDMARKS][:, :2]

        # Compute angles for key bone structure points
        user_angles = self._compute_face_angles(user_pts)
        celeb_angles = self._compute_face_angles(celeb_pts)

        return self._angle_similarity(user_angles, celeb_angles)

    def _compute_face_angles(self, pts: np.ndarray) -> np.ndarray:
        """Compute angles at key face points."""
        # Simplified: compute pairwise angles between contour points
        angles = []
        for i in range(len(pts) - 2):
            v1 = pts[i] - pts[i + 1]
            v2 = pts[i + 2] - pts[i + 1]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
        return np.array(angles)

    def _angle_similarity(self, angles1: np.ndarray, angles2: np.ndarray) -> float:
        """Compare angle distributions."""
        min_len = min(len(angles1), len(angles2))
        if min_len == 0:
            return 0.0

        # Compare first N angles
        a1 = angles1[:min_len]
        a2 = angles2[:min_len]

        # Cosine similarity of angle vectors
        dot = np.dot(a1, a2)
        norm = np.linalg.norm(a1) * np.linalg.norm(a2)
        return (dot / (norm + 1e-8) + 1) / 2  # Normalize to 0-1

    def _eye_similarity(self, user: np.ndarray, celeb: np.ndarray) -> float:
        """Compute eye shape similarity."""
        user_eyes = user[self.EYE_LANDMARKS][:, :2]
        celeb_eyes = self._normalize_landmarks(user, celeb)[self.EYE_LANDMARKS][:, :2]

        # Compare eye aspect ratios
        user_left_ar = self._eye_aspect(user_eyes[:20])
        user_right_ar = self._eye_aspect(user_eyes[20:])
        celeb_left_ar = self._eye_aspect(celeb_eyes[:20])
        celeb_right_ar = self._eye_aspect(celeb_eyes[20:])

        left_sim = 1 - min(1, abs(user_left_ar - celeb_left_ar) / 0.5)
        right_sim = 1 - min(1, abs(user_right_ar - celeb_right_ar) / 0.5)

        return (left_sim + right_sim) / 2

    def _eye_aspect(self, eye_pts: np.ndarray) -> float:
        """Eye aspect ratio (width / height)."""
        width = np.linalg.norm(eye_pts[0] - eye_pts[4])
        height = (np.linalg.norm(eye_pts[1] - eye_pts[5]) + np.linalg.norm(eye_pts[2] - eye_pts[6])) / 2
        return width / (height + 1e-8)

    def _nose_similarity(self, user: np.ndarray, celeb: np.ndarray) -> float:
        """Compute nose shape similarity."""
        user_nose = user[self.NOSE_LANDMARKS][:, :2]
        celeb_nose = self._normalize_landmarks(user, celeb)[self.NOSE_LANDMARKS][:, :2]

        # Compare nose ratios
        user_ratio = self._nose_width_height_ratio(user_nose)
        celeb_ratio = self._nose_width_height_ratio(celeb_nose)

        # Compare nose angles
        user_angle = self._nose_angle(user_nose)
        celeb_angle = self._nose_angle(celeb_nose)

        ratio_sim = 1 - min(1, abs(user_ratio - celeb_ratio) / 0.3)
        angle_sim = 1 - min(1, abs(user_angle - celeb_angle) / 0.5)

        return (ratio_sim + angle_sim) / 2

    def _nose_width_height_ratio(self, pts: np.ndarray) -> float:
        """Nose width to height ratio."""
        width = np.linalg.norm(pts[6] - pts[9])
        height = np.linalg.norm(pts[0] - pts[3])
        return width / (height + 1e-8)

    def _nose_angle(self, pts: np.ndarray) -> float:
        """Nose tip angle."""
        tip = pts[3]
        left = pts[6]
        right = pts[9]
        v1 = left - tip
        v2 = right - tip
        return np.degrees(np.arctan2(v2[1] - v1[1], v2[0] - v1[0]))

    def _lip_similarity(self, user: np.ndarray, celeb: np.ndarray) -> float:
        """Compute lip shape similarity."""
        user_lips = user[self.LIP_LANDMARKS][:, :2]
        celeb_lips = self._normalize_landmarks(user, celeb)[self.LIP_LANDMARKS][:, :2]

        # Compare lip ratios
        user_ratio = self._lip_ratio(user_lips)
        celeb_ratio = self._lip_ratio(celeb_lips)

        # Compare lip width
        user_width = np.linalg.norm(user_lips[0] - user_lips[1])
        celeb_width = np.linalg.norm(celeb_lips[0] - celeb_lips[1])

        ratio_sim = 1 - min(1, abs(user_ratio - celeb_ratio) / 0.2)
        width_sim = 1 - min(1, abs(user_width - celeb_width) / 50)

        return (ratio_sim + width_sim) / 2

    def _lip_ratio(self, pts: np.ndarray) -> float:
        """Lip height to width ratio."""
        height = np.linalg.norm(pts[2] - pts[3])
        width = np.linalg.norm(pts[0] - pts[1])
        return height / (width + 1e-8)

    def compute_overall_similarity(
        self,
        user_embedding: np.ndarray,
        celeb_embedding: np.ndarray,
    ) -> float:
        """Compute overall embedding-based similarity.

        Args:
            user_embedding: 512-dim user embedding
            celeb_embedding: 512-dim celebrity embedding

        Returns:
            Overall similarity score (0-1)
        """
        # Cosine similarity
        sim = np.dot(user_embedding, celeb_embedding)
        return float(sim)

    def compute_full_report(
        self,
        user_landmarks: np.ndarray,
        user_embedding: np.ndarray,
        celeb_landmarks: np.ndarray,
        celeb_embedding: np.ndarray,
        celeb_name: str,
    ) -> Dict:
        """Generate full similarity report.

        Returns:
            Comprehensive similarity breakdown with recommendations
        """
        dim_sim = self.compute_dim_similarity(user_landmarks, celeb_landmarks)
        overall_sim = self.compute_overall_similarity(user_embedding, celeb_embedding)

        # Weighted total
        total = sum(dim_sim[k] * self._dim_weights[k] for k in dim_sim)
        total = 0.85 * total + 0.15 * overall_sim

        # Find strongest/weakest dimensions
        strongest = max(dim_sim, key=dim_sim.get)
        weakest = min(dim_sim, key=dim_sim.get)

        return {
            "celebrity": celeb_name,
            "total_similarity": round(total * 100, 1),
            "dimensions": {
                "bone_similarity": round(dim_sim["bone"] * 100, 1),
                "eye_similarity": round(dim_sim["eyes"] * 100, 1),
                "nose_similarity": round(dim_sim["nose"] * 100, 1),
                "lip_similarity": round(dim_sim["lips"] * 100, 1),
                "overall_similarity": round(overall_sim * 100, 1),
            },
            "strongest_dimension": strongest,
            "weakest_dimension": weakest,
            "recommendations": self._generate_recommendations(dim_sim, celeb_name),
        }

    def _generate_recommendations(self, dim_sim: Dict[str, float], celeb_name: str) -> List[str]:
        """Generate makeup/style recommendations based on similarity gaps."""
        recommendations = []

        if dim_sim["eyes"] < 0.7:
            recommendations.append(f"学习{celeb_name}的眼妆技巧，强化眼神表现力")
        if dim_sim["nose"] < 0.6:
            recommendations.append(f"参考{celeb_name}的鼻影打法，但注意避免过度模仿")
        if dim_sim["lips"] < 0.65:
            recommendations.append(f"尝试{celeb_name}同款唇形塑造方式")

        if dim_sim["bone"] > 0.85:
            recommendations.append(f"骨相与{celeb_name}高度接近，可借鉴其造型风格")

        return recommendations
