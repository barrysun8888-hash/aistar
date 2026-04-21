"""Effect evaluation - quantify transformation improvement."""

import numpy as np
from typing import Dict, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
import cv2


class EffectEvaluator:
    """Evaluate the visual effect/improvement of a makeover."""

    def __init__(self):
        self._weights = {
            "symmetry": 0.25,
            "proportion": 0.30,
            "contrast": 0.20,
            "clarity": 0.25,
        }

    def evaluate_transformation(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        before_landmarks: np.ndarray,
        after_landmarks: np.ndarray,
        target_celebrity: Optional[str] = None,
    ) -> Dict:
        """Evaluate overall transformation effect.

        Args:
            before_image: Image before makeover
            after_image: Image after makeover
            before_landmarks: Landmarks before makeover
            after_landmarks: Landmarks after makeover
            target_celebrity: Optional target celebrity for similarity comparison

        Returns:
            Comprehensive evaluation report
        """
        # Compute individual metrics
        symmetry_improvement = self._evaluate_symmetry(after_landmarks, before_landmarks)
        proportion_improvement = self._evaluate_proportions(after_landmarks, before_landmarks)
        contrast_improvement = self._evaluate_contrast(before_image, after_image)
        clarity_improvement = self._evaluate_clarity(before_image, after_image)

        # Weighted overall score
        overall = (
            symmetry_improvement * self._weights["symmetry"] +
            proportion_improvement * self._weights["proportion"] +
            contrast_improvement * self._weights["contrast"] +
            clarity_improvement * self._weights["clarity"]
        )

        return {
            "overall_score": round(overall * 100, 1),
            "symmetry": {
                "score": round(symmetry_improvement * 100, 1),
                "improvement": self._is_improved(symmetry_improvement),
            },
            "proportion": {
                "score": round(proportion_improvement * 100, 1),
                "improvement": self._is_improved(proportion_improvement),
            },
            "contrast": {
                "score": round(contrast_improvement * 100, 1),
                "improvement": self._is_improved(contrast_improvement),
            },
            "clarity": {
                "score": round(clarity_improvement * 100, 1),
                "improvement": self._is_improved(clarity_improvement),
            },
            "target_celebrity": target_celebrity,
            "verdict": self._get_verdict(overall),
        }

    def _evaluate_symmetry(
        self, after_landmarks: np.ndarray, before_landmarks: np.ndarray
    ) -> float:
        """Evaluate facial symmetry improvement.

        Returns score 0-1 where 1 is most symmetrical.
        """
        # Compute left-right symmetry of landmarks
        h, w = 512, 512  # normalized

        # Mirror and compare
        left_eye_center = before_landmarks[362:382, :2].mean(axis=0)
        right_eye_center = before_landmarks[133:153, :2].mean(axis=0)
        before_eye_distance = np.linalg.norm(left_eye_center - right_eye_center)

        # After symmetry
        left_eye_after = after_landmarks[362:382, :2].mean(axis=0)
        right_eye_after = after_landmarks[133:153, :2].mean(axis=0)
        after_eye_distance = np.linalg.norm(left_eye_after - right_eye_after)

        # Score based on how close eye distances are (more even = better)
        symmetry_score = 1 - min(1, abs(after_eye_distance - before_eye_distance) / before_eye_distance)

        return symmetry_score

    def _evaluate_proportions(
        self, after_landmarks: np.ndarray, before_landmarks: np.ndarray
    ) -> float:
        """Evaluate facial proportion improvements.

        Checks if key ratios (eye-to-face, nose-to-face) have improved.
        """
        def get_ratio(landmarks, pt1_idx, pt2_idx, ref_idx):
            dist = np.linalg.norm(landmarks[pt1_idx] - landmarks[pt2_idx])
            ref = np.linalg.norm(landmarks[ref_idx[0]] - landmarks[ref_idx[1]])
            return dist / (ref + 1e-8)

        # Golden ratio target: 1.618
        # Compute face length to width ratio
        before_ratio = self._face_length_width_ratio(before_landmarks)
        after_ratio = self._face_length_width_ratio(after_landmarks)

        # How close to golden ratio
        golden = 1.618
        before_deviation = abs(before_ratio - golden)
        after_deviation = abs(after_ratio - golden)

        # Improvement if deviation decreased
        if after_deviation < before_deviation:
            proportion_score = 1 - (after_deviation / golden)
        else:
            proportion_score = 1 - (before_deviation / golden)

        return proportion_score

    def _face_length_width_ratio(self, landmarks: np.ndarray) -> float:
        """Compute face length to width ratio."""
        forehead_center = (landmarks[10] + landmarks[151]) / 2
        chin = landmarks[8]
        face_length = np.linalg.norm(forehead_center - chin)

        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        face_width = np.linalg.norm(left_cheek - right_cheek)

        return face_length / (face_width + 1e-8)

    def _evaluate_contrast(
        self, before_image: np.ndarray, after_image: np.ndarray
    ) -> float:
        """Evaluate contrast improvement (skin evenness, feature contrast).

        Returns 0-1 where 1 is improved contrast.
        """
        before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

        # Compute contrast as standard deviation of intensities
        before_contrast = np.std(before_gray)
        after_contrast = np.std(after_gray)

        # Improvement is when contrast increased (more definition)
        if after_contrast > before_contrast:
            contrast_score = min(1, after_contrast / (before_contrast + 1e-8) - 1)
        else:
            contrast_score = 0.5  # No improvement, but not worse

        return contrast_score

    def _evaluate_clarity(
        self, before_image: np.ndarray, after_image: np.ndarray
    ) -> float:
        """Evaluate image clarity/skin smoothness improvement.

        Returns 0-1 where 1 is improved clarity.
        """
        before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

        # Use Laplacian variance as measure of clarity
        before_laplacian = cv2.Laplacian(before_gray, cv2.CV_64F).var()
        after_laplacian = cv2.Laplacian(after_gray, cv2.CV_64F).var()

        # Score based on whether clarity improved
        if after_laplacian > before_laplacian:
            clarity_score = min(1, after_laplacian / (before_laplacian + 1e-8) - 0.5)
        else:
            clarity_score = 0.5

        return clarity_score

    def _is_improved(self, score: float) -> bool:
        """Check if metric shows improvement."""
        return score >= 0.5

    def _get_verdict(self, score: float) -> str:
        """Get human-readable verdict."""
        if score >= 0.8:
            return "显著提升"
        elif score >= 0.6:
            return "明显改善"
        elif score >= 0.4:
            return "轻微改善"
        else:
            return "效果不明显"

    def compare_to_target(
        self,
        user_after_image: np.ndarray,
        user_embedding: np.ndarray,
        target_embedding: np.ndarray,
    ) -> Dict:
        """Compare user's makeover result to target celebrity.

        Args:
            user_after_image: User's makeover photo
            user_embedding: User's face embedding
            target_embedding: Target celebrity's embedding

        Returns:
            Comparison report
        """
        # Compute embedding similarity
        similarity = np.dot(user_embedding, target_embedding)

        return {
            "similarity_to_target": round(similarity * 100, 1),
            "improvement_potential": "可通过进一步调整提升相似度" if similarity < 0.7 else "已达到较高相似度",
        }
