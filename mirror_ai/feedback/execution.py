"""Execution detection - measure how well user followed AI recommendations."""

import numpy as np
import cv2
from typing import Dict, List, Tuple


class ExecutionDetector:
    """Detect how well user executed the recommended makeup."""

    def __init__(self):
        self._tolerance = 0.15  # 15% tolerance for angle differences

    def detect_eye_liner_angle(
        self,
        before_landmarks: np.ndarray,
        after_landmarks: np.ndarray,
        recommended_angle: float,
    ) -> Dict:
        """Detect eye liner execution from before/after landmarks."""
        left_eye_outer = before_landmarks[362][:2]
        left_eye_inner = before_landmarks[263][:2]

        before_angle = np.degrees(np.arctan2(
            left_eye_outer[1] - left_eye_inner[1],
            left_eye_outer[0] - left_eye_inner[0]
        ))

        after_angle = np.degrees(np.arctan2(
            after_landmarks[362][1] - after_landmarks[263][1],
            after_landmarks[362][0] - after_landmarks[263][0]
        ))

        angle_change = abs(after_angle - before_angle)
        deviation = abs(angle_change - recommended_angle)

        return {
            "before_angle": round(before_angle, 2),
            "after_angle": round(after_angle, 2),
            "angle_change": round(angle_change, 2),
            "recommended": recommended_angle,
            "deviation": round(deviation, 2),
            "deviation_percent": round((deviation / recommended_angle) * 100, 1) if recommended_angle else 0,
            "execution_score": max(0, 100 - (deviation / recommended_angle * 100)) if recommended_angle else 0,
        }

    def detect_contour_execution(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        face_shape: str,
    ) -> Dict:
        """Detect contour execution by analyzing shadow changes."""
        before_gray = np.mean(before_image, axis=2)
        after_gray = np.mean(after_image, axis=2)

        diff = np.abs(after_gray.astype(float) - before_gray.astype(float))

        h, w = before_image.shape[:2]

        if face_shape == "round":
            left_cheek_region = diff[int(h*0.5):int(h*0.7), int(w*0.1):int(w*0.3)]
            right_cheek_region = diff[int(h*0.5):int(h*0.7), int(w*0.7):int(w*0.9)]
        else:
            left_cheek_region = diff[int(h*0.4):int(h*0.7), int(w*0.05):int(w*0.25)]
            right_cheek_region = diff[int(h*0.4):int(h*0.7), int(w*0.75):int(w*0.95)]

        left_change = np.mean(left_cheek_region)
        right_change = np.mean(right_cheek_region)
        avg_change = (left_change + right_change) / 2

        execution_score = min(100, avg_change * 10)

        return {
            "left_cheek_change": round(left_change, 2),
            "right_cheek_change": round(right_change, 2),
            "average_change": round(avg_change, 2),
            "execution_score": round(execution_score, 1),
            "is_executed": execution_score > 20,
        }

    def detect_lip_color_execution(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        recommended_color: Tuple[int, int, int],
        lip_landmarks: np.ndarray,
    ) -> Dict:
        """Detect if recommended lip color was applied."""
        h, w = after_image.shape[:2]
        lip_pts = (lip_landmarks[:, :2] * np.array([w, h])).astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [lip_pts], 255)

        lip_pixels = after_image[mask > 0]
        if len(lip_pixels) > 0:
            actual_color = lip_pixels.mean(axis=0)[::-1]
        else:
            actual_color = (0, 0, 0)

        color_diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(actual_color, recommended_color)))
        max_diff = np.sqrt(3 * 255 ** 2)

        similarity = 1 - (color_diff / max_diff)
        execution_score = similarity * 100

        return {
            "recommended_color": recommended_color,
            "actual_color": tuple(int(c) for c in actual_color),
            "color_similarity": round(similarity * 100, 1),
            "execution_score": round(execution_score, 1),
        }

    def detect_full_execution(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        before_landmarks: np.ndarray,
        after_landmarks: np.ndarray,
        recommendations: Dict,
    ) -> Dict:
        """Run all execution detections."""
        eye_rec = recommendations.get("eye_makeup", {})
        eye_result = {}
        if "eyeliner_angle" in eye_rec:
            eye_result = self.detect_eye_liner_angle(
                before_landmarks, after_landmarks, eye_rec["eyeliner_angle"]
            )

        face_shape = recommendations.get("face_shape", "oval")
        contour_result = self.detect_contour_execution(before_image, after_image, face_shape)

        lip_rec = recommendations.get("lip_makeup", {})
        lip_result = {}
        if "color" in lip_rec:
            lip_result = self.detect_lip_color_execution(
                before_image, after_image,
                lip_rec["color"],
                after_landmarks,
            )

        scores = []
        if eye_result:
            scores.append(eye_result.get("execution_score", 0))
        scores.append(contour_result.get("execution_score", 0))
        if lip_result:
            scores.append(lip_result.get("execution_score", 0))

        overall_score = sum(scores) / len(scores) if scores else 0

        return {
            "overall_execution_score": round(overall_score, 1),
            "eye_makeup": eye_result,
            "contour": contour_result,
            "lip_makeup": lip_result,
            "passed_threshold": overall_score >= 70,
            "suggestions": self._generate_suggestions(overall_score, eye_result, contour_result, lip_result),
        }

    def _generate_suggestions(
        self,
        overall: float,
        eye_result: Dict,
        contour_result: Dict,
        lip_result: Dict,
    ) -> List[str]:
        """Generate suggestions based on execution detection."""
        suggestions = []

        if overall < 70:
            suggestions.append("整体执行度偏低，建议参考详细教程重新尝试")

        if eye_result and eye_result.get("execution_score", 0) < 60:
            suggestions.append("眼妆角度偏差较大，建议多练习眼线技巧")

        if contour_result and contour_result.get("execution_score", 0) < 50:
            suggestions.append("修容效果不明显，注意高光和阴影的过渡")

        if lip_result and lip_result.get("execution_score", 0) < 60:
            suggestions.append("唇妆颜色与建议有偏差，可参考色板选择替代产品")

        if overall >= 85:
            suggestions.insert(0, "执行度很高！继续保持")

        return suggestions
