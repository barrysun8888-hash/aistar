"""WebGL-based virtual makeup try-on using MediaPipe landmarks."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import cv2


class VirtualTryOn:
    """WebGL-style virtual makeup try-on using facial landmarks."""

    def __init__(self):
        self._makeup_intensity = 1.0

    def set_intensity(self, intensity: float) -> None:
        """Set makeup intensity (0-1)."""
        self._makeup_intensity = max(0, min(1, intensity))

    def apply_lipstick(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        color: Tuple[int, int, int],
        opacity: float = 0.8,
    ) -> np.ndarray:
        """Apply virtual lipstick to image.

        Args:
            image: BGR image
            landmarks: 468-landmark array
            color: RGB color tuple
            opacity: Blend opacity (0-1)

        Returns:
            Image with lipstick applied
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Get lip landmarks
        lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        lip_pts = (landmarks[lip_indices, :2] * np.array([w, h])).astype(np.int32)

        # Create lip mask
        lip_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lip_mask, [lip_pts], 255)

        # Smooth the mask edges
        kernel = np.ones((3, 3), np.uint8)
        lip_mask = cv2.erode(lip_mask, kernel, iterations=1)
        lip_mask = cv2.GaussianBlur(lip_mask, (5, 5), 0)

        # Apply color
        bgr_color = (color[2], color[1], color[0])  # RGB to BGR
        colored_lip = np.full_like(image, bgr_color)

        # Blend
        effective_opacity = opacity * self._makeup_intensity
        result = np.where(
            lip_mask[:, :, np.newaxis] > 0,
            cv2.addWeighted(image, 1 - effective_opacity, colored_lip, effective_opacity, 0),
            result,
        )

        return result

    def apply_eye_makeup(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        shadow_color: Tuple[int, int, int],
        liner_color: Tuple[int, int, int],
        intensity: float = 0.6,
    ) -> np.ndarray:
        """Apply eye makeup (shadow + liner).

        Args:
            image: BGR image
            landmarks: 468-landmark array
            shadow_color: RGB eyeshadow color
            liner_color: RGB eyeliner color
            intensity: Makeup intensity

        Returns:
            Image with eye makeup applied
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Eye landmark regions
        left_eye_indices = list(range(362, 382))
        right_eye_indices = list(range(133, 153))

        # Apply eyeshadow
        for eye_indices in [left_eye_indices, right_eye_indices]:
            eye_pts = (landmarks[eye_indices, :2] * np.array([w, h])).astype(np.int32)

            # Create eye shadow region (above the eye)
            eye_center = eye_pts.mean(axis=0)
            shadow_pts = eye_pts.copy()
            shadow_pts[:, 1] -= 15  # Extend upward

            # Create mask
            shadow_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(shadow_mask, [shadow_pts.astype(np.int32)], 255)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (7, 7), 0)

            # Apply shadow color
            effective_intensity = intensity * self._makeup_intensity * 0.4
            shadow_bgr = (shadow_color[2], shadow_color[1], shadow_color[0])
            shadow_layer = np.full_like(image, shadow_bgr)

            result = np.where(
                shadow_mask[:, :, np.newaxis] > 0,
                cv2.addWeighted(image, 1 - effective_intensity, shadow_layer, effective_intensity, 0),
                result,
            )

        # Apply eyeliner (simplified as line on upper lid)
        for eye_indices in [left_eye_indices, right_eye_indices]:
            eye_pts = (landmarks[eye_indices, :2] * np.array([w, h])).astype(np.int32)

            # Upper lid line
            upper_lid_pts = []
            for i in range(len(eye_indices) // 2):
                upper_lid_pts.append(eye_pts[i])

            if upper_lid_pts:
                effective_intensity = intensity * self._makeup_intensity * 0.7
                liner_bgr = (liner_color[2], liner_color[1], liner_color[0])
                pts = np.array(upper_lid_pts, dtype=np.int32)
                for i in range(len(pts) - 1):
                    cv2.line(result, tuple(pts[i]), tuple(pts[i + 1]), liner_bgr, 2)

        return result

    def apply_blush(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        color: Tuple[int, int, int],
        intensity: float = 0.3,
    ) -> np.ndarray:
        """Apply virtual blush to cheeks.

        Args:
            image: BGR image
            landmarks: 468-landmark array
            color: RGB blush color
            intensity: Makeup intensity

        Returns:
            Image with blush applied
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Cheek landmark positions
        left_cheek = (landmarks[234, :2] * np.array([w, h])).astype(np.int32)
        right_cheek = (landmarks[454, :2] * np.array([w, h])).astype(np.int32)

        # Create soft circular blush
        for cheek in [left_cheek, right_cheek]:
            blush_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(blush_mask, tuple(cheek), 40, 255, -1)
            blush_mask = cv2.GaussianBlur(blush_mask, (21, 21), 0)

            effective_intensity = intensity * self._makeup_intensity * 0.25
            blush_bgr = (color[2], color[1], color[0])
            blush_layer = np.full_like(image, blush_bgr)

            result = np.where(
                blush_mask[:, :, np.newaxis] > 0,
                cv2.addWeighted(image, 1 - effective_intensity, blush_layer, effective_intensity, 0),
                result,
            )

        return result

    def apply_highlight(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        intensity: float = 0.4,
    ) -> np.ndarray:
        """Apply highlighter to face highlights (forehead, nose bridge, chin).

        Args:
            image: BGR image
            landmarks: 468-landmark array
            intensity: Makeup intensity

        Returns:
            Image with highlight applied
        """
        result = image.copy()
        h, w = image.shape[:2]

        highlight_color = (255, 255, 230)  # Slight warm white

        # Highlight regions
        regions = {
            "forehead": [10, 151, 336, 107],  # Center forehead
            "nose_bridge": [1, 2, 4],
            "chin": [8],
        }

        for region_name, indices in regions.items():
            region_pts = (landmarks[indices, :2] * np.array([w, h])).astype(np.int32)
            center = region_pts.mean(axis=0).astype(np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            radius = 25 if region_name == "forehead" else 15
            cv2.circle(mask, tuple(center), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (11, 11), 0)

            effective_intensity = intensity * self._makeup_intensity * 0.2
            highlight_layer = np.full_like(image, highlight_color)

            result = np.where(
                mask[:, :, np.newaxis] > 0,
                cv2.addWeighted(image, 1 - effective_intensity, highlight_layer, effective_intensity, 0),
                result,
            )

        return result

    def apply_contour(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        face_shape: str,
        intensity: float = 0.4,
    ) -> np.ndarray:
        """Apply contouring based on face shape.

        Args:
            image: BGR image
            landmarks: 468-landmark array
            face_shape: One of oval/round/square/heart/diamond
            intensity: Makeup intensity

        Returns:
            Image with contour applied
        """
        result = image.copy()
        h, w = image.shape[:2]

        # Contour shadow color (cool brown)
        shadow_color = (60, 50, 45)

        # Define contour regions based on face shape
        contour_regions = self._get_contour_regions(landmarks, face_shape, h, w)

        for region_pts in contour_regions:
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(region_pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            mask = cv2.GaussianBlur(mask, (9, 9), 0)

            effective_intensity = intensity * self._makeup_intensity * 0.3
            shadow_layer = np.full_like(image, shadow_color)

            result = np.where(
                mask[:, :, np.newaxis] > 0,
                cv2.addWeighted(image, 1 - effective_intensity, shadow_layer, effective_intensity, 0),
                result,
            )

        return result

    def _get_contour_regions(
        self, landmarks: np.ndarray, face_shape: str, h: int, w: int
    ) -> List[np.ndarray]:
        """Get contour region points for face shape."""
        pts = landmarks[:, :2] * np.array([w, h])

        if face_shape == "round":
            # Contour along cheekbones and jaw
            return [
                pts[[234, 454, 366, 136]],  # Jaw line shadow
                pts[[234, 123, 50]],  # Left cheek shadow
                pts[[454, 352, 280]],  # Right cheek shadow
            ]
        elif face_shape == "square":
            # Contour jaw angles
            return [
                pts[[136, 234, 152]],  # Left jaw
                pts[[366, 454, 300]],  # Right jaw
            ]
        elif face_shape == "heart":
            # Contour forehead and upper cheeks
            return [
                pts[[336, 107, 10]],  # Forehead shadow
            ]
        else:
            # Minimal contour for oval/diamond
            return [
                pts[[234, 454, 152]],
            ]

    def generate_before_after(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        makeup_settings: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate before and after images with makeup.

        Args:
            image: Original BGR image
            landmarks: 468-landmark array
            makeup_settings: Dict with colors and intensities

        Returns:
            Tuple of (before_image, after_image)
        """
        before = image.copy()
        after = image.copy()

        # Apply each makeup component
        if "lipstick" in makeup_settings:
            lipstick = makeup_settings["lipstick"]
            after = self.apply_lipstick(
                after, landmarks,
                color=lipstick.get("color", (180, 80, 100)),
                opacity=lipstick.get("opacity", 0.8),
            )

        if "eyeshadow" in makeup_settings:
            shadow = makeup_settings["eyeshadow"]
            after = self.apply_eye_makeup(
                after, landmarks,
                shadow_color=shadow.get("color", (80, 60, 40)),
                liner_color=shadow.get("liner", (30, 30, 30)),
                intensity=shadow.get("intensity", 0.6),
            )

        if "blush" in makeup_settings:
            blush = makeup_settings["blush"]
            after = self.apply_blush(
                after, landmarks,
                color=blush.get("color", (200, 120, 120)),
                intensity=blush.get("intensity", 0.3),
            )

        if "highlight" in makeup_settings and makeup_settings["highlight"].get("enabled", True):
            after = self.apply_highlight(
                after, landmarks,
                intensity=makeup_settings["highlight"].get("intensity", 0.4),
            )

        if "contour" in makeup_settings:
            contour = makeup_settings["contour"]
            after = self.apply_contour(
                after, landmarks,
                face_shape=contour.get("face_shape", "oval"),
                intensity=contour.get("intensity", 0.4),
            )

        return before, after

    def create_comparison_grid(
        self,
        before: np.ndarray,
        after: np.ndarray,
        labels: bool = True,
    ) -> np.ndarray:
        """Create side-by-side comparison grid.

        Args:
            before: Before image
            after: After image
            labels: Whether to add labels

        Returns:
            Comparison image
        """
        h, w = before.shape[:2]

        # Resize to same size
        after_resized = cv2.resize(after, (w, h))

        # Concatenate horizontally
        comparison = np.hstack([before_resized, after_resized])

        if labels:
            # Add labels
            label_before = np.zeros((50, w, 3), dtype=np.uint8) + 255
            label_after = np.zeros((50, w, 3), dtype=np.uint8) + 255

            cv2.putText(label_before, "Before", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(label_after, "After", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            before_row = np.vstack([label_before, before_resized])
            after_row = np.vstack([label_after, after_resized])

            comparison = np.hstack([before_row, after_row])

        return comparison
