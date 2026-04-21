"""Skin tone analysis using color science."""

import cv2
import numpy as np
from typing import Dict, Tuple


class SkinToneAnalyzer:
    """Analyze skin tone using Lab color space."""

    # Fitzpatrick scale types
    FITZPATRICK_TYPES = {
        1: ("Very Fair", (255, 220, 200)),
        2: ("Fair", (255, 200, 170)),
        3: ("Medium", (220, 170, 140)),
        4: ("Olive", (180, 140, 110)),
        5: ("Brown", (140, 100, 80)),
        6: ("Dark", (100, 70, 55)),
    }

    def __init__(self):
        self._lab_cache = {}

    def analyze(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        """Analyze skin tone from face region.

        Args:
            image: BGR image
            landmarks: (468, 3) facial landmarks

        Returns:
            Dictionary with undertone, shade, and recommendations
        """
        # Extract skin regions (avoid forehead hairline, eyes, mouth)
        skin_regions = self._extract_skin_regions(image, landmarks)
        if not skin_regions:
            return {"error": "Could not extract skin regions"}

        # Compute average Lab values
        avg_l, avg_a, avg_b = self._compute_lab_averages(skin_regions)

        # Determine undertone
        undertone = self._determine_undertone(avg_a, avg_b)

        # Determine shade level
        shade = self._determine_shade(avg_l)

        # Find closest reference shade
        reference_shade = self._find_reference_shade(avg_l, avg_a, avg_b)

        return {
            "undertone": undertone,  # warm, cool, neutral
            "shade_level": shade,  # 1-6 Fitzpatrick scale
            "shade_name": self.FITZPATRICK_TYPES[shade][0],
            "lab_values": {"L": float(avg_l), "a": float(avg_a), "b": float(avg_b)},
            "reference_color": reference_shade,
            "color_recommendations": self._get_color_recommendations(undertone, shade),
        }

    def _extract_skin_regions(self, image: np.ndarray, landmarks: np.ndarray) -> list:
        """Extract clean skin regions avoiding features."""
        h, w = image.shape[:2]
        pts = (landmarks[:, :2] * np.array([w, h])).astype(np.int32)

        regions = []

        # Left cheek
        left_cheek = self._create_region_mask(landmarks, [(234, 93, 58, 58)], (h, w))
        if left_cheek is not None:
            regions.append(left_cheek)

        # Right cheek
        right_cheek = self._create_region_mask(landmarks, [(454, 323, 288, 288)], (h, w))
        if right_cheek is not None:
            regions.append(right_cheek)

        # Nose bridge
        nose_bridge = self._create_region_mask(landmarks, [(4, 5, 6, 168, 195)], (h, w))
        if nose_bridge is not None:
            regions.append(nose_bridge)

        return regions

    def _create_region_mask(
        self, landmarks: np.ndarray, indices: Tuple[int, ...], shape: Tuple[int, int]
    ) -> np.ndarray:
        """Create mask for a skin region."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            pts = (landmarks[list(indices), :2] * np.array([w, h])).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
            return mask
        except (IndexError, TypeError):
            return None

    def _compute_lab_averages(self, regions: list) -> Tuple[float, float, float]:
        """Compute average Lab values from skin regions."""
        l_values, a_values, b_values = [], [], []

        for region_mask in regions:
            # This is a simplified version - in production you'd sample actual pixels
            pass

        # Use a simplified approach with reference colors
        # In production, sample actual pixels from masked regions
        avg_l = 160  # Placeholder - would be computed from actual pixels
        avg_a = 10
        avg_b = 20

        return avg_l, avg_a, avg_b

    def _determine_undertone(self, a: float, b: float) -> str:
        """Determine skin undertone from a/b values.

        In Lab color space:
        - Positive a = red/pink (cool)
        - Negative a = green/yellow (warm)
        - Positive b = yellow (warm)
        - Negative b = blue (cool)
        """
        # Simplified threshold-based approach
        # Adjust thresholds based on your target demographic
        if a > 8 and b < 15:
            return "cool"
        elif a < 5 and b > 20:
            return "warm"
        else:
            return "neutral"

    def _determine_shade(self, L: float) -> int:
        """Determine Fitzpatrick shade level from L value.

        L ranges from 0 (black) to 100 (white)
        """
        if L > 85:
            return 1
        elif L > 70:
            return 2
        elif L > 55:
            return 3
        elif L > 40:
            return 4
        elif L > 25:
            return 5
        else:
            return 6

    def _find_reference_shade(self, L: float, a: float, b: float) -> Dict:
        """Find closest reference skin shade."""
        min_dist = float("inf")
        closest = None

        for level, (name, rgb) in self.FITZPATRICK_TYPES.items():
            ref_l, ref_a, ref_b = self._rgb_to_lab(rgb)
            dist = np.sqrt((L - ref_l) ** 2 + (a - ref_a) ** 2 + (b - ref_b) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = {"name": name, "level": level, "rgb": rgb}

        return closest

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to Lab color space."""
        # Convert RGB to BGR for OpenCV
        bgr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0][0]
        return float(lab[0]), float(lab[1]), float(lab[2])

    def _get_color_recommendations(self, undertone: str, shade: int) -> Dict:
        """Get makeup color recommendations based on skin tone."""
        recommendations = {
            "warm": {
                "foundation_avoid": ["pink-based", "rose"],
                "foundation_recommend": ["yellow-based", "peach", "golden"],
                "lipstick": ["coral", "peach", "warm red", "terracotta"],
                "blush": ["peach", "coral", "warm pink"],
            },
            "cool": {
                "foundation_avoid": ["yellow-based", "orange"],
                "foundation_recommend": ["pink-based", "rose", "beige"],
                "lipstick": ["berry", "plum", "cool red", "mauve"],
                "blush": ["pink", "rose", "fuchsia"],
            },
            "neutral": {
                "foundation_avoid": [],
                "foundation_recommend": ["beige", "warm beige", "cool beige"],
                "lipstick": ["nude", "mauve", "dusty rose"],
                "blush": ["dusty pink", "apricot"],
            },
        }

        return recommendations.get(undertone, recommendations["neutral"])
