# Mirror AI - 智能形象进化系统
# Dependencies: mediapipe>=0.10.0, numpy>=1.24.0, Pillow>=10.0.0, streamlit>=1.28.0

"""Streamlit app for Mirror AI - 智能形象进化系统."""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import warnings
import os

# Page config
st.set_page_config(
    page_title="镜AI - 智能形象进化系统",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Face Analysis Modules
# ============================================================

class FaceLandmarks:
    """Extract facial landmarks using MediaPipe."""

    def __init__(self):
        self._mp_face_mesh = None
        self._face_mesh = None
        self._initialize()

    def _initialize(self):
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except Exception:
            pass

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract 468 facial landmarks."""
        if self._face_mesh is None:
            return None

        rgb_image = image.astype(np.uint8)
        results = self._face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            return np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        return None


class FaceShapeClassifier:
    """Classify face shape using landmarks."""

    FACE_SHAPES = ["oval", "round", "square", "heart", "diamond"]

    def classify(self, landmarks: np.ndarray, image_shape) -> str:
        """Classify face shape."""
        h, w = image_shape[:2]
        pts = landmarks[:, :2] * np.array([w, h])

        jaw_width = np.linalg.norm(pts[234] - pts[454])
        face_height = np.linalg.norm(pts[10] - pts[8])
        cheek_width = np.linalg.norm(pts[50] - pts[280])

        ratio = face_height / (jaw_width + 1e-8)

        if ratio > 1.4:
            if cheek_width / jaw_width > 1.2:
                return "diamond"
            return "oval"
        elif ratio > 1.2:
            return "heart"
        else:
            if self._is_round(pts):
                return "round"
            return "square"

    def _is_round(self, pts) -> bool:
        """Check if face is round."""
        jaw_angles = self._compute_jaw_angles(pts)
        return np.std(jaw_angles) < 5

    def _compute_jaw_angles(self, pts):
        """Compute angles along jawline."""
        jaw_indices = list(range(234, 264)) + list(range(264, 294)) + list(range(294, 324))
        angles = []
        for i in range(len(jaw_indices) - 2):
            p1 = pts[jaw_indices[i]]
            p2 = pts[jaw_indices[i + 1]]
            p3 = pts[jaw_indices[i + 2]]
            v1 = p1 - p2
            v2 = p3 - p2
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
            angles.append(np.degrees(angle))
        return angles

    def get_golden_ratio_analysis(self, landmarks: np.ndarray, image_shape) -> dict:
        """Analyze deviation from golden ratio."""
        h, w = image_shape[:2]
        pts = landmarks[:, :2] * np.array([w, h])

        forehead_center = (pts[10] + pts[151]) / 2
        chin = pts[8]
        face_length = np.linalg.norm(forehead_center - chin)

        left_cheek = pts[234]
        right_cheek = pts[454]
        face_width = np.linalg.norm(left_cheek - right_cheek)

        ratio = face_length / (face_width + 1e-8)
        golden = 1.618
        deviation = abs(ratio - golden) / golden * 100

        return {
            "ratio": round(ratio, 3),
            "deviation_from_golden": round(deviation, 2),
            "is_ideal_proportion": deviation < 10,
        }


class FeatureExtractor:
    """Extract facial feature characteristics."""

    def extract_all(self, landmarks: np.ndarray, image_shape) -> dict:
        """Extract all facial features."""
        return {
            "eyes": self._analyze_eyes(landmarks),
            "nose": self._analyze_nose(landmarks),
            "lips": self._analyze_lips(landmarks),
            "eyebrows": self._analyze_eyebrows(landmarks),
        }

    def _analyze_eyes(self, landmarks: np.ndarray) -> dict:
        """Analyze eye shape."""
        left_eye_width = np.linalg.norm(landmarks[362] - landmarks[263])
        eye_height = np.linalg.norm(landmarks[159] - landmarks[145])

        ratio = eye_height / (left_eye_width + 1e-8)

        if ratio > 0.4:
            eye_type = "圆眼"
        elif ratio > 0.3:
            eye_type = "杏仁眼"
        else:
            eye_type = "细长眼"

        return {"type": eye_type, "height_ratio": round(ratio, 2)}

    def _analyze_nose(self, landmarks: np.ndarray) -> dict:
        """Analyze nose shape."""
        nose_bridge = landmarks[4]
        nose_tip = landmarks[1]
        nose_width = np.linalg.norm(landmarks[234] - landmarks[454])

        bridge_length = np.linalg.norm(nose_bridge - nose_tip)
        width_ratio = nose_width / (bridge_length + 1e-8)

        if width_ratio > 0.5:
            nose_type = "宽鼻"
        else:
            nose_type = "窄鼻"

        return {"type": nose_type, "width_ratio": round(width_ratio, 2)}

    def _analyze_lips(self, landmarks: np.ndarray) -> dict:
        """Analyze lip shape."""
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        lip_height = np.linalg.norm(upper_lip - lower_lip)

        lip_width = np.linalg.norm(landmarks[61] - landmarks[291])
        fullness = lip_height / (lip_width + 1e-8)

        if fullness > 0.3:
            fullness_label = "厚唇"
        else:
            fullness_label = "薄唇"

        return {"fullness": fullness_label, "fullness_ratio": round(fullness, 2)}

    def _analyze_eyebrows(self, landmarks: np.ndarray) -> dict:
        """Analyze eyebrow shape."""
        left_brow_start = landmarks[336]
        left_brow_end = landmarks[296]

        brow_angle = np.degrees(np.arctan2(
            left_brow_end[1] - left_brow_start[1],
            left_brow_end[0] - left_brow_start[0]
        ))

        if brow_angle > 15:
            thickness = "上扬眉"
        elif brow_angle > 5:
            thickness = "标准眉"
        else:
            thickness = "平眉"

        return {"thickness": thickness, "angle": round(brow_angle, 1)}


class SkinToneAnalyzer:
    """Analyze skin tone using color science."""

    FITZPATRICK_TYPES = {
        1: ("Very Fair", (255, 220, 200)),
        2: ("Fair", (255, 200, 170)),
        3: ("Medium", (220, 170, 140)),
        4: ("Olive", (180, 140, 110)),
        5: ("Brown", (140, 100, 80)),
        6: ("Dark", (100, 70, 55)),
    }

    def analyze(self, image: np.ndarray, landmarks: np.ndarray) -> dict:
        """Analyze skin tone from face region."""
        h, w = image.shape[:2]
        pts = landmarks[:, :2] * np.array([w, h])

        samples = []
        for idx in [234, 454, 4]:
            x, y = int(pts[idx, 0]), int(pts[idx, 1])
            for dx in range(-10, 10, 2):
                for dy in range(-10, 10, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        samples.append(image[ny, nx])

        if not samples:
            return {"error": "Could not extract skin color"}

        samples = np.array(samples)
        avg_rgb = samples.mean(axis=0)

        r, g, b = avg_rgb[0], avg_rgb[1], avg_rgb[2]
        l = 0.299 * r + 0.587 * g + 0.114 * b
        l_norm = l / 255.0
        a = 128 + (r - b) * 0.5
        b_val = 128 + (g - b) * 0.4

        undertone = self._determine_undertone(a - 128, b_val - 128)
        shade = self._determine_shade(l_norm * 100)

        return {
            "undertone": undertone,
            "shade_level": shade,
            "shade_name": self.FITZPATRICK_TYPES[shade][0],
            "color_recommendations": self._get_color_recommendations(undertone, shade),
        }

    def _determine_undertone(self, a: float, b: float) -> str:
        """Determine skin undertone."""
        if a > 8 and b < 15:
            return "cool"
        elif a < 5 and b > 20:
            return "warm"
        return "neutral"

    def _determine_shade(self, L: float) -> int:
        """Determine Fitzpatrick shade level."""
        if L > 85: return 1
        elif L > 70: return 2
        elif L > 55: return 3
        elif L > 40: return 4
        elif L > 25: return 5
        return 6

    def _get_color_recommendations(self, undertone: str, shade: int) -> dict:
        """Get makeup color recommendations."""
        recommendations = {
            "warm": {
                "lipstick": ["coral", "peach", "warm red", "terracotta"],
                "blush": ["peach", "coral", "warm pink"],
            },
            "cool": {
                "lipstick": ["berry", "plum", "cool red", "mauve"],
                "blush": ["pink", "rose", "fuchsia"],
            },
            "neutral": {
                "lipstick": ["nude", "mauve", "dusty rose"],
                "blush": ["dusty pink", "apricot"],
            },
        }
        return recommendations.get(undertone, recommendations["neutral"])


class ExpressionAnalyzer:
    """Analyze facial expression."""

    def analyze(self, landmarks: np.ndarray) -> dict:
        """Analyze expression from landmarks."""
        smile_angle = self._compute_smile_angle(landmarks)
        eye_openness = self._compute_eye_openness(landmarks)

        if smile_angle > 15 and eye_openness > 0.3:
            classified = "happy"
        elif eye_openness < 0.15:
            classified = "serious"
        else:
            classified = "neutral"

        return {
            "classified_expression": classified,
            "smile_angle": round(smile_angle, 1),
            "eye_openness": round(eye_openness, 2),
        }

    def _compute_smile_angle(self, landmarks) -> float:
        """Compute smile angle from mouth corners."""
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        mouth_center = landmarks[13]

        angle = np.degrees(np.arctan2(
            mouth_center[1] - (left_corner[1] + right_corner[1]) / 2,
            mouth_center[0] - (left_corner[0] + right_corner[0]) / 2
        ))
        return abs(angle)

    def _compute_eye_openness(self, landmarks) -> float:
        """Compute eye openness ratio."""
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        eye_width = np.linalg.norm(landmarks[362] - landmarks[263])
        return eye_height / (eye_width + 1e-8)


# ============================================================
# Recommendation Modules
# ============================================================

class MakeupRecommender:
    """Generate makeup recommendations."""

    def recommend(self, face_shape: str, features: dict, skin_tone: dict) -> dict:
        """Generate makeup recommendations."""
        skin_recs = skin_tone.get("color_recommendations", {})
        lip_colors = skin_recs.get("lipstick", ["coral"])

        recommendations = {
            "optimize_track": {
                "strategy": "自然放大原生优势",
                "focus_advice": f"强调{features.get('eyes', {}).get('type', '标准眼型')}的独特美感",
                "lip_color": lip_colors[0] if lip_colors else "coral",
            },
            "breakthrough_track": {
                "strategy": "突破性风格尝试",
                "crossover_tip": f"参考{face_shape}脸的经典造型，融合现代元素",
            },
        }
        return recommendations


class HairstyleRecommender:
    """Generate hairstyle recommendations."""

    def recommend(self, face_shape: str, forehead_height: str = "normal", hair_texture: str = "medium") -> dict:
        """Generate hairstyle recommendations."""
        recommendations = {
            "oval": {"recommended_lengths": ["长发", "中长发"], "bangs_advice": {"shape_advice": "可尝试八字刘海"}},
            "round": {"recommended_lengths": ["长发", "锁骨发"], "bangs_advice": {"shape_advice": "侧分长刘海显脸长"}},
            "square": {"recommended_lengths": ["长发", "层次感长发"], "bangs_advice": {"shape_advice": "斜刘海柔化轮廓"}},
            "heart": {"recommended_lengths": ["中长发", "bob头"], "bangs_advice": {"shape_advice": "空气刘海平衡额头"}},
            "diamond": {"recommended_lengths": ["短发", "bob头"], "bangs_advice": {"shape_advice": "侧分增加宽度"}},
        }
        return recommendations.get(face_shape, recommendations["oval"])


class LightingRecommender:
    """Generate lighting recommendations."""

    def recommend(self, face_shape: str, skin_tone: dict = None) -> dict:
        """Generate lighting recommendations."""
        recommendations = {
            "oval": {"primary_lighting": {"name": "侧光45度", "effect": "增强立体感"}},
            "round": {"primary_lighting": {"name": "侧上方光", "effect": "拉长脸部线条"}},
            "square": {"primary_lighting": {"name": "环形光", "effect": "柔化轮廓边缘"}},
            "heart": {"primary_lighting": {"name": "蝴蝶光", "effect": "平衡上下比例"}},
            "diamond": {"primary_lighting": {"name": "三点布光", "effect": "全面突出五官"}},
        }
        return recommendations.get(face_shape, recommendations["oval"])


class StyleProfiler:
    """Profile style season."""

    def profile(self, skin_tone: dict, features: dict, expression: dict) -> dict:
        """Profile color season."""
        undertone = skin_tone.get("undertone", "neutral")

        seasons = {
            "warm": {"season_name": "春季型", "characteristics": ["活力", "明亮", "温暖"], "color_palette": {"primary": ["珊瑚", "蜜桃", "暖金"]}},
            "cool": {"season_name": "冬季型", "characteristics": ["冷艳", "对比", "鲜明"], "color_palette": {"primary": ["莓红", "紫罗兰", "冰蓝"]}},
            "neutral": {"season_name": "秋季型", "characteristics": ["柔和", "自然", "优雅"], "color_palette": {"primary": ["焦糖", "砖红", "橄榄"]}},
        }
        return seasons.get(undertone, seasons["neutral"])


# ============================================================
# Preview Modules
# ============================================================

class VirtualTryOn:
    """WebGL-style virtual makeup try-on."""

    def __init__(self):
        self._makeup_intensity = 1.0

    def set_intensity(self, intensity: float) -> None:
        """Set makeup intensity."""
        self._makeup_intensity = max(0, min(1, intensity))

    def apply_lipstick(self, image: np.ndarray, landmarks: np.ndarray, color: tuple, opacity: float = 0.8) -> np.ndarray:
        """Apply virtual lipstick."""
        result = image.copy()
        h, w = image.shape[:2]

        lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        lip_pts = (landmarks[lip_indices, :2] * np.array([w, h])).astype(int)

        pil_img = Image.fromarray(result)
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon([tuple(p) for p in lip_pts], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

        effective_opacity = int(opacity * self._makeup_intensity * 255)
        color_layer = Image.new('RGB', (w, h), (color[2], color[1], color[0]))
        color_layer.putalpha(mask)

        pil_result = Image.fromarray(result)
        pil_result = Image.composite(color_layer, pil_result, mask)

        return np.array(pil_result)

    def apply_blush(self, image: np.ndarray, landmarks: np.ndarray, color: tuple, intensity: float = 0.3) -> np.ndarray:
        """Apply virtual blush."""
        result = image.copy()
        h, w = image.shape[:2]

        left_cheek = (landmarks[234, :2] * np.array([w, h])).astype(int)
        right_cheek = (landmarks[454, :2] * np.array([w, h])).astype(int)

        for cheek in [left_cheek, right_cheek]:
            blush_mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(blush_mask)
            x, y = int(cheek[0]), int(cheek[1])
            draw.ellipse([x-40, y-40, x+40, y+40], fill=int(intensity * self._makeup_intensity * 100))
            blush_mask = blush_mask.filter(ImageFilter.GaussianBlur(radius=30))

            blush_layer = Image.new('RGB', (w, h), (color[2], color[1], color[0]))
            blush_layer.putalpha(blush_mask)

            pil_result = Image.fromarray(result)
            pil_result = Image.composite(blush_layer, pil_result, blush_mask)
            result = np.array(pil_result)

        return result

    def apply_highlight(self, image: np.ndarray, landmarks: np.ndarray, intensity: float = 0.4) -> np.ndarray:
        """Apply highlighter."""
        result = image.copy()
        h, w = image.shape[:2]
        highlight_color = (255, 255, 230)

        regions = {"forehead": [10, 151, 336, 107], "nose_bridge": [1, 2, 4], "chin": [8]}

        for region_name, indices in regions.items():
            region_pts = (landmarks[indices, :2] * np.array([w, h])).astype(int)
            center = region_pts.mean(axis=0).astype(int)
            x, y = int(center[0]), int(center[1])

            radius = 25 if region_name == "forehead" else 15

            highlight_mask = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(highlight_mask)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=int(intensity * self._makeup_intensity * 80))
            highlight_mask = highlight_mask.filter(ImageFilter.GaussianBlur(radius=15))

            highlight_layer = Image.new('RGB', (w, h), highlight_color)
            highlight_layer.putalpha(highlight_mask)

            pil_result = Image.fromarray(result)
            pil_result = Image.composite(highlight_layer, pil_result, highlight_mask)
            result = np.array(pil_result)

        return result

    def generate_before_after(self, image: np.ndarray, landmarks: np.ndarray, makeup_settings: dict):
        """Generate before and after images."""
        before = image.copy()
        after = image.copy()

        if "blush" in makeup_settings:
            blush = makeup_settings["blush"]
            after = self.apply_blush(after, landmarks, color=blush.get("color", (200, 120, 120)), intensity=blush.get("intensity", 0.3))

        if "highlight" in makeup_settings and makeup_settings["highlight"].get("enabled", True):
            after = self.apply_highlight(after, landmarks, intensity=makeup_settings["highlight"].get("intensity", 0.4))

        return before, after

    def create_comparison_grid(self, before: np.ndarray, after: np.ndarray, labels: bool = True) -> np.ndarray:
        """Create side-by-side comparison grid."""
        h, w = before.shape[:2]
        after_resized = np.array(Image.fromarray(after).resize((w, h)))

        if labels:
            h_label = 40
            label_before = np.full((h_label, w, 3), 255, dtype=np.uint8)
            label_after = np.full((h_label, w, 3), 255, dtype=np.uint8)

            draw_before = ImageDraw.Draw(Image.fromarray(label_before))
            draw_after = ImageDraw.Draw(Image.fromarray(label_after))
            draw_before.text((10, 10), "Before", fill=(0, 0, 0))
            draw_after.text((10, 10), "After", fill=(0, 0, 0))

            before_row = np.vstack([label_before, before])
            after_row = np.vstack([label_after, after_resized])
        else:
            before_row = before
            after_row = after_resized

        return np.hstack([before_row, after_row])


# ============================================================
# Feedback Modules
# ============================================================

class EffectEvaluator:
    """Evaluate the visual effect of a makeover."""

    def __init__(self):
        self._weights = {"symmetry": 0.25, "proportion": 0.30, "contrast": 0.20, "clarity": 0.25}

    def evaluate_transformation(self, before_image: np.ndarray, after_image: np.ndarray, before_landmarks: np.ndarray, after_landmarks: np.ndarray, target_celebrity: str = None) -> dict:
        """Evaluate overall transformation effect."""
        symmetry_improvement = self._evaluate_symmetry(after_landmarks, before_landmarks)
        proportion_improvement = self._evaluate_proportions(after_landmarks, before_landmarks)
        contrast_improvement = self._evaluate_contrast(before_image, after_image)
        clarity_improvement = self._evaluate_clarity(before_image, after_image)

        overall = (
            symmetry_improvement * self._weights["symmetry"] +
            proportion_improvement * self._weights["proportion"] +
            contrast_improvement * self._weights["contrast"] +
            clarity_improvement * self._weights["clarity"]
        )

        return {
            "overall_score": round(overall * 100, 1),
            "symmetry": {"score": round(symmetry_improvement * 100, 1), "improvement": symmetry_improvement >= 0.5},
            "proportion": {"score": round(proportion_improvement * 100, 1), "improvement": proportion_improvement >= 0.5},
            "contrast": {"score": round(contrast_improvement * 100, 1), "improvement": contrast_improvement >= 0.5},
            "clarity": {"score": round(clarity_improvement * 100, 1), "improvement": clarity_improvement >= 0.5},
            "target_celebrity": target_celebrity,
            "verdict": self._get_verdict(overall),
        }

    def _evaluate_symmetry(self, after_landmarks, before_landmarks) -> float:
        """Evaluate facial symmetry."""
        left_eye_center = before_landmarks[362:382, :2].mean(axis=0)
        right_eye_center = before_landmarks[133:153, :2].mean(axis=0)
        before_eye_distance = np.linalg.norm(left_eye_center - right_eye_center)

        left_eye_after = after_landmarks[362:382, :2].mean(axis=0)
        right_eye_after = after_landmarks[133:153, :2].mean(axis=0)
        after_eye_distance = np.linalg.norm(left_eye_after - right_eye_after)

        symmetry_score = 1 - min(1, abs(after_eye_distance - before_eye_distance) / before_eye_distance)
        return symmetry_score

    def _evaluate_proportions(self, after_landmarks, before_landmarks) -> float:
        """Evaluate facial proportions."""
        before_ratio = self._face_length_width_ratio(before_landmarks)
        after_ratio = self._face_length_width_ratio(after_landmarks)

        golden = 1.618
        before_deviation = abs(before_ratio - golden)
        after_deviation = abs(after_ratio - golden)

        if after_deviation < before_deviation:
            return 1 - (after_deviation / golden)
        return 1 - (before_deviation / golden)

    def _face_length_width_ratio(self, landmarks) -> float:
        """Compute face length to width ratio."""
        forehead_center = (landmarks[10] + landmarks[151]) / 2
        chin = landmarks[8]
        face_length = np.linalg.norm(forehead_center - chin)
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        face_width = np.linalg.norm(left_cheek - right_cheek)
        return face_length / (face_width + 1e-8)

    def _evaluate_contrast(self, before_image, after_image) -> float:
        """Evaluate contrast."""
        before_gray = np.mean(before_image, axis=2)
        after_gray = np.mean(after_image, axis=2)
        before_contrast = np.std(before_gray)
        after_contrast = np.std(after_gray)

        if after_contrast > before_contrast:
            return min(1, after_contrast / (before_contrast + 1e-8) - 1)
        return 0.5

    def _evaluate_clarity(self, before_image, after_image) -> float:
        """Evaluate image clarity."""
        before_gray = np.mean(before_image, axis=2)
        after_gray = np.mean(after_image, axis=2)

        before_laplacian = np.var(np.diff(before_gray, axis=0)) + np.var(np.diff(before_gray, axis=1))
        after_laplacian = np.var(np.diff(after_gray, axis=0)) + np.var(np.diff(after_gray, axis=1))

        if after_laplacian > before_laplacian:
            return min(1, after_laplacian / (before_laplacian + 1e-8) - 0.5)
        return 0.5

    def _get_verdict(self, score: float) -> str:
        """Get human-readable verdict."""
        if score >= 0.8: return "显著提升"
        elif score >= 0.6: return "明显改善"
        elif score >= 0.4: return "轻微改善"
        return "效果不明显"


class ExecutionDetector:
    """Detect how well user executed the recommended makeup."""

    def __init__(self):
        self._tolerance = 0.15

    def detect_eye_liner_angle(self, before_landmarks, after_landmarks, recommended_angle: float) -> dict:
        """Detect eye liner execution."""
        left_eye_outer = before_landmarks[362][:2]
        left_eye_inner = before_landmarks[263][:2]

        before_angle = np.degrees(np.arctan2(left_eye_outer[1] - left_eye_inner[1], left_eye_outer[0] - left_eye_inner[0]))
        after_angle = np.degrees(np.arctan2(after_landmarks[362][1] - after_landmarks[263][1], after_landmarks[362][0] - after_landmarks[263][0]))

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

    def detect_contour_execution(self, before_image, after_image, face_shape: str) -> dict:
        """Detect contour execution."""
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

    def detect_lip_color_execution(self, before_image, after_image, recommended_color: tuple, lip_landmarks) -> dict:
        """Detect lip color execution."""
        h, w = after_image.shape[:2]
        lip_pts = (lip_landmarks[:, :2] * np.array([w, h])).astype(int)

        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon([tuple(p) for p in lip_pts], fill=255)

        pil_after = Image.fromarray(after_image)
        lip_pixels = np.array(pil_after)[np.array(mask) > 0]

        if len(lip_pixels) > 0:
            actual_color = lip_pixels[:, :3].mean(axis=0)[::-1]
        else:
            actual_color = (0, 0, 0)

        color_diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(actual_color, recommended_color)))
        max_diff = np.sqrt(3 * 255 ** 2)
        similarity = 1 - (color_diff / max_diff)

        return {
            "recommended_color": recommended_color,
            "actual_color": tuple(int(c) for c in actual_color),
            "color_similarity": round(similarity * 100, 1),
            "execution_score": round(similarity * 100, 1),
        }

    def detect_full_execution(self, before_image, after_image, before_landmarks, after_landmarks, recommendations: dict) -> dict:
        """Run all execution detections."""
        eye_rec = recommendations.get("eye_makeup", {})
        eye_result = {}
        if "eyeliner_angle" in eye_rec:
            eye_result = self.detect_eye_liner_angle(before_landmarks, after_landmarks, eye_rec["eyeliner_angle"])

        face_shape = recommendations.get("face_shape", "oval")
        contour_result = self.detect_contour_execution(before_image, after_image, face_shape)

        lip_rec = recommendations.get("lip_makeup", {})
        lip_result = {}
        if "color" in lip_rec:
            lip_result = self.detect_lip_color_execution(before_image, after_image, lip_rec["color"], after_landmarks)

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

    def _generate_suggestions(self, overall: float, eye_result: dict, contour_result: dict, lip_result: dict) -> list:
        """Generate suggestions."""
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


# ============================================================
# Session State
# ============================================================

def init_session_state():
    """Initialize session state variables."""
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "landmarks" not in st.session_state:
        st.session_state.landmarks = None
    if "image" not in st.session_state:
        st.session_state.image = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None


@st.cache_resource
def load_models():
    """Load ML models (cached for performance)."""
    models = {}
    try:
        models["landmarks"] = FaceLandmarks()
    except Exception as e:
        st.warning(f"FaceLandmarks加载失败: {e}")
        return None

    try:
        models["face_shape"] = FaceShapeClassifier()
        models["features"] = FeatureExtractor()
        models["skin_tone"] = SkinToneAnalyzer()
        models["expression"] = ExpressionAnalyzer()
        models["makeup"] = MakeupRecommender()
        models["hairstyle"] = HairstyleRecommender()
        models["lighting"] = LightingRecommender()
        models["style"] = StyleProfiler()
        models["virtual_tryon"] = VirtualTryOn()
    except Exception as e:
        st.warning(f"部分模型加载失败: {e}")

    return models if models.get("landmarks") else None


def analyze_face(image: np.ndarray, models: dict) -> dict:
    """Perform full face analysis."""
    results = {}

    landmarks = models["landmarks"].extract(image)
    if landmarks is None:
        return {"error": "未检测到人脸，请上传清晰的正面照片"}

    results["landmarks"] = landmarks
    results["face_shape"] = models["face_shape"].classify(landmarks, image.shape)
    results["golden_ratio"] = models["face_shape"].get_golden_ratio_analysis(landmarks, image.shape)
    results["features"] = models["features"].extract_all(landmarks, image.shape)
    results["skin_tone"] = models["skin_tone"].analyze(image, landmarks)
    results["expression"] = models["expression"].analyze(landmarks)

    return results


# ============================================================
# Main App
# ============================================================

def main():
    init_session_state()

    st.markdown('<p class="main-header">🪞 镜AI 智能形象进化系统</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">不是告诉你"像哪个明星"，而是告诉你"如何成为更好的自己"</p>', unsafe_allow_html=True)

    models = load_models()
    if models is None:
        st.error("核心模型加载失败，请刷新重试")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📤 上传照片")

        uploaded_file = st.file_uploader("选择一张清晰的正面照片", type=["jpg", "jpeg", "png"], help="请上传清晰的正面照，避免侧脸、遮挡或模糊照片")

        if uploaded_file:
            pil_image = Image.open(uploaded_file)
            image = np.array(pil_image.convert('RGB'))
            st.session_state.image = image
            st.image(pil_image, caption="上传的照片", use_container_width=True)

        if st.session_state.image is not None and st.button("🔍 开始分析", type="primary"):
            with st.spinner("分析中..."):
                results = analyze_face(st.session_state.image, models)
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.session_state.analysis_results = results
                    st.session_state.analysis_complete = True
                    st.success("分析完成！")
                    st.rerun()

    with col2:
        if st.session_state.analysis_complete and st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.subheader("📊 分析报告")

            face_shape = results.get("face_shape", "unknown")
            col_shape, col_skin = st.columns(2)
            with col_shape:
                st.metric("脸型", face_shape)
            with col_skin:
                skin_tone = results.get("skin_tone", {})
                st.metric("肤色", f"Type {skin_tone.get('shade_level', '?')}")

            golden = results.get("golden_ratio", {})
            if golden.get("is_ideal_proportion"):
                st.success("✓ 面部比例接近黄金比例")
            else:
                st.info(f"面部比例偏离黄金比例 {golden.get('deviation_from_golden', 0):.2f}")

            features = results.get("features", {})
            st.markdown("### 👁 五官特征")

            eye_type = features.get("eyes", {}).get("type", "unknown")
            lip_fullness = features.get("lips", {}).get("fullness", "unknown")
            brow_info = features.get("eyebrows", {})

            col_eye, col_lip, col_brow = st.columns(3)
            with col_eye:
                st.write(f"**眼型**: {eye_type}")
            with col_lip:
                st.write(f"**唇形**: {lip_fullness}")
            with col_brow:
                st.write(f"**眉形**: {brow_info.get('thickness', 'unknown')}")

            expression = results.get("expression", {})
            st.markdown(f"**表情特征**: {expression.get('classified_expression', 'neutral')}")

            if skin_tone:
                st.markdown("### 🎨 肤色分析")
                st.write(f"**Undertone**: {skin_tone.get('undertone', 'unknown')}")
                st.write(f"**肤色等级**: {skin_tone.get('shade_name', 'unknown')}")
                recommendations = skin_tone.get("color_recommendations", {})
                if recommendations:
                    lip_recs = recommendations.get('lipstick', [])[:3]
                    st.write(f"**推荐口红色**: {', '.join(lip_recs) if lip_recs else '暂无'}")

            st.markdown("---")
            st.markdown("### 💄 妆容建议")

            makeup_recs = models["makeup"].recommend(face_shape=face_shape, features=features, skin_tone=skin_tone)

            tab1, tab2 = st.tabs(["优化轨", "突破轨"])
            with tab1:
                optimize = makeup_recs.get("optimize_track", {})
                st.markdown(f"**策略**: {optimize.get('strategy', '')}")
                st.markdown(f"**重点**: {optimize.get('focus_advice', '')}")
            with tab2:
                breakthrough = makeup_recs.get("breakthrough_track", {})
                st.markdown(f"**策略**: {breakthrough.get('strategy', '')}")
                st.markdown(f"**建议**: {breakthrough.get('crossover_tip', '')}")

            st.markdown("---")
            st.markdown("### ✨ 虚拟试妆")

            landmarks = results.get("landmarks")
            if landmarks is not None:
                makeup_settings = {"blush": {"color": (200, 120, 120), "intensity": 0.3}, "highlight": {"enabled": True, "intensity": 0.4}}
                before, after = models["virtual_tryon"].generate_before_after(st.session_state.image, landmarks, makeup_settings)
                comparison = models["virtual_tryon"].create_comparison_grid(before, after, labels=True)
                st.image(comparison, caption="虚拟试妆效果", use_container_width=True)
                st.caption("提示：这是基于你面部特征的虚拟预览，实际效果可能有所不同")
            else:
                st.info("虚拟试妆功能暂时不可用")

            st.markdown("---")
            st.markdown("### 💇 发型建议")

            hair_recs = models["hairstyle"].recommend(face_shape=face_shape, forehead_height="normal", hair_texture="medium")
            st.write("**推荐长度**: " + " | ".join(hair_recs.get("recommended_lengths", [])[:2]))
            bangs = hair_recs.get("bangs_advice", {})
            st.write(f"**刘海建议**: {bangs.get('shape_advice', '')}")

            st.markdown("---")
            st.markdown("### 💡 光影建议")

            light_recs = models["lighting"].recommend(face_shape=face_shape, skin_tone=skin_tone)
            primary = light_recs.get("primary_lighting", {})
            st.write(f"**推荐布光**: {primary.get('name', '侧光45度')}")
            st.write(f"**效果**: {primary.get('effect', '')}")

            st.markdown("---")
            st.markdown("### 🎭 风格定位")

            style_profile = models["style"].profile(skin_tone=skin_tone, features=features, expression=expression)
            st.metric("色彩季型", style_profile.get("season_name", "春季型"))
            st.write("**特点**: " + "、".join(style_profile.get("characteristics", [])))
            palette = style_profile.get("color_palette", {})
            if palette:
                st.write("**主推色**: " + "、".join(palette.get("primary", [])[:3]))

        else:
            st.info("👈 请上传照片并点击「开始分析」")
            st.markdown("""
            ### 📋 使用流程

            1. **上传照片** - 清晰的正面照
            2. **AI分析** - 面部特征、肤色、比例
            3. **获取建议** - 妆容、发型，光影
            4. **虚拟试妆** - 预览改造效果
            5. **实践反馈** - 上传成果照获得优化建议
            """)


if __name__ == "__main__":
    main()