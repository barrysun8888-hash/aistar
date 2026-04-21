"""Makeup recommendation engine with dual-track system."""

from typing import Dict, List, Optional
import numpy as np


# Makeup knowledge base
EYE_RECOMMENDATIONS = {
    "almond": {
        "eyeliner": "标准眼线，沿睫毛根部描画，重点加强眼尾",
        "eyeshadow": "任意色系均可，适合渐层晕染",
        " mascara": "纤长型，突出本身眼型优势",
    },
    "round": {
        "eyeliner": "拉长眼尾，眼线向外延伸0.3cm，营造狭长感",
        "eyeshadow": "深色在外眼角晕染，视觉拉长眼型",
        " mascara": "自然卷翘，避免过于浓密",
    },
    "upturned": {
        "eyeliner": "眼尾向下描画，避免上扬感",
        "eyeshadow": "外眼角用深色，注意平衡",
        " mascara": "自然款，避免过度强调眼尾",
    },
    "downturned": {
        "eyeliner": "眼尾向上45度提拉，平衡下垂感",
        "eyeshadow": "强调眼头，外眼角略过眼窝",
        " mascara": "浓密款，加强眼尾存在感",
    },
}

LIP_RECOMMENDATIONS = {
    "thin": {
        "technique": "唇线向外延伸1-2mm，使用唇线笔勾勒新轮廓",
        "lipstick": "珠光或亮面质地，增加丰盈感",
        "gloss": "中央轻抹高光，营造立体感",
    },
    "medium": {
        "technique": "自然唇形即可，强调唇峰",
        "lipstick": "任意质地，根据场合选择",
        "gloss": "可选",
    },
    "full": {
        "technique": "可用遮瑕淡化唇线，收缩视觉",
        "lipstick": "哑光质地优先，避免膨胀感",
        "gloss": "避免全唇高光，仅限唇中央",
    },
}

FACE_SHAPE_CONTOUR = {
    "oval": {
        "highlight": "T区+颧骨高位，轻薄提亮",
        "shadow": "发际线边缘，颧骨下方",
        "technique": "自然立体感，避免过度雕琢",
    },
    "round": {
        "highlight": "额头中央、下巴，视觉拉长",
        "shadow": "脸颊两侧、下颌线，收缩圆润感",
        "technique": "强调面部中央，弱化边缘",
    },
    "square": {
        "highlight": "额头中央、下巴尖、颧骨高位",
        "shadow": "下颌角、额角，柔化棱角",
        "technique": "圆形高光+方形阴影，视觉柔化",
    },
    "heart": {
        "highlight": "下巴、下颌线，收缩上宽下窄",
        "shadow": "额角、上庭两侧",
        "technique": "平衡上下比例",
    },
    "diamond": {
        "highlight": "额头中央、下巴",
        "shadow": "颧骨高位，横向收缩",
        "technique": "柔化颧骨突出感",
    },
}

BROW_RECOMMENDATIONS = {
    "low": {"shape": "略高挑，提升眼距视觉"},
    "medium": {"shape": "自然弧度即可"},
    "high": {"shape": "略平缓，避免过度高挑"},
    "thick": {"shape": "纤细或中等厚度，避免厚重"},
    "thin": {"shape": "略粗一点，增加平衡感"},
}


class MakeupRecommender:
    """Generate personalized makeup recommendations using dual-track system."""

    def __init__(self):
        self._track_weights = {"optimize": 0.6, "breakthrough": 0.4}

    def recommend(
        self,
        face_shape: str,
        features: Dict,
        skin_tone: Dict,
        target_celebrity: Optional[str] = None,
    ) -> Dict:
        """Generate comprehensive makeup recommendations.

        Args:
            face_shape: One of oval/round/square/heart/diamond
            features: Dict with eyes, nose, lips, eyebrows features
            skin_tone: Dict with undertone, shade_level, color_recommendations
            target_celebrity: Optional target celebrity for breakthrough track

        Returns:
            Complete makeup recommendation with two tracks
        """
        base_recommendations = {
            "foundation": self._recommend_foundation(skin_tone),
            "contour": self._recommend_contour(face_shape),
            "eye_makeup": self._recommend_eyes(features.get("eyes", {})),
            "lip_makeup": self._recommend_lips(features.get("lips", {}), skin_tone),
            "eyebrows": self._recommend_eyebrows(features.get("eyebrows", {})),
        }

        # Generate optimize track (minimize change, maximize impact)
        optimize_track = self._generate_optimize_track(base_recommendations, face_shape, features)

        # Generate breakthrough track (style crossover)
        breakthrough_track = self._generate_breakthrough_track(
            base_recommendations, face_shape, features, target_celebrity
        )

        return {
            "base_recommendations": base_recommendations,
            "optimize_track": optimize_track,
            "breakthrough_track": breakthrough_track,
            "skin_tone_advice": self._get_skin_tone_advice(skin_tone),
        }

    def _recommend_foundation(self, skin_tone: Dict) -> Dict:
        """Recommend foundation based on skin tone."""
        undertone = skin_tone.get("undertone", "neutral")
        shade = skin_tone.get("shade_level", 3)
        recommendations = skin_tone.get("color_recommendations", {})

        foundation_tips = {
            "warm": "选择黄调或金色调的粉底，避免粉调",
            "cool": "选择粉调或蓝调的粉底，避免黄调",
            "neutral": "选择自然色调，可尝试两者混合",
        }

        return {
            "shade": f"Fitzpatrick Type {shade}",
            "undertone": undertone,
            "tip": foundation_tips.get(undertone, "选择自然色调"),
            "avoid": recommendations.get("foundation_avoid", []),
            "recommend": recommendations.get("foundation_recommend", []),
        }

    def _recommend_contour(self, face_shape: str) -> Dict:
        """Recommend contouring based on face shape."""
        contour_info = FACE_SHAPE_CONTOUR.get(face_shape, FACE_SHAPE_CONTOUR["oval"])

        return {
            "face_shape": face_shape,
            "highlight_areas": contour_info["highlight"],
            "shadow_areas": contour_info["shadow"],
            "technique": contour_info["technique"],
        }

    def _recommend_eyes(self, eye_features: Dict) -> Dict:
        """Recommend eye makeup based on eye shape."""
        eye_type = eye_features.get("type", "almond")
        eye_recs = EYE_RECOMMENDATIONS.get(eye_type, EYE_RECOMMENDATIONS["almond"])

        # Additional tips based on eye angle
        avg_angle = (eye_features.get("left_angle", 0) + eye_features.get("right_angle", 0)) / 2

        return {
            "eye_type": eye_type,
            "eyeliner": eye_recs["eyeliner"],
            "eyeshadow": eye_recs["eyeshadow"],
            "mascara": eye_recs["mascara"],
            "special_tip": "眼尾略上扬" if avg_angle > 3 else "眼尾自然或下垂" if avg_angle < -3 else "眼型标准",
        }

    def _recommend_lips(self, lip_features: Dict, skin_tone: Dict) -> Dict:
        """Recommend lip makeup based on lip shape and skin tone."""
        lip_fullness = lip_features.get("fullness", "medium")
        lip_recs = LIP_RECOMMENDATIONS.get(lip_fullness, LIP_RECOMMENDATIONS["medium"])

        undertone = skin_tone.get("undertone", "neutral")
        color_recs = skin_tone.get("color_recommendations", {}).get("lipstick", [])

        return {
            "lip_type": lip_fullness,
            "technique": lip_recs["technique"],
            "texture": lip_recs["lipstick"],
            "highlight": lip_recs["gloss"],
            "color_recommendations": color_recs,
        }

    def _recommend_eyebrows(self, brow_features: Dict) -> Dict:
        """Recommend eyebrow shape based on current brows."""
        arch = brow_features.get("arch_type", "medium")
        thickness = brow_features.get("thickness", "medium")

        brow_rec = {**BROW_RECOMMENDATIONS.get(arch, {}), **BROW_RECOMMENDATIONS.get(thickness, {})}

        return {
            "current_arch": arch,
            "current_thickness": thickness,
            "recommended_shape": brow_rec.get("shape", "自然弧度"),
            "technique": "眉头略低于眉峰，自然过渡到眉尾",
        }

    def _get_skin_tone_advice(self, skin_tone: Dict) -> str:
        """Get overall skin tone advice."""
        undertone = skin_tone.get("undertone", "neutral")
        shade = skin_tone.get("shade_level", 3)

        advice_map = {
            "warm": "适合金色、珊瑚色、绿色系妆容，避免过冷的粉紫色",
            "cool": "适合玫瑰粉、紫色、蓝色系妆容，避免过暖的橙黄色",
            "neutral": "大部分颜色都适合，可尝试冷暖混搭",
        }

        base = advice_map.get(undertone, "")
        shade_advice = f"肤色较{'深' if shade > 3 else '浅'}，底妆注意色号选择"

        return f"{base}。{shade_advice}"

    def _generate_optimize_track(
        self, base_recs: Dict, face_shape: str, features: Dict
    ) -> Dict:
        """Generate optimize track - amplify native strengths.

        Strategy: Find the feature that's already closest to ideal and enhance it.
        """
        # Score each feature
        feature_scores = {}

        # Eyes
        eye_type = features.get("eyes", {}).get("type", "almond")
        eye_score = 0.8 if eye_type == "almond" else 0.6
        feature_scores["eyes"] = eye_score

        # Lips
        lip_fullness = features.get("lips", {}).get("fullness", "medium")
        lip_score = 0.7 if lip_fullness == "medium" else 0.5
        feature_scores["lips"] = lip_score

        # Find strongest feature
        strongest = max(feature_scores, key=feature_scores.get)

        focus_areas = {
            "eyes": "强化你的眼妆优势，选择自然显色的眼影色系",
            "lips": "突出你的唇形特点，选择适合你肤色的口红",
        }

        return {
            "strategy": "最小改动，最大提升",
            "focus_area": strongest,
            "focus_advice": focus_areas.get(strongest, ""),
            "steps": [
                base_recs["foundation"],
                base_recs["eyebrows"],
                base_recs["eye_makeup"] if strongest == "eyes" else {"note": "简化眼妆"},
                base_recs["contour"],
                base_recs["lip_makeup"] if strongest == "lips" else {"note": "简化唇妆"},
            ],
            "expected_impact": "提升10-15%整体颜值分数",
        }

    def _generate_breakthrough_track(
        self,
        base_recs: Dict,
        face_shape: str,
        features: Dict,
        target_celebrity: Optional[str] = None,
    ) -> Dict:
        """Generate breakthrough track - style crossover.

        Strategy: Recommend changes that create contrast with current features.
        """
        # Determine crossover direction
        lip_fullness = features.get("lips", {}).get("fullness", "medium")

        crossover_tips = {
            "thin": "尝试饱满唇妆，扩展唇线，看起更有女人味",
            "medium": "尝试咬唇妆或渐变唇，增添清新感",
            "full": "尝试哑光质地+清晰唇线，打造高级感",
        }

        return {
            "strategy": "反差感突破 - 尝试不同风格",
            "crossover_tip": crossover_tips.get(lip_fullness, ""),
            "target_celebrity": target_celebrity,
            "steps": [
                base_recs["foundation"],
                base_recs["eyebrows"],
                base_recs["eye_makeup"],
                base_recs["contour"],
                base_recs["lip_makeup"],
            ],
            "expected_impact": "扩展风格可能性，发现新的自己",
        }

    def get_step_by_step教程(
        self,
        recommendations: Dict,
        track: str = "optimize",
    ) -> List[Dict]:
        """Generate step-by-step tutorial from recommendations.

        Args:
            recommendations: Output from recommend()
            track: "optimize" or "breakthrough"

        Returns:
            List of steps with instructions
        """
        track_data = recommendations.get(f"{track}_track", {})
        steps = track_data.get("steps", [])

        tutorial = []
        for i, step in enumerate(steps):
            tutorial.append({
                "step": i + 1,
                "title": list(step.keys())[0] if step else f"Step {i+1}",
                "instructions": step if isinstance(step, dict) else {"content": step},
            })

        return tutorial
