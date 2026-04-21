"""Hairstyle recommendation based on face shape and features."""

from typing import Dict, List


# Hair length recommendations by face shape
FACE_SHAPE_LENGTH = {
    "oval": ["任意长度", "长短发皆宜", "根据气质选择"],
    "round": ["及肩或更长", "层次感短发", "视觉拉长脸型"],
    "square": ["中长发或长发", "柔和层次的短发", "卷发柔化棱角"],
    "heart": ["下巴长度或更长", "蓬松感短发", "平衡上下比例"],
    "diamond": ["及肩或更长", "有体积感的短发", "补充颧骨宽度"],
}

# Hair texture recommendations
TEXTURE_RECOMMENDATIONS = {
    "fine": {
        "volume": "需要增加体积感",
        "recommendations": [
            "层次感剪裁增加蓬松度",
            "使用蓬松喷雾或摩丝",
            "避免贴头皮的直发",
            "烫发可增加质感",
        ],
        "avoid": ["过于服帖的发型", "过长的直发"],
    },
    "medium": {
        "volume": "标准体积即可",
        "recommendations": [
            "大部分发型都适合",
            "可根据需求调整",
        ],
        "avoid": [],
    },
    "thick": {
        "volume": "需要控制体积",
        "recommendations": [
            "打薄处理减少体积",
            "直发或内扣都适合",
            "层次剪裁减轻厚重感",
        ],
        "avoid": ["过于蓬松的发型", "小卷"],
    },
}

# Bangs recommendations by face shape and forehead
BANGS_RECOMMENDATIONS = {
    "round": {
        "shape_advice": "避免齐刘海，选择侧分或空气刘海",
        "forehead_high": "可尝试刘海降低发际线",
        "forehead_low": "中分或侧分，露额头显脸长",
    },
    "square": {
        "shape_advice": "柔和刘海平衡棱角",
        "forehead_high": "空气刘海或法式刘海",
        "forehead_low": "侧分长刘海",
    },
    "heart": {
        "shape_advice": "下巴宽度刘海平衡上宽下窄",
        "forehead_high": "刘海是很好的选择",
        "forehead_low": "侧分或中分",
    },
    "oval": {
        "shape_advice": "几乎所有刘海都适合",
        "forehead_high": "刘海很好看",
        "forehead_low": "可无刘海或短刘海",
    },
    "diamond": {
        "shape_advice": "空气刘海补充额头宽度",
        "forehead_high": "刘海适合",
        "forehead_low": "侧分长刘海",
    },
}


class HairstyleRecommender:
    """Recommend hairstyles based on face shape and hair characteristics."""

    def recommend(
        self,
        face_shape: str,
        forehead_height: str = "normal",
        hair_texture: str = "medium",
        forehead_width: str = "normal",
    ) -> Dict:
        """Generate hairstyle recommendations.

        Args:
            face_shape: One of oval/round/square/heart/diamond
            forehead_height: "high", "normal", or "low"
            hair_texture: "fine", "medium", or "thick"
            forehead_width: "narrow", "normal", or "wide"

        Returns:
            Comprehensive hairstyle recommendations
        """
        return {
            "recommended_lengths": self._get_length_recommendations(face_shape),
            "bangs_advice": self._get_bangs_advice(face_shape, forehead_height, forehead_width),
            "texture_advice": self._get_texture_advice(hair_texture),
            "color_suggestions": self._get_color_suggestions(face_shape),
            "celebrity_references": self._get_celebrity_references(face_shape),
        }

    def _get_length_recommendations(self, face_shape: str) -> List[str]:
        """Get recommended hair lengths for face shape."""
        return FACE_SHAPE_LENGTH.get(face_shape, FACE_SHAPE_LENGTH["oval"])

    def _get_bangs_advice(
        self, face_shape: str, forehead_height: str, forehead_width: str
    ) -> Dict:
        """Get bangs advice based on face shape and forehead."""
        base_advice = BANGS_RECOMMENDATIONS.get(face_shape, BANGS_RECOMMENDATIONS["oval"])

        # Determine forehead type
        if forehead_height == "high":
            forehead_key = "forehead_high"
        elif forehead_height == "low":
            forehead_key = "forehead_low"
        else:
            forehead_key = None

        advice = {"shape_advice": base_advice["shape_advice"]}

        if forehead_key and forehead_key in base_advice:
            advice["forehead_specific"] = base_advice[forehead_key]

        if forehead_width == "wide":
            advice["width_advice"] = "避免过短刘海，选择长刘海或侧分"
        elif forehead_width == "narrow":
            advice["width_advice"] = "可尝试短刘海增加额头露出"

        return advice

    def _get_texture_advice(self, hair_texture: str) -> Dict:
        """Get advice based on hair texture."""
        return TEXTURE_RECOMMENDATIONS.get(hair_texture, TEXTURE_RECOMMENDATIONS["medium"])

    def _get_color_suggestions(self, face_shape: str) -> List[str]:
        """Get hair color suggestions."""
        # Generic suggestions - could be enhanced with skin tone integration
        return [
            "自然色系（黑/棕）最安全",
            "可根据肤色选择冷暖色调",
            "挑染可增加层次感",
            "避免与肤色过于接近的颜色",
        ]

    def _get_celebrity_references(self, face_shape: str) -> List[Dict]:
        """Get celebrity hair references for face shape."""
        references = {
            "oval": [
                {"name": "刘亦菲", "hairstyle": "长发波浪", "why": "标准脸型，任何发型都适合"},
                {"name": "刘雯", "hairstyle": "短发层次", "why": "可盐可甜"},
            ],
            "round": [
                {"name": "赵丽颖", "hairstyle": "侧分长发", "why": "视觉拉长脸型"},
                {"name": "石原里美", "hairstyle": "卷发+刘海", "why": "增加柔和感"},
            ],
            "square": [
                {"name": "倪妮", "hairstyle": "长发大卷", "why": "柔化面部棱角"},
                {"name": "钟楚曦", "hairstyle": "波浪卷发", "why": "高级感满满"},
            ],
            "heart": [
                {"name": "宋慧乔", "hairstyle": "锁骨发", "why": "平衡上下比例"},
                {"name": "杨紫", "hairstyle": "空气刘海+长发", "why": "减龄又甜美"},
            ],
            "diamond": [
                {"name": "周冬雨", "hairstyle": "短发", "why": "突出灵动气质"},
                {"name": "张天爱", "hairstyle": "长卷发", "why": "补充颧骨宽度"},
            ],
        }
        return references.get(face_shape, references["oval"])

    def recommend_by_celebrity_inspiration(
        self, target_celebrity: str, face_shape: str
    ) -> Dict:
        """Recommend modifications when inspired by a celebrity's hairstyle.

        Args:
            target_celebrity: Celebrity being referenced
            face_shape: User's face shape

        Returns:
            Advice on how to adapt celebrity's hairstyle to suit your face shape
        """
        adaptations = {
            "adjust_length": f"根据你的{face_shape}型脸，调整长度到适合位置",
            "add_layers": "增加层次感让发型更符合你的脸型",
            "bangs_adjustment": "可适当加入刘海平衡面部比例",
            "volume_adjustment": "根据发质调整体积感",
        }

        return {
            "celebrity": target_celebrity,
            "adaptations": adaptations,
            "warning": "明星造型可能需要针对个人特点调整，不要完全复制",
        }
