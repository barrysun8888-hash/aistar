"""Lighting recommendations for makeup and photo optimization."""

from typing import Dict, List


class LightingRecommender:
    """Recommend lighting setups for different purposes."""

    # Lighting presets for different effects
    LIGHTING_PRESETS = {
        "side_45": {
            "name": "侧光45度",
            "description": "经典人像布光，强化轮廓",
            "effect": "增加面部立体感，强化颧骨和下颌线",
            "similarity_boost": "+5-10%",
            "setup": {
                "light_position": "主体侧面45度",
                "light_height": "与眼睛平齐或略高",
                "intensity": "主光强度70-80%",
                "fill": "对侧用反光板补光30%",
            },
            "best_for": ["御姐风", "高级感", "增强气场"],
        },
        "butterfly": {
            "name": "蝴蝶光（派拉蒙光）",
            "description": "光源正前方偏上，在颧骨下方投射出蝴蝶状阴影",
            "effect": "美化面部，减少皱纹，突出颧骨",
            "similarity_boost": "+3-5%",
            "setup": {
                "light_position": "正前方偏上",
                "light_height": "高出主体30-45度",
                "intensity": "100%",
                "fill": "下方反光板补光50%",
            },
            "best_for": ["甜美风", "减少面部阴影", "柔化面部"],
        },
        "loop": {
            "name": "环状光",
            "description": "侧上方光源，在面部一侧形成环形阴影",
            "effect": "自然立体感，兼具柔美和轮廓",
            "similarity_boost": "+3-5%",
            "setup": {
                "light_position": "侧面30-45度，偏前",
                "light_height": "高出30度",
                "intensity": "主光70%",
                "fill": "对侧眼神光补光",
            },
            "best_for": ["日常妆", "自然光感", "通用场景"],
        },
        "rembrandt": {
            "name": "伦勃朗光",
            "description": "三角形光斑出现在阴影面颧骨上",
            "effect": "戏剧化效果，增强个性表达",
            "similarity_boost": "+5-8%",
            "setup": {
                "light_position": "侧面90度或更后",
                "light_height": "高出45度",
                "intensity": "主光100%",
                "fill": "对侧几乎无补光",
            },
            "best_for": ["艺术照", "增强男性特质", "戏剧感"],
        },
        "soft_front": {
            "name": "正面柔光",
            "description": "大面积柔光箱正前方",
            "effect": "柔化皮肤，减少瑕疵，高曝光",
            "similarity_boost": "+3-5%",
            "setup": {
                "light_position": "正前方",
                "light_height": "平视高度",
                "intensity": "100%",
                "diffusion": "使用柔光箱或反光伞",
            },
            "best_for": ["直播", "美妆展示", "柔光皮肤"],
        },
    }

    # Color temperature recommendations
    COLOR_TEMPERATURE = {
        "warm": {
            "name": "暖光 (3200K-4500K)",
            "effect": "营造温暖亲切感，提亮肤色",
            "recommend_for": ["甜美风格", "暖色调妆容", "亲和氛围"],
            "avoid_for": ["冷白皮", "强调锐利感"],
        },
        "neutral": {
            "name": "中性光 (4500K-5500K)",
            "effect": "忠实还原肤色，最通用",
            "recommend_for": ["日常妆容", "通用场景", "产品展示"],
            "avoid_for": [],
        },
        "cool": {
            "name": "冷光 (5500K-6500K)",
            "effect": "清爽专业感，减少泛红",
            "recommend_for": ["职场风格", "强调轮廓", "白皙肤色"],
            "avoid_for": ["暖黄皮", "强调健康感"],
        },
    }

    def recommend(
        self,
        face_shape: str,
        skin_tone: Dict,
        purpose: str = "daily",
    ) -> Dict:
        """Generate lighting recommendations.

        Args:
            face_shape: One of oval/round/square/heart/diamond
            skin_tone: Dict with undertone info
            purpose: "daily", "photo", "video", "stage"

        Returns:
            Lighting setup recommendations
        """
        undertone = skin_tone.get("undertone", "neutral")

        # Select preset based on face shape
        preset_key = self._get_preset_for_face_shape(face_shape)
        preset = self.LIGHTING_PRESETS[preset_key]

        # Select color temperature based on undertone
        if undertone == "warm":
            color_temp = self.COLOR_TEMPERATURE["warm"]
        elif undertone == "cool":
            color_temp = self.COLOR_TEMPERATURE["cool"]
        else:
            color_temp = self.COLOR_TEMPERATURE["neutral"]

        return {
            "primary_lighting": preset,
            "color_temperature": color_temp,
            "practical_tips": self._get_practical_tips(preset_key, undertone),
            "adjustments_for_face_shape": self._get_adjustments(face_shape),
        }

    def _get_preset_for_face_shape(self, face_shape: str) -> str:
        """Get best lighting preset for face shape."""
        preset_map = {
            "round": "side_45",
            "square": "loop",
            "heart": "butterfly",
            "oval": "soft_front",
            "diamond": "rembrandt",
        }
        return preset_map.get(face_shape, "side_45")

    def _get_adjustments(self, face_shape: str) -> Dict[str, str]:
        """Get lighting adjustments for specific face shapes."""
        adjustments = {
            "round": "使用侧光强调轮廓，减少正面曝光",
            "square": "使用柔和侧光柔化棱角，避免硬光",
            "heart": "下方补光平衡上宽下窄",
            "oval": "几乎所有布光都适合，正面光最佳",
            "diamond": "使用蝴蝶光或环状光补充颧骨宽度",
        }
        return {"adjustment": adjustments.get(face_shape, "")}

    def _get_practical_tips(self, preset_key: str, undertone: str) -> List[str]:
        """Get practical tips for achieving the look without professional equipment."""
        tips = {
            "side_45": [
                "窗边自然光是最好的侧光替代",
                "用白色床单或纸板作为反光板",
                "避免顶光，会在眼窝产生阴影",
            ],
            "butterfly": [
                "正对窗户或光源",
                "下方放白色反光板补光",
                "用窗帘软化强光",
            ],
            "loop": [
                "窗户在侧面30-45度",
                "对侧用白色板补眼神光",
                "注意鼻子阴影不碰到嘴唇",
            ],
            "rembrandt": [
                "光源在侧面90度",
                "几乎不用补光",
                "阴影面积较大是正常的",
            ],
            "soft_front": [
                "大面积柔光（如窗帘）",
                "正对光源",
                "可用白色床单反射补光",
            ],
        }

        base_tips = tips.get(preset_key, [])

        if undertone == "warm":
            base_tips.append("暖色调妆容在暖光下更显气色")
        elif undertone == "cool":
            base_tips.append("冷白灯光下避免过蓝，适当调整白平衡")

        return base_tips

    def get_phone_camera_tips(self) -> Dict:
        """Get tips for optimal phone camera lighting."""
        return {
            "natural_light": {
                "best_time": "日出后1小时或日落前2小时",
                "avoid": "正午阳光（硬阴影）",
                "position": "面向窗户或光源",
            },
            "indoor": {
                "avoid": "顶部直射光（如吸顶灯）",
                "prefer": "面前45度柔光",
                "hack": "用白色手机补光灯或手电筒补光",
            },
            "night": {
                "prefer": "环形补光灯",
                "position": "正前方或略高",
                "avoid": "背后有强光源（过曝）",
            },
            "common_mistakes": [
                "面部过暗：增加正面补光",
                "油光反射：使用哑光产品或调整角度",
                "阴阳脸：注意左右脸光线均匀",
            ],
        }
