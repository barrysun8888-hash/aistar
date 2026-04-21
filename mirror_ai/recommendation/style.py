"""Style profiling based on color season theory and personal characteristics."""

from typing import Dict, List, Tuple


# Color season definitions
COLOR_SEASONS = {
    "spring": {
        "name": "春季型",
        "characteristics": ["温暖", "明亮", "活泼", "轻盈"],
        "palette": {
            "primary": ["珊瑚色", "暖粉色", "桃色", "杏色"],
            "secondary": ["草绿色", "浅蓝色", "暖黄色", "米色"],
            "accent": ["金色", "橙红色", "珊瑚红"],
            "avoid": ["深紫色", "黑色", "深蓝色", "灰色"],
        },
        "best_for": ["日常妆容", "职场妆容", "派对妆容"],
    },
    "summer": {
        "name": "夏季型",
        "characteristics": ["凉爽", "柔和", "温柔", "优雅"],
        "palette": {
            "primary": ["浅粉色", "薰衣草紫", "玫瑰灰", "雾蓝色"],
            "secondary": ["薄荷绿", "浅灰色", "淡黄色", "玫瑰粉"],
            "accent": ["银白色", "玫瑰金", "淡紫色"],
            "avoid": ["橙色", "金色", "过深的颜色"],
        },
        "best_for": ["清新妆容", "约会妆容", "日常通勤"],
    },
    "autumn": {
        "name": "秋季型",
        "characteristics": ["温暖", "浓郁", "成熟", "复古"],
        "palette": {
            "primary": ["砖红色", "焦糖色", "棕红色", "橙棕色"],
            "secondary": ["橄榄绿", "土黄色", "芥末黄", "咖啡色"],
            "accent": ["金色", "古铜色", "深绿色"],
            "avoid": ["粉色", "浅蓝色", "薰衣草紫"],
        },
        "best_for": ["大地色妆容", "复古风格", "秋冬妆容"],
    },
    "winter": {
        "name": "冬季型",
        "characteristics": ["凉爽", "鲜艳", "强烈", "高对比"],
        "palette": {
            "primary": ["正红色", "莓果色", "紫红色", "玫红色"],
            "secondary": ["深蓝色", "墨绿色", "黑色", "纯白色"],
            "accent": ["银白色", "宝蓝色", "对比色"],
            "avoid": ["暖黄色", "橙色", "驼色", "金色（过于暖）"],
        },
        "best_for": ["晚宴妆容", "派对妆容", "高对比妆容"],
    },
}


class StyleProfiler:
    """Profile user's style characteristics and recommend color palettes."""

    def profile(
        self,
        skin_tone: Dict,
        features: Dict,
        expression: Dict,
    ) -> Dict:
        """Generate complete style profile.

        Args:
            skin_tone: Dict with undertone, shade_level, etc.
            features: Dict with eye, lip, brow features
            expression: Dict with smile_curve, expression info

        Returns:
            Complete style profile with color season and recommendations
        """
        # Determine color season
        color_season = self._determine_color_season(skin_tone, features, expression)

        # Get season details
        season_info = COLOR_SEASONS.get(color_season, COLOR_SEASONS["spring"])

        # Recommend specific looks
        looks = self._recommend_looks(color_season, features)

        return {
            "color_season": color_season,
            "season_name": season_info["name"],
            "characteristics": season_info["characteristics"],
            "color_palette": season_info["palette"],
            "looks": looks,
            "accessory_suggestions": self._get_accessory_suggestions(color_season),
        }

    def _determine_color_season(
        self,
        skin_tone: Dict,
        features: Dict,
        expression: Dict,
    ) -> str:
        """Determine user's color season.

        Algorithm:
        - Cool undertone + high contrast features → Winter
        - Cool undertone + soft features → Summer
        - Warm undertone + warm features → Autumn
        - Warm undertone + bright features → Spring
        """
        undertone = skin_tone.get("undertone", "neutral")
        shade_level = skin_tone.get("shade_level", 3)

        # Get feature contrasts
        lip_fullness = features.get("lips", {}).get("fullness", "medium")
        eye_type = features.get("eyes", {}).get("type", "almond")

        # Calculate contrast level
        contrast = self._calculate_contrast(features, expression)

        # Decision tree
        if undertone == "cool":
            if contrast > 0.6 or shade_level <= 2:
                return "winter"
            else:
                return "summer"
        elif undertone == "warm":
            if contrast > 0.5 or shade_level >= 4:
                return "autumn"
            else:
                return "spring"
        else:  # neutral
            if contrast > 0.55:
                return "winter" if shade_level <= 3 else "autumn"
            else:
                return "spring" if shade_level <= 3 else "summer"

    def _calculate_contrast(self, features: Dict, expression: Dict) -> float:
        """Calculate facial feature contrast level (0-1)."""
        scores = []

        # Eye contrast
        eye_type = features.get("eyes", {}).get("type", "almond")
        if eye_type in ["round", "upturned"]:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # Lip contrast
        lip_fullness = features.get("lips", {}).get("fullness", "medium")
        fullness_scores = {"thin": 0.3, "medium": 0.5, "full": 0.8}
        scores.append(fullness_scores.get(lip_fullness, 0.5))

        # Expression contrast
        expression_intensity = expression.get("expression_intensity", 0.5)
        scores.append(expression_intensity)

        return sum(scores) / len(scores)

    def _recommend_looks(self, color_season: str, features: Dict) -> List[Dict]:
        """Recommend specific looks for color season."""
        looks_by_season = {
            "spring": [
                {
                    "name": "清新春日",
                    "description": "轻薄底妆+粉色系眼影+珊瑚色唇妆",
                    "intensity": "低",
                    "mood": "活泼、少女感",
                },
                {
                    "name": "职场丽人",
                    "description": "自然眉形+大地色眼影+豆沙色唇",
                    "intensity": "中",
                    "mood": "干练、亲和",
                },
            ],
            "summer": [
                {
                    "name": "薰衣草之梦",
                    "description": "淡紫色眼影+玫瑰灰腮红+MLBB唇色",
                    "intensity": "低",
                    "mood": "温柔、梦幻",
                },
                {
                    "name": "法式优雅",
                    "description": "自然眼妆+哑光玫瑰色+简约配饰",
                    "intensity": "中",
                    "mood": "优雅、随性",
                },
            ],
            "autumn": [
                {
                    "name": "焦糖拿铁",
                    "description": "暖棕色眼影+砖红色唇妆+古铜色修容",
                    "intensity": "中",
                    "mood": "成熟、温暖",
                },
                {
                    "name": "复古美人",
                    "description": "精致眼线+正红色唇+蓬松发型",
                    "intensity": "高",
                    "mood": "复古、惊艳",
                },
            ],
            "winter": [
                {
                    "name": "冰雪女王",
                    "description": "烟熏眼妆+正红色唇+高对比修容",
                    "intensity": "高",
                    "mood": "冷艳、强大气场",
                },
                {
                    "name": "派对焦点",
                    "description": "亮片眼妆+莓果色唇+精致高光",
                    "intensity": "高",
                    "mood": "闪耀、吸睛",
                },
            ],
        }

        return looks_by_season.get(color_season, looks_by_season["spring"])

    def _get_accessory_suggestions(self, color_season: str) -> Dict[str, List[str]]:
        """Get accessory suggestions for color season."""
        suggestions = {
            "spring": {
                "jewelry": ["金色圆形耳环", "珍珠项链", "暖色调宝石"],
                "scarves": ["暖色印花", "轻盈材质", "粉色/橙色系"],
                "frames": ["金色眼镜框", "暖棕色发夹"],
            },
            "summer": {
                "jewelry": ["银色细链", "珍珠", "玫瑰金"],
                "scarves": ["浅色轻薄", "薰衣草紫", "浅蓝色"],
                "frames": ["银色眼镜框", "纤细发箍"],
            },
            "autumn": {
                "jewelry": ["金色/古铜色调", "琥珀", "绿松石"],
                "scarves": ["格纹", "驼色", "橄榄绿"],
                "frames": ["玳瑁色", "深棕色"],
            },
            "winter": {
                "jewelry": ["银白色", "钻石", "红宝石", "蓝宝石"],
                "scarves": ["高对比色", "深红", "纯黑", "纯白"],
                "frames": ["黑色", "银色", "几何形状"],
            },
        }
        return suggestions.get(color_season, suggestions["spring"])

    def get_collar_suggestions(self, face_shape: str) -> Dict[str, List[str]]:
        """Get clothing collar suggestions for face shape."""
        collars = {
            "oval": {
                "best": ["任意领型", "V领", "圆领", "方领"],
                "avoid": [],
                "recommendation": "标准脸型，几乎所有领型都适合",
            },
            "round": {
                "best": ["V领", "U领", "长项链"],
                "avoid": ["高领", "圆领", "小领口"],
                "recommendation": "V领视觉拉长颈部线条",
            },
            "square": {
                "best": ["圆领", "船领", "飘带领"],
                "avoid": ["方领", "尖领"],
                "recommendation": "柔和领型平衡面部棱角",
            },
            "heart": {
                "best": ["方领", "圆领", "一字肩"],
                "avoid": ["高领", "小圆领"],
                "recommendation": "平衡上宽下窄，可露锁骨",
            },
            "diamond": {
                "best": ["圆领", "船领", "高领"],
                "avoid": ["V领过深"],
                "recommendation": "高领或蓬松领口补充颧骨宽度",
            },
        }
        return collars.get(face_shape, collars["oval"])
