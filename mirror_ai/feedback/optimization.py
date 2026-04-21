"""Dynamic optimization - adjust recommendations based on feedback."""

import numpy as np
from typing import Dict, List, Optional
import json
import os


class DynamicOptimizer:
    """Dynamically optimize recommendations based on user feedback and results."""

    def __init__(self, user_profile_path: Optional[str] = None):
        """Initialize optimizer.

        Args:
            user_profile_path: Path to store user profile data
        """
        self._user_profile_path = user_profile_path or "user_profile.json"
        self._user_profile: Dict = self._load_profile()

    def _load_profile(self) -> Dict:
        """Load user profile from disk."""
        if os.path.exists(self._user_profile_path):
            try:
                with open(self._user_profile_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return self._create_default_profile()

    def _create_default_profile(self) -> Dict:
        """Create default user profile structure."""
        return {
            "successful_styles": [],
            "failed_styles": [],
            "style_weights": {
                "optimize": 0.6,
                "breakthrough": 0.4,
            },
            "feedback_history": [],
        }

    def record_result(
        self,
        user_id: str,
        recommendations: Dict,
        execution_result: Dict,
        evaluation_result: Dict,
        user_feedback: Optional[str] = None,
    ) -> None:
        """Record makeover result for learning.

        Args:
            user_id: User identifier
            recommendations: Original recommendations given
            execution_result: Execution detection results
            evaluation_result: Effect evaluation results
            user_feedback: Optional user text feedback
        """
        result = {
            "recommendations": recommendations,
            "execution_score": execution_result.get("overall_execution_score", 0),
            "effect_score": evaluation_result.get("overall_score", 0),
            "user_feedback": user_feedback,
        }

        # Update successful/failed styles
        if evaluation_result.get("overall_score", 0) >= 70:
            self._user_profile["successful_styles"].append({
                "type": recommendations.get("track", "unknown"),
                "face_shape": recommendations.get("face_shape", "unknown"),
            })
        else:
            self._user_profile["failed_styles"].append({
                "type": recommendations.get("track", "unknown"),
                "face_shape": recommendations.get("face_shape", "unknown"),
            })

        # Limit history size
        if len(self._user_profile["successful_styles"]) > 50:
            self._user_profile["successful_styles"] = self._user_profile["successful_styles"][-50:]
        if len(self._user_profile["failed_styles"]) > 50:
            self._user_profile["failed_styles"] = self._user_profile["failed_styles"][-50:]

        self._user_profile["feedback_history"].append(result)

        # Save profile
        self._save_profile()

    def _save_profile(self) -> None:
        """Save user profile to disk."""
        try:
            with open(self._user_profile_path, "w") as f:
                json.dump(self._user_profile, f, indent=2)
        except IOError:
            pass

    def optimize_recommendations(
        self,
        base_recommendations: Dict,
        user_id: str,
    ) -> Dict:
        """Optimize recommendations based on user's history.

        Args:
            base_recommendations: Base recommendations to optimize
            user_id: User identifier

        Returns:
            Optimized recommendations
        """
        optimized = base_recommendations.copy()

        # Adjust weights based on history
        successful = self._user_profile["successful_styles"]
        failed = self._user_profile["failed_styles"]

        if successful and not failed:
            # User responds well to recommendations, keep similar approach
            pass
        elif failed and not successful:
            # User hasn't had success, suggest trying different track
            optimized["suggested_track"] = "breakthrough" if base_recommendations.get("track") == "optimize" else "optimize"
        elif failed and successful:
            # Analyze pattern
            failed_types = set(s["type"] for s in failed)
            successful_types = set(s["type"] for s in successful)

            if "optimize" in failed_types and "breakthrough" in successful_types:
                optimized["suggested_track"] = "breakthrough"
            elif "breakthrough" in failed_types and "optimize" in successful_types:
                optimized["suggested_track"] = "optimize"

        # Add personalized tips based on history
        personalized_tips = self._generate_personalized_tips()
        if personalized_tips:
            optimized["personalized_tips"] = personalized_tips

        return optimized

    def _generate_personalized_tips(self) -> List[str]:
        """Generate tips based on user's history."""
        tips = []

        successful = self._user_profile["successful_styles"]
        if len(successful) >= 3:
            # Find common successful patterns
            types = [s["type"] for s in successful]
            if types.count("optimize") > types.count("breakthrough"):
                tips.append("你之前的优化轨建议效果较好，建议继续沿用此方向")
            elif types.count("breakthrough") > types.count("optimize"):
                tips.append("你之前的突破轨建议效果较好，建议继续探索不同风格")

        return tips

    def get_user_dna(self) -> Dict:
        """Get user's personal 'style DNA'.

        Returns learned patterns about what works for this user.
        """
        return {
            "successful_patterns": self._user_profile["successful_styles"],
            "failed_patterns": self._user_profile["failed_styles"],
            "preference": self._get_preference(),
            "style_learning": self._get_style_learning(),
        }

    def _get_preference(self) -> str:
        """Infer user's style preference from history."""
        successful = self._user_profile["successful_styles"]
        if not successful:
            return "unknown"

        types = [s["type"] for s in successful[-10:]]  # Recent history
        optimize_count = types.count("optimize")
        breakthrough_count = types.count("breakthrough")

        if optimize_count > breakthrough_count * 1.5:
            return "倾向自然优化风格"
        elif breakthrough_count > optimize_count * 1.5:
            return "倾向突破尝试新风格"
        else:
            return "均衡偏好"

    def _get_style_learning(self) -> Dict:
        """Analyze what the system has learned about this user."""
        successful = self._user_profile["successful_styles"]
        if not successful:
            return {"status": "insufficient_data"}

        # Analyze successful combinations
        face_shapes = [s["face_shape"] for s in successful]
        style_types = [s["type"] for s in successful]

        return {
            "total_successful_looks": len(successful),
            "preferred_track": max(set(style_types), key=style_types.count) if style_types else "unknown",
            "most_common_face_shape": max(set(face_shapes), key=face_shapes.count) if face_shapes else "unknown",
            "confidence": min(1, len(successful) / 10),  # More data = higher confidence
        }
