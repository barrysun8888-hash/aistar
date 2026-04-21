"""User profile management for personalized experience."""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime


class UserProfile:
    """Manage user profiles for personalized recommendations."""

    def __init__(self, user_id: str, storage_path: str = "user_profiles"):
        """Initialize user profile.

        Args:
            user_id: Unique user identifier
            storage_path: Directory to store profile data
        """
        self._user_id = user_id
        self._storage_path = storage_path
        self._profile_path = os.path.join(storage_path, f"{user_id}.json")
        self._profile: Dict = self._load_profile()

    def _load_profile(self) -> Dict:
        """Load profile from disk or create new."""
        if os.path.exists(self._profile_path):
            try:
                with open(self._profile_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        return self._create_default_profile()

    def _create_default_profile(self) -> Dict:
        """Create default profile structure."""
        return {
            "user_id": self._user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "face_shape": None,
            "skin_tone": None,
            "color_season": None,
            "successful_looks": [],
            "feedback_history": [],
            "preferences": {
                "track": "optimize",
                "intensity": 0.7,
            },
        }

    def update(self, updates: Dict) -> None:
        """Update profile with new data.

        Args:
            updates: Dict of fields to update
        """
        self._profile.update(updates)
        self._profile["last_updated"] = datetime.now().isoformat()
        self._save_profile()

    def _save_profile(self) -> None:
        """Save profile to disk."""
        os.makedirs(self._storage_path, exist_ok=True)
        with open(self._profile_path, "w") as f:
            json.dump(self._profile, f, indent=2)

    def record_look(
        self,
        look_type: str,
        recommendations: Dict,
        result_score: float,
    ) -> None:
        """Record a successful look for future reference.

        Args:
            look_type: Type of look (e.g., "optimize", "breakthrough")
            recommendations: The recommendations given
            result_score: Evaluation score
        """
        look_record = {
            "timestamp": datetime.now().isoformat(),
            "type": look_type,
            "recommendations": recommendations,
            "result_score": result_score,
        }

        self._profile["successful_looks"].append(look_record)

        # Keep only last 20 looks
        if len(self._profile["successful_looks"]) > 20:
            self._profile["successful_looks"] = self._profile["successful_looks"][-20:]

        self._save_profile()

    def get_successful_look(self, look_type: Optional[str] = None) -> Optional[Dict]:
        """Get most successful look of a given type.

        Args:
            look_type: Filter by look type, or None for any

        Returns:
            Most successful look dict or None
        """
        looks = self._profile["successful_looks"]

        if look_type:
            looks = [l for l in looks if l.get("type") == look_type]

        if not looks:
            return None

        return max(looks, key=lambda l: l.get("result_score", 0))

    def get_style_dna(self) -> Dict:
        """Get user's learned style DNA.

        Returns:
            Dict with learned patterns
        """
        looks = self._profile["successful_looks"]
        if not looks:
            return {"status": "no_data"}

        recent_looks = looks[-10:]  # Last 10 looks

        return {
            "total_looks": len(looks),
            "preferred_track": self._most_common([l.get("type") for l in recent_looks]),
            "average_score": sum(l.get("result_score", 0) for l in recent_looks) / len(recent_looks),
            "consistent_improvement": self._check_consistent_improvement(looks),
        }

    def _most_common(self, lst: List) -> str:
        """Get most common item in list."""
        if not lst:
            return "unknown"
        return max(set(lst), key=lst.count)

    def _check_consistent_improvement(self, looks: List) -> bool:
        """Check if scores are improving over time."""
        if len(looks) < 3:
            return False

        recent_scores = [l.get("result_score", 0) for l in looks[-5:]]
        first_half_avg = sum(recent_scores[:2]) / 2
        second_half_avg = sum(recent_scores[-2:]) / 2

        return second_half_avg > first_half_avg

    def to_dict(self) -> Dict:
        """Export profile as dict."""
        return self._profile.copy()
