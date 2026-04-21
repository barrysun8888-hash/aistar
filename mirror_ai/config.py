"""Configuration settings for Mirror AI."""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class MirrorAIConfig:
    """Main configuration for Mirror AI application."""

    # Face Analysis
    mediapipe_model_path: str = "face_landmarker.task"
    landmark_count: int = 468

    # Celebrity Similarity
    celebrity_db_path: str = "data/celebrity_db"
    embedding_dim: int = 512
    max_similar_celebrities: int = 5

    # Face Shape Classification
    face_shapes: List[str] = None

    # Style Space
    style_space_dimensions: int = 4

    # User Profile
    user_profile_path: str = "data/user_profiles"

    # Preview
    default_makeup_intensity: float = 0.7

    # Feedback
    execution_threshold: float = 70.0
    improvement_threshold: float = 60.0

    def __post_init__(self):
        if self.face_shapes is None:
            self.face_shapes = ["oval", "round", "square", "heart", "diamond"]

    @classmethod
    def from_env(cls) -> "MirrorAIConfig":
        """Create config from environment variables."""
        return cls(
            mediapipe_model_path=os.getenv("MEDIAPIPE_MODEL", cls.mediapipe_model_path),
            celebrity_db_path=os.getenv("CELEB_DB_PATH", cls.celebrity_db_path),
            user_profile_path=os.getenv("USER_PROFILE_PATH", cls.user_profile_path),
        )


# Default config instance
config = MirrorAIConfig()
