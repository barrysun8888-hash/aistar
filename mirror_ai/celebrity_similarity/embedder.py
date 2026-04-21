"""Celebrity embedding using InsightFace ArcFace."""

import numpy as np
from typing import Optional, List
import cv2
import os

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class CelebrityEmbedder:
    """Generate face embeddings using ArcFace (via InsightFace)."""

    def __init__(self, providers: Optional[List] = None):
        """Initialize ArcFace embedder.

        Args:
            providers: ONNX providers (e.g., ['CPUExecutionProvider'])
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "insightface not installed. Install with: pip install insightface"
            )

        self._app = FaceAnalysis(name="buffalo_l", providers=providers or ['CPUExecutionProvider'])
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    def embed(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Generate 512-dim embedding for a face.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            512-dim embedding vector, or None if no face detected
        """
        faces = self._app.get(image)
        if not faces:
            return None

        # Use the largest face
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        return face.normed_embedding

    def embed_batch(self, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple images.

        Args:
            images: List of BGR images

        Returns:
            List of 512-dim embedding vectors (None if no face detected)
        """
        return [self.embed(img) for img in images]

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: 512-dim vector
            embedding2: 512-dim vector

        Returns:
            Similarity score in range [-1, 1]
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        )

    def compute_distance(self, embedding1: np.ndarray, embedding