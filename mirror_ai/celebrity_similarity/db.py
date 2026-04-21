"""Celebrity database using FAISS for vector similarity search."""

import numpy as np
from typing import List, Dict, Optional, Tuple
import faiss
import pickle
import os


class CelebrityDatabase:
    """FAISS-backed celebrity face database for similarity search."""

    def __init__(self, embedding_dim: int = 512):
        """Initialize empty celebrity database.

        Args:
            embedding_dim: Dimension of face embeddings (512 for ArcFace)
        """
        self._embedding_dim = embedding_dim
        self._index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine sim for normalized)
        self._celebrities: List[Dict] = []
        self._embedding_to_id: Dict[int, int] = {}  # FAISS idx -> celebrity id

    def add_celebrity(
        self,
        celebrity_id: str,
        name: str,
        embedding: np.ndarray,
        style_tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add a celebrity to the database.

        Args:
            celebrity_id: Unique identifier
            name: Celebrity name
            embedding: 512-dim face embedding
            style_tags: List of style tags (e.g., ["甜美", "御姐", "少年感"])
            metadata: Additional metadata

        Returns:
            Index in database
        """
        # Normalize embedding for cosine similarity
        normed = embedding / (np.linalg.norm(embedding) + 1e-8)

        celebrity = {
            "id": celebrity_id,
            "name": name,
            "embedding": normed,
            "style_tags": style_tags or [],
            "metadata": metadata or {},
        }

        idx = len(self._celebrities)
        self._celebrities.append(celebrity)
        self._index.add(normed.reshape(1, -1).astype(np.float32))

        return idx

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        style_filter: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Search for most similar celebrities.

        Args:
            query_embedding: 512-dim query embedding
            k: Number of results to return
            style_filter: If provided, only return results with these tags

        Returns:
            List of dicts with celebrity info and similarity score
        """
        normed = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores, indices = self._index.search(normed.reshape(1, -1).astype(np.float32), k * 3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._celebrities):
                continue

            celeb = self._celebrities[idx].copy()
            celeb["similarity"] = float(score)

            # Apply style filter
            if style_filter:
                if not any(tag in celeb.get("style_tags", []) for tag in style_filter):
                    continue

            results.append(celeb)

            if len(results) >= k:
                break

        return results

    def get_celebrity(self, idx: int) -> Optional[Dict]:
        """Get celebrity by index."""
        if 0 <= idx < len(self._celebrities):
            return self._celebrities[idx].copy()
        return None

    def get_by_id(self, celebrity_id: str) -> Optional[Dict]:
        """Get celebrity by ID."""
        for celeb in self._celebrities:
            if celeb["id"] == celebrity_id:
                return celeb.copy()
        return None

    def __len__(self) -> int:
        return len(self._celebrities)

    def save(self, path: str) -> None:
        """Save database to disk.

        Args:
            path: Path to save file (without extension)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, f"{path}.index")

        # Save celebrity metadata
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(self._celebrities, f)

    def load(self, path: str) -> None:
        """Load database from disk.

        Args:
            path: Path to saved file (without extension)
        """
        if not os.path.exists(f"{path}.index"):
            raise FileNotFoundError(f"Index file not found: {path}.index")

        # Load FAISS index
        self._index = faiss.read_index(f"{path}.index")

        # Load celebrity metadata
        with open(f"{path}.meta", "rb") as f:
            self._celebrities = pickle.load(f)

    @classmethod
    def from_images(
        cls,
        embedder,
        image_paths: List[str],
        names: List[str],
        style_tags: Optional[List[List[str]]] = None,
    ) -> "CelebrityDatabase":
        """Build database from a list of celebrity images.

        Args:
            embedder: CelebrityEmbedder instance
            image_paths: List of paths to celebrity images
            names: List of celebrity names
            style_tags: Optional list of style tags per celebrity

        Returns:
            Populated CelebrityDatabase
        """
        db = cls()

        for i, (img_path, name) in enumerate(zip(image_paths, names)):
            img = cv2.imread(img_path)
            if img is None:
                continue

            embedding = embedder.embed(img)
            if embedding is None:
                continue

            tags = style_tags[i] if style_tags and i < len(style_tags) else []
            db.add_celebrity(
                celebrity_id=f"celeb_{i}",
                name=name,
                embedding=embedding,
                style_tags=tags,
            )

        return db
