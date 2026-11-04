"""Embedding strategies."""

import hashlib
import math

from src.strategies.base import EmbedderBase


class HashingEmbedder(EmbedderBase):
    """Tiny offline encoder for a demo."""

    def __init__(self, dimensions: int = 128):
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for word in text.lower().split():
                slot = int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.dimensions
                vector[slot] += 1.0
            length = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / length for value in vector])
        return vectors


class SentenceTransformerEmbedder(EmbedderBase):
    """Optional learned encoder."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
