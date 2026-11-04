"""Chunking strategies."""

import re

from src.strategies.base import ChunkingStrategyBase


class WholeDocumentChunking(ChunkingStrategyBase):
    def split(self, text: str) -> list[str]:
        return [text] if text else []


class FixedSizeChunking(ChunkingStrategyBase):
    def __init__(self, size: int = 800, overlap: int = 100):
        if size < 1 or overlap < 0 or overlap >= size:
            raise ValueError("Require size > overlap >= 0")
        self.size = size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        step = self.size - self.overlap
        return [
            text[index : index + self.size]
            for index in range(0, len(text), step)
            if text[index : index + self.size].strip()
            and (index == 0 or len(text[index : index + self.size]) > self.overlap)
        ]


class ParagraphChunking(ChunkingStrategyBase):
    def split(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
