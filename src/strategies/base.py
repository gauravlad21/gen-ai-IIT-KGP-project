"""Simple inheritance-first base classes for strategy components."""

from abc import ABC, abstractmethod
from typing import Any

from src.models import Document, Evidence


class EmbedderBase(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class ChunkingStrategyBase(ABC):
    @abstractmethod
    def split(self, text: str) -> list[str]: ...


class SearchStrategyBase(ABC):
    @abstractmethod
    def search(self, query: str, documents: list[Document], limit: int = 4) -> list[Evidence]: ...


class RAGStrategyBase(ABC):
    @abstractmethod
    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]: ...


class PromptStrategyBase(ABC):
    @abstractmethod
    def build(self, question: str, evidence: list[Evidence]) -> str: ...


class AnswerStrategyBase(ABC):
    @abstractmethod
    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        prompt,
        history: list[dict[str, str]] | None = None,
    ) -> dict: ...


class MemoryStrategyBase(ABC):
    @abstractmethod
    def get(self, session_id: str) -> list[dict[str, str]]: ...

    @abstractmethod
    def add(self, session_id: str, question: str, answer: str) -> None: ...


class EvaluatorBase(ABC):
    @abstractmethod
    def evaluate(self, answer: dict, evidence: list[Evidence], expected_answer: str | None = None) -> dict: ...
