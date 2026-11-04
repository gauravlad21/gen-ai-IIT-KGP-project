"""Memory strategies."""

from src.strategies.base import MemoryStrategyBase


class ConversationMemory(MemoryStrategyBase):
    def __init__(self):
        self.turns: dict[str, list[dict[str, str]]] = {}

    def get(self, session_id: str) -> list[dict[str, str]]:
        return self.turns.get(session_id, [])[-5:]

    def add(self, session_id: str, question: str, answer: str) -> None:
        self.turns.setdefault(session_id, []).append({"question": question, "answer": answer})


class NoMemory(MemoryStrategyBase):
    def get(self, session_id: str) -> list[dict[str, str]]:
        return []

    def add(self, session_id: str, question: str, answer: str) -> None:
        return None
