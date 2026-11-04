"""RAG routing strategies."""

import re
from typing import Any, Callable

from src.models import Document, Evidence
from src.strategies.base import RAGStrategyBase
from src.strategies.search import ChromaSearch


class SimpleRAG(RAGStrategyBase):
    def __init__(self, search=None):
        self.search = search or ChromaSearch()

    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        return self.search.search(query, documents)


class VisualRAG(RAGStrategyBase):
    def __init__(self, vision: Callable | None = None):
        self.vision = vision

    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        if self.vision is None or image is None:
            return []
        return self.vision(query, image)


class CombinedRAG(RAGStrategyBase):
    def __init__(self, text=None, visual=None):
        self.text = text or SimpleRAG()
        self.visual = visual or VisualRAG()

    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        return self.text.retrieve(query, documents) + self.visual.retrieve(query, documents, image)


class AutoRAG(CombinedRAG):
    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        if image is None:
            return self.text.retrieve(query, documents)
        return super().retrieve(query, documents, image)


class MultiStepRAG(SimpleRAG):
    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        clauses = [part.strip() for part in re.split(r"\band\b|[,?]", query, flags=re.I) if part.strip()]
        found = []
        seen = set()
        for clause in [query, *clauses]:
            for item in self.search.search(clause, documents):
                key = (item.doc_id, item.question)
                if key not in seen:
                    found.append(item)
                    seen.add(key)
        return found


class AgenticRAG(SimpleRAG):
    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        found = self.search.search(query, documents, limit=8)
        words = set(re.findall(r"\w+", query.lower()))
        found.sort(
            key=lambda item: len(words & set(re.findall(r"\w+", f"{item.question} {item.text}".lower()))),
            reverse=True,
        )
        return found[:4]
