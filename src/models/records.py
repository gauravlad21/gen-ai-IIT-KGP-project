"""Core record shapes used in the app and pipeline."""

from dataclasses import dataclass


@dataclass
class QAPair:
    question: str
    answer: str


@dataclass
class Document:
    doc_id: str
    content: str
    qa_pairs: list[QAPair]
    image_path: str | None = None


@dataclass
class Evidence:
    doc_id: str
    text: str
    question: str = ""
    answer: str = ""
    image_path: str | None = None
    score: float = 0.0
