"""Small replaceable strategies used by the copilot pipeline."""

import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from src.data_loader import Document


@dataclass
class Evidence:
    doc_id: str
    text: str
    question: str = ""
    answer: str = ""
    image_path: str | None = None
    score: float = 0.0


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Tiny offline encoder for a demo; swap in a trained embedding model later."""

    def __init__(self, dimensions: int = 128):
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for word in re.findall(r"\w+", text.lower()):
                slot = int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.dimensions
                vector[slot] += 1.0
            length = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / length for value in vector])
        return vectors


class SentenceTransformerEmbedder:
    """Optional learned encoder; model downloads happen only when selected."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class SearchStrategy(Protocol):
    def search(self, query: str, documents: list[Document], limit: int = 4) -> list[Evidence]: ...


class ChunkingStrategy(Protocol):
    def split(self, text: str) -> list[str]: ...


class WholeDocumentChunking:
    def split(self, text: str) -> list[str]:
        return [text] if text else []


class FixedSizeChunking:
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


class ParagraphChunking:
    def split(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


class RAGStrategy(Protocol):
    def retrieve(
        self, query: str, documents: list[Document], image: Any = None
    ) -> list[Evidence]: ...


def _qa_evidence(
    documents: list[Document], chunker: ChunkingStrategy | None = None
) -> list[Evidence]:
    chunker = chunker or WholeDocumentChunking()
    return [
        Evidence(
            doc_id=document.doc_id,
            text=chunk,
            question=pair.question,
            answer=pair.answer,
            image_path=document.image_path,
        )
        for document in documents
        for pair in document.qa_pairs
        for chunk in chunker.split(document.content)
    ]


class KeywordSearch:
    """Simple BM25-like word overlap search requiring no model or database."""

    def search(self, query: str, documents: list[Document], limit: int = 4) -> list[Evidence]:
        terms = set(re.findall(r"\w+", query.lower()))
        ranked = _qa_evidence(documents)
        ranked.sort(
            key=lambda item: len(
                terms & set(re.findall(r"\w+", f"{item.question} {item.text}".lower()))
            ),
            reverse=True,
        )
        return ranked[:limit]


class ChromaSearch:
    """Chroma vector search. The encoder can be replaced without changing the pipeline."""

    def __init__(
        self,
        path: str | None = None,
        embedder: Embedder | None = None,
        collection_name: str = "finance_sample",
        chunker: ChunkingStrategy | None = None,
    ):
        import chromadb

        self.client = chromadb.PersistentClient(
            path=path or os.getenv("CHROMA_PATH", "data/processed/chroma")
        )
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = embedder or HashingEmbedder()
        self.chunker = chunker or WholeDocumentChunking()

    def search(self, query: str, documents: list[Document], limit: int = 4) -> list[Evidence]:
        if limit < 1:
            raise ValueError("limit must be positive")
        items = _qa_evidence(documents, self.chunker)
        if items:
            self.collection.upsert(
                ids=[f"{item.doc_id}:{index}" for index, item in enumerate(items)],
                documents=[f"{item.question}\n{item.answer}\n{item.text}" for item in items],
                metadatas=[
                    {
                        "doc_id": item.doc_id,
                        "question": item.question,
                        "answer": item.answer,
                        "image_path": item.image_path or "",
                    }
                    for item in items
                ],
                embeddings=self.embedder.encode(
                    [f"{item.question} {item.text}" for item in items]
                ),
            )
        count = self.collection.count()
        if not count:
            return []
        result = self.collection.query(
            query_embeddings=self.embedder.encode([query]),
            n_results=min(limit, count),
            include=["documents", "metadatas", "distances"],
        )
        evidence = []
        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            evidence.append(
                Evidence(
                    doc_id=metadata["doc_id"],
                    text=text,
                    question=metadata["question"],
                    answer=metadata["answer"],
                    image_path=metadata.get("image_path") or None,
                    score=1.0 / (1.0 + float(distance)),
                )
            )
        return evidence


class SimpleRAG:
    def __init__(self, search: SearchStrategy | None = None):
        self.search = search or ChromaSearch()

    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        return self.search.search(query, documents)


class VisualRAG:
    """Uses a supplied vision function; no pretend image model is bundled."""

    def __init__(self, vision: Callable | None = None):
        self.vision = vision

    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        if self.vision is None or image is None:
            return []
        return self.vision(query, image)


class CombinedRAG:
    def __init__(self, text: RAGStrategy | None = None, visual: RAGStrategy | None = None):
        self.text = text or SimpleRAG()
        self.visual = visual or VisualRAG()

    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        return self.text.retrieve(query, documents) + self.visual.retrieve(query, documents, image)


class AutoRAG(CombinedRAG):
    """Use text retrieval by default and add image QA when an image is supplied."""

    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        if image is None:
            return self.text.retrieve(query, documents)
        return super().retrieve(query, documents, image)


class MultiStepRAG(SimpleRAG):
    """Search the original question and simple clauses, then remove duplicate hits."""

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
    """A small retrieve-then-rerank variant; inject a learned reranker later."""

    def retrieve(self, query: str, documents: list[Document], image: Any = None) -> list[Evidence]:
        found = self.search.search(query, documents, limit=8)
        words = set(re.findall(r"\w+", query.lower()))
        found.sort(
            key=lambda item: len(
                words & set(re.findall(r"\w+", f"{item.question} {item.text}".lower()))
            ),
            reverse=True,
        )
        return found[:4]


class PromptStrategy(Protocol):
    def build(self, question: str, evidence: list[Evidence]) -> str: ...


class SimplePrompt:
    def build(self, question: str, evidence: list[Evidence]) -> str:
        context = "\n".join(item.text for item in evidence)
        return f"Question: {question}\nEvidence:\n{context}"


class ZeroShotPrompt(SimplePrompt):
    def build(self, question: str, evidence: list[Evidence]) -> str:
        return (
            "You answer questions about financial documents. Use only evidence below. "
            "If evidence is insufficient, refuse and cite the source document.\n"
            + super().build(question, evidence)
        )


class FewShotPrompt(SimplePrompt):
    """Place to add manually verified examples; none are invented by default."""

    def __init__(self, examples: list[dict[str, str]] | None = None):
        self.examples = examples or []

    def build(self, question: str, evidence: list[Evidence]) -> str:
        examples = "\n".join(
            f"Example question: {item['question']}\nVerified answer: {item['answer']}"
            for item in self.examples
        )
        return examples + "\n" + super().build(question, evidence)


class DecompositionPrompt(SimplePrompt):
    """Break a multi-part question into evidence-checkable parts."""

    def build(self, question: str, evidence: list[Evidence]) -> str:
        return (
            "Check each part of the question against the evidence before answering. "
            "Do not include hidden reasoning in the final answer.\n"
            + super().build(question, evidence)
        )


class AnswerStrategy(Protocol):
    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        prompt: PromptStrategy,
        history: list[dict[str, str]] | None = None,
    ) -> dict: ...


class LLMAnswer:
    """Connect an injected model callable without tying the pipeline to one provider."""

    def __init__(self, model: Callable[[str], dict]):
        self.model = model

    def answer(self, question: str, evidence: list[Evidence], prompt: PromptStrategy, history=None) -> dict:
        context = "\n".join(
            f"{item.doc_id}: {item.question} {item.answer} {item.text}" for item in evidence
        )
        request = prompt.build(question, evidence)
        if history:
            request = f"Conversation history: {history}\n{request}"
        return self.model(request + "\nRetrieved context:\n" + context)


class DatasetAnswer:
    """Uses the dataset's existing QA answer as a transparent no-LLM baseline."""

    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        prompt: PromptStrategy,
        history: list[dict[str, str]] | None = None,
    ) -> dict:
        if not evidence:
            return _refusal("No document evidence was found.")
        prompt.build(question, evidence)
        question_terms = _meaningful_words(question)
        best = max(
            evidence,
            key=lambda item: len(question_terms & _meaningful_words(item.question)),
        )
        overlap = len(question_terms & _meaningful_words(best.question))
        if not best.answer or overlap == 0:
            return _refusal("The retrieved passages do not contain a matching dataset answer.")
        return {
            "answer": best.answer,
            "citations": [{"doc_id": best.doc_id}],
            "should_refuse": False,
            "reason": "",
        }


class CriticAnswer:
    """A small wrapper that refuses an answer if its citation is not in evidence."""

    def __init__(self, answerer: AnswerStrategy | None = None):
        self.answerer = answerer or DatasetAnswer()

    def answer(self, question: str, evidence: list[Evidence], prompt: PromptStrategy, history=None) -> dict:
        result = self.answerer.answer(question, evidence, prompt, history)
        valid_ids = {item.doc_id for item in evidence}
        if any(citation.get("doc_id") not in valid_ids for citation in result["citations"]):
            return _refusal("Answer citation did not match retrieved evidence.")
        return result


class VotingAnswer:
    """Use the most common response from injected answer strategies."""

    def __init__(self, answerers: list[AnswerStrategy]):
        if not answerers:
            raise ValueError("Provide at least one answer strategy.")
        self.answerers = answerers

    def answer(self, question: str, evidence: list[Evidence], prompt: PromptStrategy, history=None) -> dict:
        answers = [
            item.answer(question, evidence, prompt, history)
            for item in self.answerers
        ]
        counts = Counter(item["answer"] for item in answers)
        winning_answer, votes = counts.most_common(1)[0]
        if votes <= len(self.answerers) // 2:
            return _refusal("Answer strategies did not agree.")
        return next(item for item in answers if item["answer"] == winning_answer)


class MemoryStrategy(Protocol):
    def get(self, session_id: str) -> list[dict[str, str]]: ...
    def add(self, session_id: str, question: str, answer: str) -> None: ...


class ConversationMemory:
    def __init__(self):
        self.turns: dict[str, list[dict[str, str]]] = {}

    def get(self, session_id: str) -> list[dict[str, str]]:
        return self.turns.get(session_id, [])[-5:]

    def add(self, session_id: str, question: str, answer: str) -> None:
        self.turns.setdefault(session_id, []).append(
            {"question": question, "answer": answer}
        )


class NoMemory:
    def get(self, session_id: str) -> list[dict[str, str]]:
        return []

    def add(self, session_id: str, question: str, answer: str) -> None:
        return None


class Evaluator(Protocol):
    def evaluate(self, answer: dict, evidence: list[Evidence], expected_answer: str | None = None) -> dict[str, Any]: ...


class SimpleEvaluator:
    def evaluate(self, answer: dict, evidence: list[Evidence], expected_answer: str | None = None) -> dict:
        return {
            "retrieved_count": len(evidence),
            "has_citation": bool(answer.get("citations")),
            "refused": bool(answer.get("should_refuse")),
            "exact_match": (
                answer.get("answer", "").strip().casefold()
                == expected_answer.strip().casefold()
                if expected_answer is not None
                else None
            ),
        }


class JudgeEvaluator:
    """Inject one evaluator callable; it can be backed by an LLM later."""

    def __init__(self, judge: Callable[[dict, list[Evidence]], dict]):
        self.judge = judge

    def evaluate(self, answer: dict, evidence: list[Evidence], expected_answer: str | None = None) -> dict:
        return self.judge(answer, evidence)


class VotingEvaluator:
    """Combine multiple evaluator callables by majority vote."""

    def __init__(self, judges: list[Callable[[dict, list[Evidence]], dict]]):
        if not judges:
            raise ValueError("Provide at least one evaluator.")
        self.judges = judges

    def evaluate(self, answer: dict, evidence: list[Evidence], expected_answer: str | None = None) -> dict:
        results = [judge(answer, evidence) for judge in self.judges]
        keys = set().union(*(result.keys() for result in results))
        majority = {}
        for key in keys:
            counts = Counter(repr(result.get(key)) for result in results)
            winner = counts.most_common(1)[0][0]
            majority[key] = next(result[key] for result in results if repr(result.get(key)) == winner)
        return majority


def _refusal(reason: str) -> dict:
    return {
        "answer": "I cannot answer from the available document evidence.",
        "citations": [],
        "should_refuse": True,
        "reason": reason,
    }


def _meaningful_words(text: str) -> set[str]:
    stop_words = {"a", "an", "and", "are", "for", "how", "in", "is", "it", "of", "on", "the", "to", "was", "what", "who"}
    return set(re.findall(r"\w+", text.lower())) - stop_words


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Helper for trying a different vector encoding or comparing test vectors."""
    return sum(a * b for a, b in zip(left, right))
