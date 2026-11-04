"""Search strategies (keyword and Chroma)."""

import os
import re

from src.models import Document, Evidence
from src.strategies.base import ChunkingStrategyBase, SearchStrategyBase
from src.strategies.chunking import WholeDocumentChunking
from src.strategies.embeddings import HashingEmbedder


def build_qa_evidence(
    documents: list[Document],
    chunker: ChunkingStrategyBase | None = None,
) -> list[Evidence]:
    use_chunker = chunker or WholeDocumentChunking()
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
        for chunk in use_chunker.split(document.content)
    ]


class KeywordSearch(SearchStrategyBase):
    def search(self, query: str, documents: list[Document], limit: int = 4) -> list[Evidence]:
        terms = set(re.findall(r"\w+", query.lower()))
        ranked = build_qa_evidence(documents)
        ranked.sort(
            key=lambda item: len(terms & set(re.findall(r"\w+", f"{item.question} {item.text}".lower()))),
            reverse=True,
        )
        return ranked[:limit]


class LimitedSearch(SearchStrategyBase):
    """Force a configured top-k, regardless of caller defaults."""

    def __init__(self, base: SearchStrategyBase, limit: int):
        if limit < 1:
            raise ValueError("limit must be positive")
        self.base = base
        self.limit = limit

    def search(self, query: str, documents: list[Document], limit: int = 4) -> list[Evidence]:
        return self.base.search(query, documents, self.limit)


class ChromaSearch(SearchStrategyBase):
    def __init__(
        self,
        path: str | None = None,
        embedder=None,
        collection_name: str = "finance_sample",
        chunker: ChunkingStrategyBase | None = None,
    ):
        import chromadb

        self.client = chromadb.PersistentClient(path=path or os.getenv("CHROMA_PATH", "data/processed/chroma"))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = embedder or HashingEmbedder()
        self.chunker = chunker or WholeDocumentChunking()

    def search(self, query: str, documents: list[Document], limit: int = 4) -> list[Evidence]:
        if limit < 1:
            raise ValueError("limit must be positive")
        items = build_qa_evidence(documents, self.chunker)
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
                embeddings=self.embedder.encode([f"{item.question} {item.text}" for item in items]),
            )
        count = self.collection.count()
        if not count:
            return []
        result = self.collection.query(
            query_embeddings=self.embedder.encode([query]),
            n_results=min(limit, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Evidence(
                doc_id=metadata["doc_id"],
                text=text,
                question=metadata["question"],
                answer=metadata["answer"],
                image_path=metadata.get("image_path") or None,
                score=1.0 / (1.0 + float(distance)),
            )
            for text, metadata, distance in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
