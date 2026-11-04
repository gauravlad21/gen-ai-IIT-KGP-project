"""Environment-driven configuration for strategy selection."""

import os
from dataclasses import dataclass


def _load_local_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


@dataclass
class AppConfig:
    search_strategy: str = "chroma"
    rag_strategy: str = "auto"
    prompt_strategy: str = "simple"
    answer_strategy: str = "dataset"
    memory_strategy: str = "conversation"
    evaluator_strategy: str = "simple"
    chunking_strategy: str = "whole"
    embedder_strategy: str = "hashing"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: str = "data/processed/chroma"
    chroma_collection: str = "finance_sample"
    metrics_db_path: str = "data/processed/metrics.sqlite3"
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_limit: int = 4

    @classmethod
    def from_env(cls) -> "AppConfig":
        _load_local_env()
        return cls(
            search_strategy=os.getenv("SEARCH_STRATEGY", cls.search_strategy),
            rag_strategy=os.getenv("RAG_STRATEGY", cls.rag_strategy),
            prompt_strategy=os.getenv("PROMPT_STRATEGY", cls.prompt_strategy),
            answer_strategy=os.getenv("ANSWER_STRATEGY", cls.answer_strategy),
            memory_strategy=os.getenv("MEMORY_STRATEGY", cls.memory_strategy),
            evaluator_strategy=os.getenv("EVALUATOR_STRATEGY", cls.evaluator_strategy),
            chunking_strategy=os.getenv("CHUNKING_STRATEGY", cls.chunking_strategy),
            embedder_strategy=os.getenv("EMBEDDER_STRATEGY", cls.embedder_strategy),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", cls.embedding_model_name),
            chroma_path=os.getenv("CHROMA_PATH", cls.chroma_path),
            chroma_collection=os.getenv("CHROMA_COLLECTION", cls.chroma_collection),
            metrics_db_path=os.getenv("METRICS_DB_PATH", cls.metrics_db_path),
            chunk_size=int(os.getenv("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(cls.chunk_overlap))),
            retrieval_limit=int(os.getenv("RETRIEVAL_LIMIT", str(cls.retrieval_limit))),
        )
