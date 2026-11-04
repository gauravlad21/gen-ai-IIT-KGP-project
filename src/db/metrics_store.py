"""Simple SQLite storage for pipeline run metrics and comparison rows."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MetricsStore:
    path: str = "data/processed/metrics.sqlite3"

    def __post_init__(self):
        destination = Path(self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    run_name TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    route TEXT NOT NULL,
                    search_strategy TEXT NOT NULL,
                    rag_strategy TEXT NOT NULL,
                    prompt_strategy TEXT NOT NULL,
                    answer_strategy TEXT NOT NULL,
                    memory_strategy TEXT NOT NULL,
                    evaluator_strategy TEXT NOT NULL,
                    chunking_strategy TEXT NOT NULL,
                    embedder_strategy TEXT NOT NULL,
                    retrieved_count INTEGER NOT NULL,
                    has_citation INTEGER NOT NULL,
                    refused INTEGER NOT NULL,
                    exact_match REAL,
                    latency_ms REAL NOT NULL
                )
                """
            )

    def _connect(self):
        return sqlite3.connect(self.path)

    def insert_run(self, row: dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO run_metrics (
                    run_name, question, answer, route,
                    search_strategy, rag_strategy, prompt_strategy, answer_strategy,
                    memory_strategy, evaluator_strategy, chunking_strategy, embedder_strategy,
                    retrieved_count, has_citation, refused, exact_match, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["run_name"],
                    row["question"],
                    row["answer"],
                    row["route"],
                    row["search_strategy"],
                    row["rag_strategy"],
                    row["prompt_strategy"],
                    row["answer_strategy"],
                    row["memory_strategy"],
                    row["evaluator_strategy"],
                    row["chunking_strategy"],
                    row["embedder_strategy"],
                    int(row["retrieved_count"]),
                    int(bool(row["has_citation"])),
                    int(bool(row["refused"])),
                    row.get("exact_match"),
                    float(row["latency_ms"]),
                ),
            )
            return int(cursor.lastrowid)

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, run_name, question, answer, route,
                       search_strategy, rag_strategy, prompt_strategy, answer_strategy,
                       memory_strategy, evaluator_strategy, chunking_strategy, embedder_strategy,
                       retrieved_count, has_citation, refused, exact_match, latency_ms
                FROM run_metrics
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        keys = [
            "id",
            "created_at",
            "run_name",
            "question",
            "answer",
            "route",
            "search_strategy",
            "rag_strategy",
            "prompt_strategy",
            "answer_strategy",
            "memory_strategy",
            "evaluator_strategy",
            "chunking_strategy",
            "embedder_strategy",
            "retrieved_count",
            "has_citation",
            "refused",
            "exact_match",
            "latency_ms",
        ]
        return [dict(zip(keys, row)) for row in rows]
