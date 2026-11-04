"""Simple SQLite storage for pipeline run metrics and comparison rows."""

import sqlite3
import json
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_setups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    setup_name TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    sample_limit INTEGER NOT NULL,
                    vector_db TEXT NOT NULL,
                    vector_path TEXT NOT NULL,
                    vector_collection TEXT NOT NULL,
                    embedder_strategy TEXT NOT NULL,
                    chunking_strategy TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    indexed_items INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    run_name TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    faithfulness REAL NOT NULL,
                    answer_accuracy REAL NOT NULL,
                    refusal_rate REAL NOT NULL,
                    citation_rate REAL NOT NULL,
                    taxonomy_counts_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    eval_run_id INTEGER NOT NULL,
                    point_index INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    expected_answer TEXT NOT NULL,
                    predicted_answer TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    image_path TEXT NOT NULL DEFAULT '',
                    is_correct INTEGER NOT NULL,
                    is_faithful INTEGER NOT NULL,
                    failure_category TEXT NOT NULL,
                    note TEXT NOT NULL,
                    FOREIGN KEY(eval_run_id) REFERENCES eval_runs(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    run_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    route TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    strategies_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    critiques_json TEXT NOT NULL,
                    retrieved_json TEXT NOT NULL
                )
                """
            )
            self._ensure_column(connection, "eval_runs", "taxonomy_counts_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(connection, "eval_observations", "failure_category", "TEXT NOT NULL DEFAULT 'none'")
            self._ensure_column(connection, "eval_observations", "image_path", "TEXT NOT NULL DEFAULT ''")

    def _connect(self):
        return sqlite3.connect(self.path)

    @staticmethod
    def _ensure_column(connection, table_name: str, column_name: str, column_def: str) -> None:
        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")

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

    def insert_rag_setup(self, row: dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO rag_setups (
                    setup_name, dataset_name, sample_limit, vector_db, vector_path, vector_collection,
                    embedder_strategy, chunking_strategy, chunk_size, chunk_overlap, indexed_items
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["setup_name"],
                    row["dataset_name"],
                    int(row["sample_limit"]),
                    row["vector_db"],
                    row["vector_path"],
                    row["vector_collection"],
                    row["embedder_strategy"],
                    row["chunking_strategy"],
                    int(row["chunk_size"]),
                    int(row["chunk_overlap"]),
                    int(row["indexed_items"]),
                ),
            )
            return int(cursor.lastrowid)

    def list_rag_setups(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, setup_name, dataset_name, sample_limit, vector_db, vector_path,
                       vector_collection, embedder_strategy, chunking_strategy, chunk_size, chunk_overlap,
                       indexed_items
                FROM rag_setups
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        keys = [
            "id",
            "created_at",
            "setup_name",
            "dataset_name",
            "sample_limit",
            "vector_db",
            "vector_path",
            "vector_collection",
            "embedder_strategy",
            "chunking_strategy",
            "chunk_size",
            "chunk_overlap",
            "indexed_items",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def insert_eval_run(self, summary: dict[str, Any], observations: list[dict[str, Any]]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO eval_runs (
                    run_name, sample_size, faithfulness, answer_accuracy, refusal_rate, citation_rate, taxonomy_counts_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary["run_name"],
                    int(summary["sample_size"]),
                    float(summary["faithfulness"]),
                    float(summary["answer_accuracy"]),
                    float(summary["refusal_rate"]),
                    float(summary["citation_rate"]),
                    json.dumps(summary.get("taxonomy_counts", {})),
                ),
            )
            eval_run_id = int(cursor.lastrowid)
            for index, item in enumerate(observations):
                connection.execute(
                    """
                    INSERT INTO eval_observations (
                        eval_run_id, point_index, question, expected_answer, predicted_answer,
                        doc_id, image_path, is_correct, is_faithful, failure_category, note
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eval_run_id,
                        index,
                        item["question"],
                        item["expected_answer"],
                        item["predicted_answer"],
                        item["doc_id"],
                        item.get("image_path", ""),
                        int(bool(item["is_correct"])),
                        int(bool(item["is_faithful"])),
                        item.get("failure_category", "none"),
                        item["note"],
                    ),
                )
            return eval_run_id

    def list_eval_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, run_name, sample_size, faithfulness, answer_accuracy, refusal_rate, citation_rate, taxonomy_counts_json
                FROM eval_runs ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        keys = [
            "id",
            "created_at",
            "run_name",
            "sample_size",
            "faithfulness",
            "answer_accuracy",
            "refusal_rate",
            "citation_rate",
            "taxonomy_counts_json",
        ]
        parsed = []
        for row in rows:
            record = dict(zip(keys, row))
            record["taxonomy_counts"] = json.loads(record.pop("taxonomy_counts_json") or "{}")
            parsed.append(record)
        return parsed

    def list_eval_observations(self, eval_run_id: int, limit: int = 300) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, eval_run_id, point_index, question, expected_answer, predicted_answer,
                       doc_id, image_path, is_correct, is_faithful, failure_category, note
                FROM eval_observations
                WHERE eval_run_id = ?
                ORDER BY point_index ASC
                LIMIT ?
                """,
                (eval_run_id, limit),
            ).fetchall()
        keys = [
            "id",
            "eval_run_id",
            "point_index",
            "question",
            "expected_answer",
            "predicted_answer",
            "doc_id",
            "image_path",
            "is_correct",
            "is_faithful",
            "failure_category",
            "note",
        ]
        return [dict(zip(keys, row)) for row in rows]

    def insert_query_trace(self, row: dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO query_traces (
                    run_name, session_id, question, answer, route, latency_ms,
                    strategies_json, metrics_json, steps_json, citations_json, critiques_json, retrieved_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["run_name"],
                    row["session_id"],
                    row["question"],
                    row["answer"],
                    row["route"],
                    float(row["latency_ms"]),
                    json.dumps(row["strategies"]),
                    json.dumps(row["metrics"]),
                    json.dumps(row["steps"]),
                    json.dumps(row["citations"]),
                    json.dumps(row["critiques"]),
                    json.dumps(row["retrieved"]),
                ),
            )
            return int(cursor.lastrowid)

    def list_query_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, run_name, session_id, question, answer, route, latency_ms,
                       strategies_json, metrics_json, steps_json, citations_json, critiques_json, retrieved_json
                FROM query_traces
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        keys = [
            "id",
            "created_at",
            "run_name",
            "session_id",
            "question",
            "answer",
            "route",
            "latency_ms",
            "strategies_json",
            "metrics_json",
            "steps_json",
            "citations_json",
            "critiques_json",
            "retrieved_json",
        ]
        parsed = []
        for row in rows:
            item = dict(zip(keys, row))
            item["strategies"] = json.loads(item.pop("strategies_json") or "{}")
            item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
            item["steps"] = json.loads(item.pop("steps_json") or "[]")
            item["citations"] = json.loads(item.pop("citations_json") or "[]")
            item["critiques"] = json.loads(item.pop("critiques_json") or "[]")
            item["retrieved"] = json.loads(item.pop("retrieved_json") or "[]")
            parsed.append(item)
        return parsed
