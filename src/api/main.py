"""FastAPI service exposing pipeline query and metrics dashboard APIs."""

from pathlib import Path
from time import perf_counter
from contextlib import asynccontextmanager
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import AppConfig
from src.data import load_dataset_subset, load_saved_sample, save_sample
from src.db import MetricsStore
from src.pipeline import build_strategies_from_config, run_pipeline

from .schemas import (
    EvaluationRequest,
    EvaluationResponse,
    MetricsCreateRequest,
    MetricsListResponse,
    QueryRequest,
    QueryResponse,
    RagSetupRequest,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup()
    yield


app = FastAPI(title="Financial Copilot API", version="0.1.0", lifespan=lifespan)

_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _ROOT / "frontend"
_SAMPLE_IMAGE_DIR = _ROOT / "data" / "sample" / "images"

app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
app.mount("/sample-images", StaticFiles(directory=str(_SAMPLE_IMAGE_DIR)), name="sample_images")


def _strategy_snapshot(config: AppConfig) -> dict:
    return {
        "search_strategy": config.search_strategy,
        "rag_strategy": config.rag_strategy,
        "prompt_strategy": config.prompt_strategy,
        "answer_strategy": config.answer_strategy,
        "memory_strategy": config.memory_strategy,
        "evaluator_strategy": config.evaluator_strategy,
        "chunking_strategy": config.chunking_strategy,
        "embedder_strategy": config.embedder_strategy,
    }


def _ensure_state() -> None:
    required = ("config", "strategies", "metrics_store", "documents")
    if all(hasattr(app.state, name) for name in required):
        return
    startup()


def startup():
    config = AppConfig.from_env()
    app.state.config = config
    app.state.strategies = build_strategies_from_config(config)
    app.state.metrics_store = MetricsStore(config.metrics_db_path)
    try:
        app.state.documents = load_saved_sample()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Missing data/sample/documents.json. Run `uv run python -m scripts.download_sample`."
        ) from exc


def _critiques(answer: dict, evidence: list[dict]) -> list[str]:
    critiques = []
    if answer.get("should_refuse"):
        critiques.append(f"Refusal: {answer.get('reason', 'No reason provided')}")
    if not evidence:
        critiques.append("No evidence chunks retrieved.")
    if not answer.get("citations"):
        critiques.append("No citations were attached.")
    else:
        critiques.append("Citations were attached for retrieved evidence.")
    if evidence and all(item.get("score", 0) < 0.3 for item in evidence):
        critiques.append("Similarity scores are low; answer confidence should be treated carefully.")
    return critiques


def _failure_category(
    expected_answer: str,
    predicted_answer: str,
    should_refuse: bool,
    citations: list[dict],
    expected_doc_id: str,
    retrieved_count: int,
) -> str:
    if retrieved_count == 0:
        return "retrieval_miss"
    if citations and not any(item.get("doc_id") == expected_doc_id for item in citations):
        return "citation_error"
    if should_refuse:
        return "reasoning_error"
    if predicted_answer.strip().casefold() != expected_answer.strip().casefold():
        return "hallucination"
    return "none"


def _image_url(image_path: str | None) -> str:
    if not image_path:
        return ""
    return f"/sample-images/{Path(image_path).name}"


@app.get("/")
def home():
    return FileResponse(str(_FRONTEND_DIR / "pages.html"))


@app.get("/query")
def query_page():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.get("/pages")
def pages():
    return FileResponse(str(_FRONTEND_DIR / "pages.html"))


@app.get("/rag")
def rag_setup_page():
    return FileResponse(str(_FRONTEND_DIR / "rag.html"))


@app.get("/dashboard")
def dashboard():
    return FileResponse(str(_FRONTEND_DIR / "traces.html"))


@app.get("/evaluation")
def evaluation_page():
    return FileResponse(str(_FRONTEND_DIR / "evaluation.html"))


@app.get("/api/health")
def health():
    _ensure_state()
    return {"status": "ok", "documents": len(app.state.documents)}


@app.get("/api/config")
def config():
    _ensure_state()
    return _strategy_snapshot(app.state.config)


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    _ensure_state()
    started = perf_counter()
    try:
        result = run_pipeline(
            question=payload.question,
            documents=app.state.documents,
            session_id=payload.session_id,
            strategies=app.state.strategies,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = (perf_counter() - started) * 1000
    metrics = result["metrics"]
    app.state.metrics_store.insert_run(
        {
            "run_name": payload.run_name,
            "question": payload.question,
            "answer": result["answer"]["answer"],
            "route": result["route"],
            **_strategy_snapshot(app.state.config),
            "retrieved_count": metrics.get("retrieved_count", 0),
            "has_citation": metrics.get("has_citation", False),
            "refused": metrics.get("refused", False),
            "exact_match": metrics.get("exact_match"),
            "latency_ms": latency_ms,
        }
    )
    app.state.metrics_store.insert_query_trace(
        {
            "run_name": payload.run_name,
            "session_id": payload.session_id,
            "question": payload.question,
            "answer": result["answer"]["answer"],
            "route": result["route"],
            "latency_ms": latency_ms,
            "strategies": _strategy_snapshot(app.state.config),
            "metrics": {**metrics, "latency_ms": round(latency_ms, 2)},
            "steps": result["steps"],
            "citations": result["answer"].get("citations", []),
            "critiques": _critiques(
                result["answer"],
                [
                    {
                        "doc_id": item.doc_id,
                        "score": item.score,
                    }
                    for item in result["evidence"]
                ],
            ),
            "retrieved": [
                {
                    "doc_id": item.doc_id,
                    "question": item.question,
                    "answer": item.answer,
                    "chunk_text": item.text[:4000],
                    "image_path": item.image_path,
                    "score": round(item.score, 4),
                }
                for item in result["evidence"]
            ],
        }
    )
    retrieved = [
        {
            "doc_id": item.doc_id,
            "question": item.question,
            "answer": item.answer,
            "chunk_text": item.text[:4000],
            "image_path": item.image_path,
            "score": round(item.score, 4),
        }
        for item in result["evidence"]
    ]
    return QueryResponse(
        answer=result["answer"],
        route=result["route"],
        metrics={**metrics, "latency_ms": round(latency_ms, 2)},
        steps=result["steps"],
        citations=result["answer"].get("citations", []),
        retrieved=retrieved,
        critiques=_critiques(result["answer"], retrieved),
    )


@app.post("/api/metrics")
def create_metrics(payload: MetricsCreateRequest):
    _ensure_state()
    row_id = app.state.metrics_store.insert_run(
        {**payload.model_dump(), **_strategy_snapshot(app.state.config)}
    )
    return {"id": row_id}


@app.get("/api/metrics", response_model=MetricsListResponse)
def list_metrics(limit: int = 200):
    _ensure_state()
    return MetricsListResponse(rows=app.state.metrics_store.list_runs(limit=limit))


@app.get("/api/traces")
def list_traces(limit: int = 100):
    _ensure_state()
    return {"rows": app.state.metrics_store.list_query_traces(limit)}


@app.post("/api/rag/setup")
def rag_setup(payload: RagSetupRequest):
    _ensure_state()
    if payload.vector_db != "chroma":
        raise HTTPException(status_code=400, detail="Only 'chroma' vector db is supported in this project.")
    if payload.chunk_overlap >= payload.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size.")

    if payload.use_saved_sample:
        documents = load_saved_sample()
    else:
        rows = load_dataset_subset(limit=payload.sample_limit, dataset_name=payload.dataset_name)
        documents = save_sample(rows, "data/sample")

    config = AppConfig.from_env()
    config.search_strategy = "chroma"
    config.rag_strategy = "auto"
    config.prompt_strategy = config.prompt_strategy
    config.answer_strategy = config.answer_strategy
    config.memory_strategy = config.memory_strategy
    config.evaluator_strategy = config.evaluator_strategy
    config.embedder_strategy = payload.embedder_strategy
    config.chunking_strategy = payload.chunking_strategy
    config.chunk_size = payload.chunk_size
    config.chunk_overlap = payload.chunk_overlap
    config.chroma_path = payload.vector_path
    config.chroma_collection = payload.vector_collection

    strategies = build_strategies_from_config(config)
    # Force index creation by executing one search call.
    if documents and documents[0].qa_pairs:
        run_pipeline(
            question=documents[0].qa_pairs[0].question,
            documents=documents,
            session_id="indexing",
            strategies=strategies,
        )

    app.state.config = config
    app.state.strategies = strategies
    app.state.documents = documents
    setup_id = app.state.metrics_store.insert_rag_setup(
        {
            "setup_name": payload.setup_name,
            "dataset_name": payload.dataset_name,
            "sample_limit": payload.sample_limit,
            "vector_db": payload.vector_db,
            "vector_path": payload.vector_path,
            "vector_collection": payload.vector_collection,
            "embedder_strategy": payload.embedder_strategy,
            "chunking_strategy": payload.chunking_strategy,
            "chunk_size": payload.chunk_size,
            "chunk_overlap": payload.chunk_overlap,
            "indexed_items": sum(len(item.qa_pairs) for item in documents),
        }
    )
    return {
        "setup_id": setup_id,
        "documents": len(documents),
        "qa_pairs": sum(len(item.qa_pairs) for item in documents),
    }


@app.get("/api/rag/setups")
def list_rag_setups(limit: int = 30):
    _ensure_state()
    return {"rows": app.state.metrics_store.list_rag_setups(limit)}


@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest):
    _ensure_state()
    datapoints = []
    for document in app.state.documents:
        for pair in document.qa_pairs:
            datapoints.append((document.doc_id, pair.question, pair.answer, document.image_path or ""))
            if len(datapoints) >= payload.sample_size:
                break
        if len(datapoints) >= payload.sample_size:
            break
    if not datapoints:
        raise HTTPException(status_code=400, detail="No datapoints available for evaluation.")

    observations = []
    failure_counts: Counter[str] = Counter()
    correct = 0
    faithful = 0
    refused = 0
    cited = 0
    for doc_id, question, expected_answer, image_path in datapoints:
        result = run_pipeline(
            question=question,
            documents=app.state.documents,
            session_id=payload.session_id,
            expected_answer=expected_answer,
            strategies=app.state.strategies,
        )
        predicted = result["answer"]["answer"]
        is_correct = predicted.strip().casefold() == expected_answer.strip().casefold()
        has_citation = bool(result["answer"].get("citations"))
        citation_match = has_citation and any(
            item.get("doc_id") == doc_id for item in result["answer"]["citations"]
        )
        is_faithful = bool(citation_match and not result["answer"].get("should_refuse"))
        category = _failure_category(
            expected_answer=expected_answer,
            predicted_answer=predicted,
            should_refuse=bool(result["answer"].get("should_refuse")),
            citations=result["answer"].get("citations", []),
            expected_doc_id=doc_id,
            retrieved_count=int(result["metrics"].get("retrieved_count", 0)),
        )
        note = (
            "Correct and grounded in cited doc."
            if is_correct and is_faithful
            else "Wrong answer text."
            if not is_correct
            else "Answer text matches but citation/refusal is weak."
        )
        failure_counts[category] += 1
        observations.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "predicted_answer": predicted,
                "doc_id": doc_id,
                "image_path": image_path,
                "image_url": _image_url(image_path),
                "is_correct": is_correct,
                "is_faithful": is_faithful,
                "failure_category": category,
                "note": note,
            }
        )
        correct += int(is_correct)
        faithful += int(is_faithful)
        refused += int(bool(result["answer"].get("should_refuse")))
        cited += int(has_citation)

    total = len(observations)
    summary = {
        "run_name": payload.run_name,
        "sample_size": total,
        "faithfulness": round(100 * faithful / total, 2),
        "answer_accuracy": round(100 * correct / total, 2),
        "refusal_rate": round(100 * refused / total, 2),
        "citation_rate": round(100 * cited / total, 2),
        "taxonomy_counts": dict(failure_counts),
    }
    eval_run_id = app.state.metrics_store.insert_eval_run(summary, observations)
    return EvaluationResponse(eval_run_id=eval_run_id, summary=summary, observations=observations)


@app.get("/api/evaluations")
def list_evaluations(limit: int = 20):
    _ensure_state()
    return {"rows": app.state.metrics_store.list_eval_runs(limit)}


@app.get("/api/evaluations/{eval_run_id}")
def get_evaluation(eval_run_id: int, limit: int = 300):
    _ensure_state()
    observations = app.state.metrics_store.list_eval_observations(eval_run_id, limit)
    for item in observations:
        item["image_url"] = _image_url(item.get("image_path", ""))
    return {
        "observations": observations,
    }
