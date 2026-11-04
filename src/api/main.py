"""FastAPI service exposing pipeline query and metrics dashboard APIs."""

from pathlib import Path
from time import perf_counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import AppConfig
from src.data import load_saved_sample
from src.db import MetricsStore
from src.pipeline import build_strategies_from_config, run_pipeline

from .schemas import MetricsCreateRequest, MetricsListResponse, QueryRequest, QueryResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup()
    yield


app = FastAPI(title="Financial Copilot API", version="0.1.0", lifespan=lifespan)

_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _ROOT / "frontend"

app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


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


@app.get("/")
def home():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.get("/dashboard")
def dashboard():
    return FileResponse(str(_FRONTEND_DIR / "dashboard.html"))


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
    return QueryResponse(
        answer=result["answer"],
        route=result["route"],
        metrics={**metrics, "latency_ms": round(latency_ms, 2)},
        steps=result["steps"],
        citations=result["answer"].get("citations", []),
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
