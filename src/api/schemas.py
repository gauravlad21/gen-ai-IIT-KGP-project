"""Request and response schemas for the FastAPI service."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = "api-default"
    run_name: str = "interactive"


class QueryResponse(BaseModel):
    answer: dict
    route: str
    metrics: dict
    steps: list[str]
    citations: list[dict]
    retrieved: list[dict]
    critiques: list[str]


class MetricsCreateRequest(BaseModel):
    run_name: str = "manual"
    question: str
    answer: str
    route: str = "text"
    retrieved_count: int = 0
    has_citation: bool = False
    refused: bool = False
    exact_match: float | None = None
    latency_ms: float = 0.0


class MetricsListResponse(BaseModel):
    rows: list[dict]


class RagSetupRequest(BaseModel):
    setup_name: str = "manual-setup"
    dataset_name: str = "sujet-ai/Sujet-Finance-QA-Vision-100k"
    sample_limit: int = Field(default=10, ge=1, le=200)
    vector_db: str = "chroma"
    vector_path: str = "data/processed/chroma"
    vector_collection: str = "finance_sample"
    embedder_strategy: str = "hashing"
    chunking_strategy: str = "whole"
    chunk_size: int = Field(default=800, ge=50, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)
    use_saved_sample: bool = False


class EvaluationRequest(BaseModel):
    run_name: str = "eval-run"
    sample_size: int = Field(default=100, ge=1, le=500)
    session_id: str = "eval-session"


class EvaluationResponse(BaseModel):
    eval_run_id: int
    summary: dict
    observations: list[dict]
