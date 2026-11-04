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

