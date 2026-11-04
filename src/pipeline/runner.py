"""One small injectable function that runs the copilot from input to response."""

from dataclasses import dataclass, field
from typing import Any, Callable

from src.models import Document, Evidence
from src.strategies import (
    AnswerStrategyBase,
    AutoRAG,
    ConversationMemory,
    DatasetAnswer,
    EvaluatorBase,
    MemoryStrategyBase,
    PromptStrategyBase,
    RAGStrategyBase,
    SimpleEvaluator,
    SimplePrompt,
)


@dataclass
class PipelineStrategies:
    """Swap one component at a time; the rest of the flow remains unchanged."""

    rag: RAGStrategyBase = field(default_factory=AutoRAG)
    answerer: AnswerStrategyBase = field(default_factory=DatasetAnswer)
    prompt: PromptStrategyBase = field(default_factory=SimplePrompt)
    memory: MemoryStrategyBase = field(default_factory=ConversationMemory)
    evaluator: EvaluatorBase = field(default_factory=SimpleEvaluator)
    request_validator: Callable[[str], None] | None = None
    response_validator: Callable[[dict, list[Evidence]], dict] | None = None
    router: Callable[[str, Any], str] | None = None


def validate_request(question: str) -> None:
    if not question or not question.strip():
        raise ValueError("Enter a question.")


def validate_response(answer: dict, evidence: list[Evidence]) -> dict:
    if not isinstance(answer, dict):
        raise ValueError("Answer strategy must return a dictionary.")
    if not isinstance(answer.get("answer"), str):
        raise ValueError("Answer strategy must return an answer string.")
    if not isinstance(answer.get("should_refuse"), bool):
        raise ValueError("Answer strategy must return a should_refuse boolean.")
    if not isinstance(answer.get("reason", ""), str):
        raise ValueError("Answer strategy refusal reason must be text.")
    valid_ids = {item.doc_id for item in evidence}
    citations = answer.get("citations", [])
    if not isinstance(citations, list) or any(
        not isinstance(item, dict) or item.get("doc_id") not in valid_ids for item in citations
    ):
        return {
            "answer": "I cannot verify the answer citation.",
            "citations": [],
            "should_refuse": True,
            "reason": "Citation validation failed.",
        }
    return answer


def route_request(question: str, image: Any = None) -> str:
    return "visual" if image is not None else "text"


def run_pipeline(
    question: str,
    documents: list[Document],
    session_id: str = "default",
    expected_answer: str | None = None,
    image: Any = None,
    strategies: PipelineStrategies | None = None,
) -> dict:
    """Run validate → route → memory → retrieve → answer → evaluate → validate."""
    choices = strategies or PipelineStrategies()
    validate = choices.request_validator or validate_request
    response_check = choices.response_validator or validate_response
    router = choices.router or route_request

    validate(question)
    route = router(question, image)
    history = choices.memory.get(session_id)
    evidence = choices.rag.retrieve(question, documents, image)
    answer = choices.answerer.answer(question, evidence, choices.prompt, history)
    metrics = choices.evaluator.evaluate(answer, evidence, expected_answer)
    answer = response_check(answer, evidence)
    choices.memory.add(session_id, question, answer["answer"])

    return {
        "answer": answer,
        "route": route,
        "evidence": evidence,
        "history": history,
        "metrics": metrics,
        "steps": [
            "validate request",
            f"route: {route}",
            "load memory",
            "search documents",
            "answer with prompt strategy",
            "evaluate",
            "validate response",
        ],
    }
