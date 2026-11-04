"""Evaluation strategies."""

from collections import Counter
from typing import Callable

from src.models import Evidence
from src.strategies.base import EvaluatorBase


class SimpleEvaluator(EvaluatorBase):
    def evaluate(self, answer: dict, evidence: list[Evidence], expected_answer: str | None = None) -> dict:
        return {
            "retrieved_count": len(evidence),
            "has_citation": bool(answer.get("citations")),
            "refused": bool(answer.get("should_refuse")),
            "exact_match": (
                answer.get("answer", "").strip().casefold() == expected_answer.strip().casefold()
                if expected_answer is not None
                else None
            ),
        }


class JudgeEvaluator(EvaluatorBase):
    def __init__(self, judge: Callable[[dict, list[Evidence]], dict]):
        self.judge = judge

    def evaluate(self, answer: dict, evidence: list[Evidence], expected_answer: str | None = None) -> dict:
        return self.judge(answer, evidence)


class VotingEvaluator(EvaluatorBase):
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
