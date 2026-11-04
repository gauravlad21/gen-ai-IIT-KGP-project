"""Prompt construction strategies."""

from src.models import Evidence
from src.strategies.base import PromptStrategyBase


class SimplePrompt(PromptStrategyBase):
    def build(self, question: str, evidence: list[Evidence]) -> str:
        context = "\n".join(item.text for item in evidence)
        return f"Question: {question}\nEvidence:\n{context}"


class ZeroShotPrompt(SimplePrompt):
    def build(self, question: str, evidence: list[Evidence]) -> str:
        return (
            "You answer questions about financial documents. Use only evidence below. "
            "If evidence is insufficient, refuse and cite the source document.\n"
            + super().build(question, evidence)
        )


class FewShotPrompt(SimplePrompt):
    def __init__(self, examples: list[dict[str, str]] | None = None):
        self.examples = examples or []

    def build(self, question: str, evidence: list[Evidence]) -> str:
        examples = "\n".join(
            f"Example question: {item['question']}\nVerified answer: {item['answer']}"
            for item in self.examples
        )
        return examples + "\n" + super().build(question, evidence)


class DecompositionPrompt(SimplePrompt):
    def build(self, question: str, evidence: list[Evidence]) -> str:
        return (
            "Check each part of the question against the evidence before answering. "
            "Do not include hidden reasoning in the final answer.\n"
            + super().build(question, evidence)
        )
