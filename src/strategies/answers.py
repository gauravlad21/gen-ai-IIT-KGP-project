"""Answer-generation strategies."""

from collections import Counter
from typing import Callable

from src.models import Evidence
from src.strategies.base import AnswerStrategyBase
from src.strategies.helpers import meaningful_words, refusal


class LLMAnswer(AnswerStrategyBase):
    def __init__(self, model: Callable[[str], dict]):
        self.model = model

    def answer(self, question: str, evidence: list[Evidence], prompt, history=None) -> dict:
        context = "\n".join(
            f"{item.doc_id}: {item.question} {item.answer} {item.text}" for item in evidence
        )
        request = prompt.build(question, evidence)
        if history:
            request = f"Conversation history: {history}\n{request}"
        return self.model(request + "\nRetrieved context:\n" + context)


class DatasetAnswer(AnswerStrategyBase):
    def answer(self, question: str, evidence: list[Evidence], prompt, history=None) -> dict:
        if not evidence:
            return refusal("No document evidence was found.")
        prompt.build(question, evidence)
        question_terms = meaningful_words(question)
        best = max(evidence, key=lambda item: len(question_terms & meaningful_words(item.question)))
        overlap = len(question_terms & meaningful_words(best.question))
        if not best.answer or overlap == 0:
            return refusal("The retrieved passages do not contain a matching dataset answer.")
        return {
            "answer": best.answer,
            "citations": [{"doc_id": best.doc_id}],
            "should_refuse": False,
            "reason": "",
        }


class CriticAnswer(AnswerStrategyBase):
    def __init__(self, answerer: AnswerStrategyBase | None = None):
        self.answerer = answerer or DatasetAnswer()

    def answer(self, question: str, evidence: list[Evidence], prompt, history=None) -> dict:
        result = self.answerer.answer(question, evidence, prompt, history)
        valid_ids = {item.doc_id for item in evidence}
        if any(citation.get("doc_id") not in valid_ids for citation in result["citations"]):
            return refusal("Answer citation did not match retrieved evidence.")
        return result


class VotingAnswer(AnswerStrategyBase):
    def __init__(self, answerers: list[AnswerStrategyBase]):
        if not answerers:
            raise ValueError("Provide at least one answer strategy.")
        self.answerers = answerers

    def answer(self, question: str, evidence: list[Evidence], prompt, history=None) -> dict:
        answers = [item.answer(question, evidence, prompt, history) for item in self.answerers]
        counts = Counter(item["answer"] for item in answers)
        winning_answer, votes = counts.most_common(1)[0]
        if votes <= len(self.answerers) // 2:
            return refusal("Answer strategies did not agree.")
        return next(item for item in answers if item["answer"] == winning_answer)
