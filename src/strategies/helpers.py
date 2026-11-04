"""Shared helper functions for strategies."""

import re


def refusal(reason: str) -> dict:
    return {
        "answer": "I cannot answer from the available document evidence.",
        "citations": [],
        "should_refuse": True,
        "reason": reason,
    }


def meaningful_words(text: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "to",
        "was",
        "what",
        "who",
    }
    return set(re.findall(r"\w+", text.lower())) - stop_words


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
