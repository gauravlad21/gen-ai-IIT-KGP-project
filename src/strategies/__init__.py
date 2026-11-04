"""Beginner-friendly strategy classes grouped by concern."""

from src.strategies.answers import CriticAnswer, DatasetAnswer, LLMAnswer, VotingAnswer
from src.strategies.base import (
    AnswerStrategyBase,
    ChunkingStrategyBase,
    EmbedderBase,
    EvaluatorBase,
    MemoryStrategyBase,
    PromptStrategyBase,
    RAGStrategyBase,
    SearchStrategyBase,
)
from src.strategies.chunking import FixedSizeChunking, ParagraphChunking, WholeDocumentChunking
from src.strategies.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from src.strategies.evaluators import JudgeEvaluator, SimpleEvaluator, VotingEvaluator
from src.strategies.helpers import cosine_similarity
from src.strategies.memory import ConversationMemory, NoMemory
from src.strategies.prompts import DecompositionPrompt, FewShotPrompt, SimplePrompt, ZeroShotPrompt
from src.strategies.rag import AgenticRAG, AutoRAG, CombinedRAG, MultiStepRAG, SimpleRAG, VisualRAG
from src.strategies.search import ChromaSearch, KeywordSearch, LimitedSearch

__all__ = [
    "AnswerStrategyBase",
    "ChunkingStrategyBase",
    "EmbedderBase",
    "EvaluatorBase",
    "MemoryStrategyBase",
    "PromptStrategyBase",
    "RAGStrategyBase",
    "SearchStrategyBase",
    "CriticAnswer",
    "DatasetAnswer",
    "LLMAnswer",
    "VotingAnswer",
    "FixedSizeChunking",
    "ParagraphChunking",
    "WholeDocumentChunking",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "JudgeEvaluator",
    "SimpleEvaluator",
    "VotingEvaluator",
    "ConversationMemory",
    "NoMemory",
    "DecompositionPrompt",
    "FewShotPrompt",
    "SimplePrompt",
    "ZeroShotPrompt",
    "AgenticRAG",
    "AutoRAG",
    "CombinedRAG",
    "MultiStepRAG",
    "SimpleRAG",
    "VisualRAG",
    "ChromaSearch",
    "KeywordSearch",
    "LimitedSearch",
    "cosine_similarity",
]
