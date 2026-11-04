"""Build pipeline strategies from config values."""

from typing import Callable

from src.config import AppConfig
from src.pipeline.runner import PipelineStrategies
from src.strategies import (
    AgenticRAG,
    AutoRAG,
    ChromaSearch,
    CombinedRAG,
    ConversationMemory,
    CriticAnswer,
    DatasetAnswer,
    DecompositionPrompt,
    FewShotPrompt,
    FixedSizeChunking,
    HashingEmbedder,
    KeywordSearch,
    LLMAnswer,
    MultiStepRAG,
    NoMemory,
    ParagraphChunking,
    SentenceTransformerEmbedder,
    SimpleEvaluator,
    SimplePrompt,
    SimpleRAG,
    VisualRAG,
    VotingAnswer,
    VotingEvaluator,
    WholeDocumentChunking,
    ZeroShotPrompt,
    LimitedSearch,
)


def _build_chunker(config: AppConfig):
    if config.chunking_strategy == "whole":
        return WholeDocumentChunking()
    if config.chunking_strategy == "fixed":
        return FixedSizeChunking(size=config.chunk_size, overlap=config.chunk_overlap)
    if config.chunking_strategy == "paragraph":
        return ParagraphChunking()
    raise ValueError(f"Unknown chunking strategy: {config.chunking_strategy}")


def _build_embedder(config: AppConfig):
    if config.embedder_strategy == "hashing":
        return HashingEmbedder()
    if config.embedder_strategy == "sentence_transformer":
        return SentenceTransformerEmbedder(config.embedding_model_name)
    raise ValueError(f"Unknown embedder strategy: {config.embedder_strategy}")


def _build_search(config: AppConfig):
    base = None
    if config.search_strategy == "keyword":
        base = KeywordSearch()
    elif config.search_strategy == "chroma":
        base = ChromaSearch(
            path=config.chroma_path,
            collection_name=config.chroma_collection,
            embedder=_build_embedder(config),
            chunker=_build_chunker(config),
        )
    if base is None:
        raise ValueError(f"Unknown search strategy: {config.search_strategy}")
    return LimitedSearch(base, config.retrieval_limit)


def _build_rag(config: AppConfig, search):
    if config.rag_strategy == "simple":
        return SimpleRAG(search)
    if config.rag_strategy == "multistep":
        return MultiStepRAG(search)
    if config.rag_strategy == "agentic":
        return AgenticRAG(search)
    if config.rag_strategy == "visual":
        return VisualRAG()
    if config.rag_strategy == "combined":
        return CombinedRAG(text=SimpleRAG(search), visual=VisualRAG())
    if config.rag_strategy == "auto":
        return AutoRAG(text=SimpleRAG(search), visual=VisualRAG())
    raise ValueError(f"Unknown RAG strategy: {config.rag_strategy}")


def _build_prompt(config: AppConfig):
    if config.prompt_strategy == "simple":
        return SimplePrompt()
    if config.prompt_strategy == "zero_shot":
        return ZeroShotPrompt()
    if config.prompt_strategy == "few_shot":
        return FewShotPrompt()
    if config.prompt_strategy == "decomposition":
        return DecompositionPrompt()
    raise ValueError(f"Unknown prompt strategy: {config.prompt_strategy}")


def _build_answer(config: AppConfig, model_callable: Callable[[str], dict] | None = None):
    if config.answer_strategy == "dataset":
        return DatasetAnswer()
    if config.answer_strategy == "critic":
        return CriticAnswer(DatasetAnswer())
    if config.answer_strategy == "voting":
        return VotingAnswer([DatasetAnswer(), CriticAnswer(DatasetAnswer())])
    if config.answer_strategy == "llm":
        if model_callable is None:
            raise ValueError("ANSWER_STRATEGY=llm requires a model callable.")
        return LLMAnswer(model_callable)
    raise ValueError(f"Unknown answer strategy: {config.answer_strategy}")


def _build_memory(config: AppConfig):
    if config.memory_strategy == "conversation":
        return ConversationMemory()
    if config.memory_strategy == "none":
        return NoMemory()
    raise ValueError(f"Unknown memory strategy: {config.memory_strategy}")


def _build_evaluator(config: AppConfig):
    if config.evaluator_strategy == "simple":
        return SimpleEvaluator()
    if config.evaluator_strategy == "voting":
        left = SimpleEvaluator()
        right = SimpleEvaluator()
        return VotingEvaluator(
            [
                lambda answer, evidence: left.evaluate(answer, evidence),
                lambda answer, evidence: right.evaluate(answer, evidence),
            ]
        )
    if config.evaluator_strategy == "judge":
        raise ValueError("EVALUATOR_STRATEGY=judge needs a custom judge callable.")
    raise ValueError(f"Unknown evaluator strategy: {config.evaluator_strategy}")


def build_strategies_from_config(
    config: AppConfig,
    model_callable: Callable[[str], dict] | None = None,
) -> PipelineStrategies:
    search = _build_search(config)
    return PipelineStrategies(
        rag=_build_rag(config, search),
        prompt=_build_prompt(config),
        answerer=_build_answer(config, model_callable=model_callable),
        memory=_build_memory(config),
        evaluator=_build_evaluator(config),
    )
