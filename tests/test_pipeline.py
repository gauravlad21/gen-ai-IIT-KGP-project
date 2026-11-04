import unittest

from src.config import AppConfig
from src.data import parse_qa_pairs
from src.models import Document, QAPair
from src.pipeline import PipelineStrategies, build_strategies_from_config, run_pipeline
from src.strategies import (
    CriticAnswer,
    FixedSizeChunking,
    KeywordSearch,
    LLMAnswer,
    SimpleRAG,
    ZeroShotPrompt,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(
                doc_id="invoice-1",
                content="Mallard Media Services invoice for billboard advertisements.",
                qa_pairs=[
                    QAPair("What company issued the invoice?", "Mallard Media Services")
                ],
            )
        ]
        self.strategies = PipelineStrategies(rag=SimpleRAG(KeywordSearch()))
        self.multi_qa_documents = [
            Document(
                doc_id="invoice-2",
                content="Invoice with multiple fields.",
                qa_pairs=[
                    QAPair("What company issued the invoice?", "Mallard Media Services"),
                    QAPair("What city is listed?", "New York"),
                    QAPair("What is invoice date?", "June 19, 1995"),
                ],
            )
        ]

    def test_pipeline_answers_with_dataset_citation(self):
        result = run_pipeline(
            "What company issued the invoice?",
            self.documents,
            strategies=self.strategies,
        )
        self.assertEqual(result["answer"]["answer"], "Mallard Media Services")
        self.assertEqual(result["answer"]["citations"], [{"doc_id": "invoice-1"}])
        self.assertEqual(result["route"], "text")
        self.assertEqual(len(result["steps"]), 7)

    def test_unknown_question_is_refused(self):
        result = run_pipeline(
            "What is the weather tomorrow?",
            self.documents,
            strategies=self.strategies,
        )
        self.assertTrue(result["answer"]["should_refuse"])
        self.assertEqual(result["answer"]["citations"], [])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            run_pipeline("  ", self.documents, strategies=self.strategies)

    def test_history_is_available_on_next_turn(self):
        first = run_pipeline(
            "What company issued the invoice?",
            self.documents,
            session_id="test",
            strategies=self.strategies,
        )
        second = run_pipeline(
            "What company issued the invoice?",
            self.documents,
            session_id="test",
            strategies=self.strategies,
        )
        self.assertEqual(first["history"], [])
        self.assertEqual(second["history"][0]["answer"], "Mallard Media Services")

    def test_dataset_qa_json_parsing(self):
        pairs = parse_qa_pairs('[{"question":"Q","answer":"A"}]')
        self.assertEqual(pairs, [QAPair("Q", "A")])

    def test_fixed_chunker_does_not_return_overlap_only_tail(self):
        self.assertEqual(
            FixedSizeChunking(size=4, overlap=1).split("abcdefghij"),
            ["abcd", "defg", "ghij"],
        )

    def test_injected_answerer_receives_selected_prompt(self):
        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            return {
                "answer": "Test answer",
                "citations": [{"doc_id": "invoice-1"}],
                "should_refuse": False,
                "reason": "",
            }

        choices = PipelineStrategies(
            rag=SimpleRAG(KeywordSearch()),
            answerer=CriticAnswer(LLMAnswer(fake_model)),
            prompt=ZeroShotPrompt(),
        )
        result = run_pipeline(
            "What company issued the invoice?",
            self.documents,
            strategies=choices,
        )
        self.assertIn("Use only evidence", prompts[0])
        self.assertEqual(result["answer"]["answer"], "Test answer")

    def test_build_strategies_from_config(self):
        config = AppConfig(
            search_strategy="keyword",
            rag_strategy="simple",
            prompt_strategy="decomposition",
            answer_strategy="critic",
            memory_strategy="none",
            evaluator_strategy="simple",
            chunking_strategy="whole",
            embedder_strategy="hashing",
            retrieval_limit=1,
        )
        strategies = build_strategies_from_config(config)
        result = run_pipeline(
            "What company issued the invoice?",
            self.multi_qa_documents,
            strategies=strategies,
        )
        self.assertEqual(len(result["evidence"]), 1)
        self.assertFalse(result["history"])


if __name__ == "__main__":
    unittest.main()
