import importlib
import os
import tempfile
import unittest

from fastapi.testclient import TestClient


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        os.environ["SEARCH_STRATEGY"] = "keyword"
        os.environ["RAG_STRATEGY"] = "simple"
        os.environ["PROMPT_STRATEGY"] = "simple"
        os.environ["ANSWER_STRATEGY"] = "dataset"
        os.environ["MEMORY_STRATEGY"] = "conversation"
        os.environ["EVALUATOR_STRATEGY"] = "simple"
        os.environ["METRICS_DB_PATH"] = os.path.join(cls.temp_dir.name, "metrics.sqlite3")

        module = importlib.import_module("src.api.main")
        module = importlib.reload(module)
        cls.client = TestClient(module.app)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertGreaterEqual(payload["documents"], 1)
        self.assertEqual(self.client.get("/dashboard").status_code, 200)
        self.assertEqual(self.client.get("/query").status_code, 200)
        self.assertEqual(self.client.get("/pages").status_code, 200)

    def test_query_and_metrics(self):
        query = {
            "question": "What is the purpose of Recommendation No. 6?",
            "session_id": "api-test",
            "run_name": "test-run",
        }
        response = self.client.post("/api/query", json=query)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("answer", payload)
        self.assertIn("metrics", payload)
        self.assertTrue(payload["citations"])
        self.assertIn("retrieved", payload)
        self.assertIn("critiques", payload)

        metrics = self.client.get("/api/metrics?limit=10")
        self.assertEqual(metrics.status_code, 200)
        rows = metrics.json()["rows"]
        self.assertTrue(any(row["run_name"] == "test-run" for row in rows))
        traces = self.client.get("/api/traces?limit=10")
        self.assertEqual(traces.status_code, 200)
        self.assertTrue(any(row["run_name"] == "test-run" for row in traces.json()["rows"]))

    def test_rag_setup_and_evaluation(self):
        setup = {
            "setup_name": "api-setup",
            "dataset_name": "sujet-ai/Sujet-Finance-QA-Vision-100k",
            "sample_limit": 10,
            "vector_db": "chroma",
            "vector_path": "data/processed/chroma",
            "vector_collection": "finance_sample",
            "embedder_strategy": "hashing",
            "chunking_strategy": "whole",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "use_saved_sample": True,
        }
        setup_response = self.client.post("/api/rag/setup", json=setup)
        self.assertEqual(setup_response.status_code, 200)
        self.assertGreaterEqual(setup_response.json()["documents"], 1)

        evaluate_response = self.client.post(
            "/api/evaluate",
            json={"run_name": "eval-test", "sample_size": 5, "session_id": "eval-api"},
        )
        self.assertEqual(evaluate_response.status_code, 200)
        eval_payload = evaluate_response.json()
        self.assertIn("summary", eval_payload)
        self.assertEqual(eval_payload["summary"]["sample_size"], 5)
        self.assertIn("taxonomy_counts", eval_payload["summary"])
        self.assertEqual(len(eval_payload["observations"]), 5)
        self.assertIn("failure_category", eval_payload["observations"][0])
        self.assertIn("image_url", eval_payload["observations"][0])
        details = self.client.get(f"/api/evaluations/{eval_payload['eval_run_id']}?limit=5")
        self.assertEqual(details.status_code, 200)
        self.assertIn("image_url", details.json()["observations"][0])


if __name__ == "__main__":
    unittest.main()
