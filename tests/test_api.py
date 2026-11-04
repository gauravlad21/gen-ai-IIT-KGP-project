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

        metrics = self.client.get("/api/metrics?limit=10")
        self.assertEqual(metrics.status_code, 200)
        rows = metrics.json()["rows"]
        self.assertTrue(any(row["run_name"] == "test-run" for row in rows))


if __name__ == "__main__":
    unittest.main()
