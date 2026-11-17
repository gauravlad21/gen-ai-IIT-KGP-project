import contextlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from scripts.ask_ollama import main
from src.models import Evidence
from src.ollama import OllamaModel
from src.strategies import LLMAnswer, ZeroShotPrompt


ANSWER = {
    "answer": "$100",
    "citations": [{"doc_id": "invoice"}],
    "should_refuse": False,
    "reason": "",
}


def response_for(answer=ANSWER, **extra):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps(
        {"done": True, "response": json.dumps(answer), **extra}
    ).encode()
    return response


class OllamaTests(unittest.TestCase):
    @patch("src.ollama.urlopen")
    def test_adapter_posts_one_local_request_and_retains_raw_response(self, urlopen):
        urlopen.return_value = response_for()
        model = OllamaModel("test:small")
        self.assertEqual(model("a prompt"), ANSWER)
        urlopen.assert_called_once()
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "test:small")
        self.assertEqual(payload["prompt"], "a prompt")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(model.last_prompt, "a prompt")
        self.assertEqual(model.last_response["response"], json.dumps(ANSWER))

    @patch("src.ollama.urlopen")
    def test_connection_failure_is_explicit(self, urlopen):
        urlopen.side_effect = URLError("connection refused")
        with self.assertRaisesRegex(RuntimeError, "ollama serve"):
            OllamaModel("test:small")("prompt")

    @patch("src.ollama.urlopen")
    def test_read_timeout_is_a_model_failure_not_an_outage(self, urlopen):
        urlopen.side_effect = TimeoutError("timed out")
        with self.assertRaisesRegex(ValueError, "did not finish"):
            OllamaModel("test:small", timeout=5)("prompt")

    @patch("src.ollama.urlopen")
    def test_http_failure_preserves_server_message(self, urlopen):
        urlopen.side_effect = HTTPError(
            "http://127.0.0.1:11434/api/generate",
            404, "Not Found", {}, BytesIO(b'{"error":"model not found"}'),
        )
        with self.assertRaisesRegex(RuntimeError, "model not found"):
            OllamaModel("test:small")("prompt")

    @patch("src.ollama.urlopen")
    def test_malformed_model_json_is_not_a_refusal(self, urlopen):
        urlopen.return_value = response_for(response="not JSON")
        model = OllamaModel("test:small")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            model("prompt")
        self.assertEqual(model.last_response["response"], "not JSON")

    @patch("src.ollama.urlopen")
    def test_invalid_answer_fields_are_rejected(self, urlopen):
        invalid_answers = [
            [],
            {**ANSWER, "should_refuse": "false"},
            {**ANSWER, "reason": None},
            {**ANSWER, "citations": []},
            {**ANSWER, "citations": [{}]},
            {**ANSWER, "citations": [{"doc_id": ""}]},
            {**ANSWER, "answer": ""},
            {**ANSWER, "should_refuse": True},
            {**ANSWER, "should_refuse": True, "citations": [], "reason": ""},
        ]
        for answer in invalid_answers:
            with self.subTest(answer=answer):
                urlopen.return_value = response_for(answer)
                with self.assertRaises(ValueError):
                    OllamaModel("test:small")("prompt")

    @patch("src.ollama.urlopen")
    def test_valid_refusal_is_accepted(self, urlopen):
        refusal = {
            "answer": "Cannot answer.", "citations": [],
            "should_refuse": True, "reason": "The bank account is not stated.",
        }
        urlopen.return_value = response_for(refusal)
        self.assertEqual(OllamaModel("test:small")("prompt"), refusal)

    @patch("src.ollama.urlopen")
    def test_incomplete_or_truncated_generation_is_rejected(self, urlopen):
        for extra in ({"done": False}, {"done_reason": "length"}):
            with self.subTest(extra=extra):
                urlopen.return_value = response_for(**extra)
                with self.assertRaises(ValueError):
                    OllamaModel("test:small")("prompt")

    def test_llm_prompt_receives_only_document_text_not_qa_labels(self):
        model = Mock(return_value=ANSWER)
        prompt = Mock(wraps=ZeroShotPrompt())
        LLMAnswer(model).answer(
            "What is the total?",
            [Evidence("invoice", "Total: $100", "PRIVATE_REFERENCE_QUESTION", "PRIVATE_LABEL")],
            prompt,
        )
        supplied = prompt.build.call_args.args[1][0]
        self.assertEqual(supplied.question, "")
        self.assertEqual(supplied.answer, "")
        request = model.call_args.args[0]
        self.assertIn("Document ID: invoice", request)
        self.assertIn("Total: $100", request)
        self.assertNotIn("PRIVATE_REFERENCE_QUESTION", request)
        self.assertNotIn("PRIVATE_LABEL", request)
        self.assertIn('"should_refuse"', request)

    def test_chroma_keeps_labels_out_of_returned_document_text(self):
        from src.models import Document, QAPair
        from src.strategies import ChromaSearch

        collection = Mock()
        collection.count.return_value = 1
        collection.query.return_value = {
            "documents": [["Total: $100"]],
            "metadatas": [[{
                "doc_id": "invoice", "question": "Reference question",
                "answer": "PRIVATE_LABEL", "image_path": "",
            }]],
            "distances": [[0.1]],
        }
        client = Mock()
        client.get_or_create_collection.return_value = collection
        module = Mock()
        module.PersistentClient.return_value = client
        with patch.dict(sys.modules, {"chromadb": module}):
            evidence = ChromaSearch().search(
                "What is the total?",
                [Document("invoice", "Total: $100", [QAPair("Reference question", "PRIVATE_LABEL")])],
            )
        self.assertEqual(collection.upsert.call_args.kwargs["documents"], ["Total: $100"])
        self.assertEqual(evidence[0].text, "Total: $100")
        self.assertEqual(evidence[0].answer, "PRIVATE_LABEL")


class ExperimentCommandTests(unittest.TestCase):
    @patch("src.ollama.urlopen")
    def test_success_and_failure_attempts_are_saved(self, urlopen):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.json"
            sample.write_text(json.dumps([{
                "doc_id": "invoice", "content": "Total: $100",
                "qa_pairs": [{"question": "Reference question", "answer": "PRIVATE_LABEL"}],
            }]), encoding="utf-8")
            for name, response in (
                ("success", response_for()),
                ("bad_json", response_for(response="invalid")),
                ("wrong_citation", response_for({
                    **ANSWER, "citations": [{"doc_id": "another-document"}],
                })),
                ("offline", URLError("connection refused")),
            ):
                with self.subTest(name=name):
                    urlopen.side_effect = response if isinstance(response, URLError) else None
                    urlopen.return_value = response
                    output_dir = Path(directory) / name
                    argv = [
                        "ask_ollama", "--model", "test:small", "--doc-id", "invoice",
                        "--question", "What is the total?", "--sample", str(sample),
                        "--output-dir", str(output_dir),
                    ]
                    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(StringIO()):
                        if name == "success":
                            main()
                        else:
                            with contextlib.redirect_stderr(StringIO()), self.assertRaises(SystemExit) as exc:
                                main()
                            self.assertEqual(exc.exception.code, 1)
                    files = list(output_dir.glob("*.json"))
                    self.assertEqual(len(files), 1)
                    record = json.loads(files[0].read_text())
                    self.assertEqual(record["status"], "completed" if name == "success" else "error")
                    self.assertNotIn("PRIVATE_LABEL", record["prompt"])
                    self.assertEqual(record["context_mode"], "supplied_document_text")
                    self.assertGreaterEqual(record["latency_seconds"], 0)
                    if name == "bad_json":
                        self.assertEqual(record["raw_response"]["response"], "invalid")
                    if name == "wrong_citation":
                        self.assertEqual(
                            record["model_answer"]["citations"], [{"doc_id": "another-document"}]
                        )

    @patch("src.ollama.urlopen")
    def test_unknown_document_is_rejected_before_model_call(self, urlopen):
        with patch.object(sys, "argv", [
            "ask_ollama", "--model", "test:small", "--doc-id", "missing", "--question", "Q?",
        ]), contextlib.redirect_stderr(StringIO()), self.assertRaises(SystemExit) as exc:
            main()
        self.assertEqual(exc.exception.code, 1)
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
