import json
import unittest
from collections import Counter
from decimal import Decimal

from src.data import load_saved_sample

EXPECTED_COUNTS = {"extraction": 10, "arithmetic": 8, "multi_part": 6, "unanswerable": 6}


def _normalize(text: str) -> str:
    return text.lower().replace(",", "")


class EvalSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("data/eval/questions.json", encoding="utf-8") as handle:
            cls.questions = json.load(handle)
        cls.documents = {doc.doc_id: doc.content for doc in load_saved_sample("data/sample/documents.json")}

    def test_category_counts_and_unique_ids(self):
        self.assertEqual(dict(Counter(q["category"] for q in self.questions)), EXPECTED_COUNTS)
        ids = [q["id"] for q in self.questions]
        self.assertEqual(len(ids), len(set(ids)))

    def test_documents_exist(self):
        for item in self.questions:
            self.assertIn(item["doc_id"], self.documents, item["id"])

    def test_refusal_fields_are_consistent(self):
        for item in self.questions:
            refuse = item["category"] == "unanswerable"
            self.assertEqual(item["should_refuse"], refuse, item["id"])
            self.assertEqual(bool(item["answer_parts"]), not refuse, item["id"])

    def test_extracted_references_appear_in_document(self):
        for item in self.questions:
            if item["category"] not in {"extraction", "multi_part"}:
                continue
            content = _normalize(self.documents[item["doc_id"]])
            for part in item["answer_parts"]:
                with self.subTest(item["id"], part=part):
                    self.assertIn(_normalize(part), content)

    def test_arithmetic_references_match_derivation(self):
        operators = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
        }
        for item in self.questions:
            if item["category"] != "arithmetic":
                continue
            left, op, right, *scale = item["derivation"].split()
            result = operators[op](Decimal(left), Decimal(right))
            if scale:  # "... * 100" percentage form
                result *= Decimal(scale[-1])
            reference = Decimal(item["answer_parts"][0].replace("$", "").replace(",", "").rstrip("%"))
            with self.subTest(item["id"]):
                self.assertEqual(result, reference)


if __name__ == "__main__":
    unittest.main()
