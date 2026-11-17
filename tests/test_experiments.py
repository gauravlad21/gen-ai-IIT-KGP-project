import json
import unittest

from src.data import load_saved_sample
from src.experiments import STRATEGIES, build_prompt, part_found, run_attempt, score, summarize, vote

ITEM = {
    "id": "T1", "doc_id": "doc", "category": "extraction", "question": "What is the total?",
    "answer_parts": ["$1,000.00"], "should_refuse": False,
}
REFUSAL_ITEM = {**ITEM, "id": "T2", "category": "unanswerable", "answer_parts": [], "should_refuse": True}


def answer(text="$1000", refuse=False, doc="doc"):
    return {
        "answer": text,
        "citations": [] if refuse else [{"doc_id": doc}],
        "should_refuse": refuse,
        "reason": "missing" if refuse else "",
    }


class FakeModel:
    def __init__(self, outcome):
        self.outcome = outcome
        self.last_response = None

    def __call__(self, prompt):
        self.last_response = {"response": json.dumps(self.outcome) if isinstance(self.outcome, dict) else "bad"}
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class ExperimentTests(unittest.TestCase):
    def test_prompts_exist_for_every_strategy_and_exclude_labels(self):
        for strategy in STRATEGIES:
            prompt = build_prompt(strategy, "doc", "Total: $1,000.00", "What is the total?")
            self.assertIn("Document ID: doc", prompt)
            self.assertIn("Total: $1,000.00", prompt)
        self.assertIn("steps", build_prompt("decomposition", "doc", "x", "q"))
        self.assertNotIn("Example", build_prompt("baseline", "doc", "x", "q"))
        self.assertIn("Example", build_prompt("structured_fewshot", "doc", "x", "q"))

    def test_few_shot_examples_do_not_use_evaluation_documents(self):
        prompt = build_prompt("structured_fewshot", "doc", "", "q")
        for document in load_saved_sample("data/sample/documents.json"):
            self.assertNotIn(document.doc_id, prompt)
        self.assertNotIn("Total Budget", prompt)

    def test_numeric_matching_ignores_formatting(self):
        self.assertTrue(part_found("$1,000.00", "The total is $1000"))
        self.assertTrue(part_found("25%", "about 25.0 %"))
        self.assertFalse(part_found("$1,000.00", "The total is $100"))
        self.assertTrue(part_found("Haubens & Coleman, L.P.", "haubens & coleman l.p."))
        self.assertFalse(part_found("Carol Hrycaj", "Someone else"))

    def test_scoring_categories(self):
        self.assertEqual(score(ITEM, answer())["failure"], "none")
        self.assertEqual(score(ITEM, answer("$5"))["failure"], "wrong_answer")
        self.assertEqual(score(ITEM, answer(refuse=True))["failure"], "false_refusal")
        self.assertEqual(score(ITEM, answer(doc="other"))["failure"], "wrong_citation")
        self.assertEqual(score(ITEM, None)["failure"], "format_error")
        self.assertTrue(score(REFUSAL_ITEM, answer(refuse=True))["correct"])
        self.assertEqual(score(REFUSAL_ITEM, answer())["failure"], "fabricated_answer")
        multi = {**ITEM, "answer_parts": ["$1,000.00", "Acme"]}
        self.assertEqual(score(multi, answer("$1000"))["failure"], "incomplete_multi_part")

    def test_vote_picks_majority_and_breaks_ties_by_first_sample(self):
        samples = [{"answer": answer(t)} for t in ("$1", "$2", "$2")]
        self.assertEqual(vote(samples)[0]["answer"], "$2")
        tie = [{"answer": answer("$1")}, {"answer": answer("$2")}, {"answer": None}]
        self.assertEqual(vote(tie)[0]["answer"], "$1")
        self.assertEqual(vote([{"answer": None}]), (None, {}))

    def test_self_consistency_makes_n_calls_with_distinct_seeds(self):
        seeds = []

        def factory(temperature, seed):
            seeds.append((temperature, seed))
            return FakeModel(answer("$1000"))

        record = run_attempt("self_consistency", ITEM, "text", factory, samples=3, temperature=0.7)
        self.assertEqual(record["model_calls"], 3)
        self.assertEqual(len({seed for _, seed in seeds}), 3)
        self.assertTrue(all(temp == 0.7 for temp, _ in seeds))
        self.assertTrue(record["correct"])

    def test_invalid_output_is_a_format_error_but_ollama_failure_propagates(self):
        record = run_attempt("baseline", ITEM, "text", lambda t, s: FakeModel(ValueError("Model generated invalid JSON")))
        self.assertEqual(record["status"], "format_error")
        self.assertEqual(record["failure"], "format_error")
        with self.assertRaises(RuntimeError):
            run_attempt("baseline", ITEM, "text", lambda t, s: FakeModel(RuntimeError("Ollama down")))

    def test_summary_rates(self):
        good = run_attempt("baseline", ITEM, "text", lambda t, s: FakeModel(answer()))
        bad = run_attempt("baseline", REFUSAL_ITEM, "text", lambda t, s: FakeModel(answer()))
        row = summarize([good, bad])["baseline"]
        self.assertEqual(row["accuracy"], 0.5)
        self.assertEqual(row["correct_refusal_rate"], 0.0)
        self.assertEqual(row["failures"], {"fabricated_answer": 1})


if __name__ == "__main__":
    unittest.main()
