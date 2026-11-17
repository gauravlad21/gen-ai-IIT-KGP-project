"""Run the prompting-strategy comparison and save every attempt."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.data import load_saved_sample
from src.experiments import STRATEGIES, run_attempt, summarize
from src.ollama import OllamaModel


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="An installed Ollama model name/tag")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--ids", nargs="+", help="Only run these question IDs (e.g. E01 U03)")
    parser.add_argument("--samples", type=int, default=5, help="Self-consistency samples per question")
    parser.add_argument("--temperature", type=float, default=0.7, help="Self-consistency sampling temperature")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--think", action="store_true", help="Enable the model's built-in thinking mode")
    parser.add_argument("--questions", default="data/eval/questions.json")
    parser.add_argument("--sample", default="data/sample/documents.json")
    parser.add_argument("--output-dir", default="data/processed/experiments")
    parser.add_argument("--resume", help="Continue an interrupted run directory, skipping finished attempts")
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")
    try:
        questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
        contents = {doc.doc_id: doc.content for doc in load_saved_sample(args.sample)}
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Input error: {exc}\n")
    if args.ids:
        unknown = set(args.ids) - {q["id"] for q in questions}
        if unknown:
            parser.error(f"Unknown question IDs: {sorted(unknown)}")
        questions = [q for q in questions if q["id"] in args.ids]

    def model_factory(temperature: float, seed: int) -> OllamaModel:
        return OllamaModel(
            args.model, timeout=args.timeout, temperature=temperature, seed=seed, think=args.think
        )

    records = []
    if args.resume:
        run_dir = Path(args.resume)
        try:
            saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            lines = (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines if line]
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Cannot resume: {exc}\n")
        expected = {
            "model": args.model, "think": args.think,
            "self_consistency_samples": args.samples, "self_consistency_temperature": args.temperature,
        }
        if any(saved.get(key) != value for key, value in expected.items()):
            parser.error("--resume needs the same --model, --think, --samples and --temperature as the original run")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(args.output_dir) / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        config = {
            "model": args.model,
            "think": args.think,
            "strategies": args.strategies,
            "question_ids": [q["id"] for q in questions],
            "self_consistency_samples": args.samples,
            "self_consistency_temperature": args.temperature,
            "single_call_temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "context_mode": "supplied_document_text",
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    done = {(r["strategy"], r["id"]) for r in records}
    with (run_dir / "attempts.jsonl").open("a", encoding="utf-8") as sink:
        for strategy in args.strategies:
            for item in questions:
                if (strategy, item["id"]) in done:
                    continue
                try:
                    record = run_attempt(
                        strategy, item, contents[item["doc_id"]], model_factory,
                        samples=args.samples, temperature=args.temperature,
                    )
                except RuntimeError as exc:
                    parser.exit(1, f"Ollama failure on {strategy}/{item['id']} (partial results kept in {run_dir}): {exc}\n")
                records.append(record)
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                print(f"{strategy:<20} {item['id']}  correct={record['correct']!s:<5} failure={record['failure']}", flush=True)

    summary = summarize(records)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved results to {run_dir}\n")
    header = f"{'strategy':<20}{'accuracy':>9}{'valid_fmt':>10}{'refusal_ok':>11}{'false_ref':>10}{'cite_ok':>8}{'calls':>7}{'sec/q':>7}"
    print(header)
    for strategy, row in summary.items():
        print(
            f"{strategy:<20}{_fmt(row['accuracy']):>9}{_fmt(row['valid_format_rate']):>10}"
            f"{_fmt(row['correct_refusal_rate']):>11}{_fmt(row['false_refusal_rate']):>10}"
            f"{_fmt(row['citation_ok_rate']):>8}{row['model_calls']:>7}{row['mean_latency_seconds']:>7.1f}"
        )


if __name__ == "__main__":
    main()
