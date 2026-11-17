"""Run one supplied-document baseline attempt and persist its result."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from src.data import load_saved_sample
from src.models import Evidence
from src.ollama import OllamaModel
from src.pipeline.runner import validate_request, validate_response
from src.strategies import LLMAnswer, ZeroShotPrompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="An installed Ollama model name/tag")
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--sample", default="data/sample/documents.json")
    parser.add_argument("--output-dir", default="data/processed/prompt_runs")
    args = parser.parse_args()

    try:
        validate_request(args.question)
        documents = load_saved_sample(args.sample)
        document = next((item for item in documents if item.doc_id == args.doc_id), None)
        if document is None:
            raise ValueError(f"Unknown document ID: {args.doc_id}")
        if not document.content.strip():
            raise ValueError("The selected document has no source text.")
        model = OllamaModel(args.model)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Input error: {exc}\n")

    evidence = [Evidence(doc_id=document.doc_id, text=document.content)]
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "prompt_strategy": "zero_shot_v1",
        "context_mode": "supplied_document_text",
        "doc_id": document.doc_id,
        "question": args.question,
        "temperature": 0,
        "seed": 42,
        "num_ctx": 8192,
    }
    started = perf_counter()
    error = None
    try:
        answer = LLMAnswer(model).answer(args.question, evidence, ZeroShotPrompt())
        record["model_answer"] = answer
        validated = validate_response(answer, evidence)
        record["answer"] = validated
        if validated != answer:
            raise ValueError("Model cited a document outside the supplied evidence.")
        record["status"] = "completed"
    except (RuntimeError, ValueError) as exc:
        error = str(exc)
        record.update(status="error", error=error)
    record.update(
        latency_seconds=perf_counter() - started,
        prompt=model.last_prompt,
        raw_response=model.last_response,
    )
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = output_dir / f"{stamp}.json"
        with output.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
    except OSError as exc:
        parser.exit(1, f"Could not save attempt: {exc}\n")
    print(f"Saved attempt: {output}")
    if error is not None:
        parser.exit(1, f"Experiment failed: {error}\n")
    print(json.dumps(record["answer"], indent=2))


if __name__ == "__main__":
    main()
