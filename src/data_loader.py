"""Load and save a small, explicit sample from the Hugging Face dataset."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class QAPair:
    question: str
    answer: str


@dataclass
class Document:
    doc_id: str
    content: str
    qa_pairs: list[QAPair]
    image_path: str | None = None


def parse_qa_pairs(value: Any) -> list[QAPair]:
    """The dataset stores qa_pairs as a JSON string."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("qa_pairs must be a list or a JSON-encoded list")
    return [
        QAPair(question=str(pair["question"]), answer=str(pair["answer"]))
        for pair in value
    ]


def load_dataset_subset(limit: int = 10, dataset_name: str = "sujet-ai/Sujet-Finance-QA-Vision-100k") -> list[dict]:
    """Stream only the requested number of records; never download the full dataset."""
    if limit < 1:
        raise ValueError("limit must be positive")
    iterator = None
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_url
        from huggingface_hub.utils import get_session
    except ImportError as exc:
        raise RuntimeError("Run `uv sync` to install the datasets library") from exc

    try:
        files = HfApi().list_repo_files(dataset_name, repo_type="dataset")
        shards = sorted(
            name
            for name in files
            if name.startswith("data/train-") and name.endswith(".parquet")
        )
        if not shards:
            raise ValueError(f"No train Parquet files found in {dataset_name}")
        urls = [
            hf_hub_url(dataset_name, name, repo_type="dataset")
            for name in shards
        ]
        dataset = load_dataset(
            "parquet",
            data_files={"train": urls},
            split="train",
            streaming=True,
        ).take(limit)
        iterator = iter(dataset)
        rows = []
        for _ in range(limit):
            try:
                rows.append(next(iterator))
            except StopIteration:
                break
        return rows
    finally:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if close:
                close()
        get_session().close()


def save_sample(rows: list[dict], output_dir: str = "data/sample") -> list[Document]:
    """Save a small dataset sample and its images for repeatable local runs."""
    root = Path(output_dir)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    documents = []

    for row in rows:
        doc_id = str(row["doc_id"])
        image_path = None
        image = row.get("image")
        if image is not None:
            image_name = Path(doc_id).name
            if not Path(image_name).suffix:
                image_name += ".jpg"
            destination = image_dir / image_name
            if hasattr(image, "save"):
                image.save(destination)
            else:
                destination.write_bytes(image)
            image_path = str(destination)

        documents.append(
            Document(
                doc_id=doc_id,
                content=str(row.get("content") or ""),
                qa_pairs=parse_qa_pairs(row.get("qa_pairs", [])),
                image_path=image_path,
            )
        )

    manifest = root / "documents.json"
    manifest.write_text(
        json.dumps([asdict(document) for document in documents], indent=2),
        encoding="utf-8",
    )
    return documents


def load_saved_sample(path: str = "data/sample/documents.json") -> list[Document]:
    """Load a sample already saved locally; no network access is needed."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Document(
            doc_id=item["doc_id"],
            content=item["content"],
            qa_pairs=[QAPair(**pair) for pair in item["qa_pairs"]],
            image_path=item.get("image_path"),
        )
        for item in records
    ]
