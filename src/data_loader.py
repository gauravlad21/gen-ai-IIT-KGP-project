"""Compatibility imports from the new folder-based layout."""

from src.data.loader import load_dataset_subset, load_saved_sample, parse_qa_pairs, save_sample
from src.models.records import Document, QAPair

__all__ = [
    "Document",
    "QAPair",
    "load_dataset_subset",
    "load_saved_sample",
    "parse_qa_pairs",
    "save_sample",
]
