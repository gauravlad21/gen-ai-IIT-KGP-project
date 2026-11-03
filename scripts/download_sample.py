"""Download only 10 rows and images from the configured Hugging Face dataset."""

import sys
from pathlib import Path

from src.data_loader import load_dataset_subset, save_sample


if __name__ == "__main__":
    manifest = Path("data/sample/documents.json")
    if manifest.exists() and "--refresh" not in sys.argv:
        print("The 10-document sample is already saved in data/sample/. Use --refresh to reload it.")
    else:
        rows = load_dataset_subset(limit=10)
        documents = save_sample(rows, "data/sample")
        print(f"Saved {len(documents)} documents and their QA pairs to data/sample/")
