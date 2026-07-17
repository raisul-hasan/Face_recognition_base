from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_manager import LFWDatasetManager


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    manager = LFWDatasetManager()
    manifest = manager.prepare_splits()
    logging.info("Dataset prepared successfully")
    logging.info("Stats: %s", manifest.stats)


if __name__ == "__main__":
    main()
