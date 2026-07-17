from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.build_gallery import build_gallery_embeddings
from baseline.recognize_image import recognize_path
from src.dataset_manager import LFWDatasetManager
from src.utils import EMBEDDINGS_DIR, GALLERY_DIR, OUTPUTS_DIR, TEST_IMAGES_DIR, ensure_dir, iter_image_files
from baseline.webcam import run_webcam


LOGGER = logging.getLogger(__name__)


def run_pipeline(threshold: float, webcam_frames: int | None) -> dict[str, object]:
    ensure_dir(OUTPUTS_DIR)

    manager = LFWDatasetManager()
    manifest = manager.prepare_splits()
    LOGGER.info("Dataset prepared: %s", manifest.stats)

    metadata = build_gallery_embeddings(GALLERY_DIR, EMBEDDINGS_DIR)
    LOGGER.info("Gallery embeddings built for %d identities", len(metadata))

    recognize_path(TEST_IMAGES_DIR / "known", OUTPUTS_DIR / "known", threshold)
    recognize_path(TEST_IMAGES_DIR / "unknown", OUTPUTS_DIR / "unknown", threshold)

    webcam_status = "skipped"
    if webcam_frames is not None:
        run_webcam(threshold=threshold, camera_index=0, width=1280, height=720, max_frames=webcam_frames)
        webcam_status = f"attempted ({webcam_frames} frames)"

    return {
        "dataset_stats": manifest.stats,
        "selected_identities": manifest.selected_identities,
        "unknown_identities": manifest.unknown_identities,
        "gallery_embeddings": len(metadata),
        "webcam_status": webcam_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete face recognition baseline pipeline")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--webcam-frames", type=int, default=None, help="Optional webcam frame limit for validation")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    summary = run_pipeline(args.threshold, args.webcam_frames)
    LOGGER.info("Pipeline summary: %s", summary)


if __name__ == "__main__":
    main()
