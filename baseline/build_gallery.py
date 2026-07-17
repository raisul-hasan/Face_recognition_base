from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm import tqdm

from src.dataset_manager import LFWDatasetManager
from src.detector import FaceDetector
from src.gallery import save_embedding, save_metadata
from src.reporting import save_gallery_report
from src.utils import GALLERY_DIR, EMBEDDINGS_DIR, ensure_dir, iter_image_files, l2_normalize, load_json, save_json, slugify


LOGGER = logging.getLogger(__name__)


def build_gallery_embeddings(gallery_dir: Path, embeddings_dir: Path) -> dict[str, dict[str, object]]:
    ensure_dir(embeddings_dir)
    detector = FaceDetector()
    identity_map = load_json(gallery_dir / "identity_map.json", default={}) or {}
    metadata: dict[str, dict[str, object]] = {}

    for identity_folder in sorted([path for path in gallery_dir.iterdir() if path.is_dir()]):
        image_paths = iter_image_files(identity_folder)
        embeddings: list[np.ndarray] = []
        image_names: list[str] = []

        for image_path in tqdm(image_paths, desc=f"Building {identity_folder.name}", leave=False):
            image, faces = detector.detect_from_path(image_path)
            if not faces:
                LOGGER.warning("No face detected in gallery image: %s", image_path)
                continue
            best_face = detector.select_best_face(faces)
            if best_face.embedding is None:
                continue
            embeddings.append(best_face.embedding)
            image_names.append(image_path.name)

        if not embeddings:
            LOGGER.warning("Skipping %s because no valid embeddings were extracted", identity_folder.name)
            continue

        averaged = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))
        save_embedding(embeddings_dir / f"{identity_folder.name}.npy", averaged)
        metadata[identity_folder.name] = {
            "original_identity": identity_map.get(identity_folder.name, identity_folder.name),
            "image_count": len(image_names),
            "source_images": image_names,
        }

    save_metadata(embeddings_dir / "gallery_index.json", metadata)
    save_json(embeddings_dir / "gallery_summary.json", {"identities": list(metadata.keys()), "count": len(metadata)})
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build averaged gallery embeddings from gallery images")
    parser.add_argument("--gallery-dir", type=Path, default=GALLERY_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--prepare-dataset", action="store_true", help="Download and prepare the LFW split first")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.prepare_dataset:
        manager = LFWDatasetManager()
        manifest = manager.prepare_splits()
        LOGGER.info("Prepared dataset with stats: %s", manifest.stats)
    metadata = build_gallery_embeddings(args.gallery_dir, args.embeddings_dir)
    report_path = save_gallery_report()
    LOGGER.info("Built %d gallery embeddings", len(metadata))
    LOGGER.info("Saved gallery statistics to %s", report_path)


if __name__ == "__main__":
    main()
