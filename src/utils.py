from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
GALLERY_DIR = ROOT_DIR / "gallery"
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
MODELS_DIR = ROOT_DIR / "models"
TRAINING_DIR = ROOT_DIR / "training"
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUTS_DIR = ROOT_DIR / "outputs"
TEST_IMAGES_DIR = ROOT_DIR / "test_images"
KNOWN_TEST_DIR = TEST_IMAGES_DIR / "known"
UNKNOWN_TEST_DIR = TEST_IMAGES_DIR / "unknown"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
# Keep InsightFace's downloaded model files inside the project.  This avoids
# relying on a writable user-profile directory and makes the runtime assets
# easy to locate when the project is moved between machines.
INSIGHTFACE_ROOT_DIR = ROOT_DIR / ".insightface"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple console logger once."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    slug = slug.strip("_")
    return slug or "identity"


def iter_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    suffix = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Unable to encode image for saving: {path}")
    encoded.tofile(str(path))


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(array)
    if norm <= 0:
        return array
    return array / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    first = l2_normalize(a)
    second = l2_normalize(b)
    if first.shape != second.shape:
        raise ValueError("Vectors must have matching shapes for cosine similarity")
    return float(np.dot(first, second))


def largest_face_index(bboxes: np.ndarray, det_scores: np.ndarray | None = None) -> int:
    if bboxes.size == 0:
        raise ValueError("No bounding boxes provided")
    areas = np.maximum(0.0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(0.0, bboxes[:, 3] - bboxes[:, 1])
    if det_scores is None:
        return int(np.argmax(areas))
    weighted = areas * np.asarray(det_scores, dtype=np.float32)
    return int(np.argmax(weighted))


@dataclass(slots=True)
class ImageRecord:
    path: Path
    identity: str
