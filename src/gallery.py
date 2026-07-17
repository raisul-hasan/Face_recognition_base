from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import l2_normalize, load_json, save_json

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GalleryEntry:
    identity: str
    embedding: np.ndarray
    source_images: list[str]
    count: int


@dataclass(slots=True)
class GalleryIndex:
    identities: list[str]
    embeddings: np.ndarray
    metadata: dict[str, Any]


def save_embedding(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, l2_normalize(embedding).astype(np.float32))


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    save_json(path, metadata)


def load_gallery_index(embeddings_dir: Path) -> GalleryIndex:
    identities: list[str] = []
    vectors: list[np.ndarray] = []
    metadata: dict[str, Any] = {}

    metadata_path = embeddings_dir / "gallery_index.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path, default={}) or {}

    for file_path in sorted(embeddings_dir.glob("*.npy")):
        if file_path.name.endswith(".npy"):
            identity = file_path.stem
            vector = np.load(file_path)
            identities.append(identity)
            vectors.append(l2_normalize(vector))

    if not vectors:
        raise FileNotFoundError(f"No gallery embeddings found in {embeddings_dir}")

    return GalleryIndex(
        identities=identities,
        embeddings=np.stack(vectors, axis=0).astype(np.float32),
        metadata=metadata,
    )
