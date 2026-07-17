from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .utils import INSIGHTFACE_ROOT_DIR, ensure_dir, largest_face_index, l2_normalize, read_image

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectedFace:
    bbox: np.ndarray
    kps: np.ndarray | None
    det_score: float
    embedding: np.ndarray | None


class FaceDetector:
    """SCRFD-backed face detector and embedding extractor.

    InsightFace's FaceAnalysis internally uses SCRFD for detection and ArcFace
    for recognition, which keeps the implementation pretrained-only and fully
    automatic.
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0) -> None:
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover - import-time environment guard
            raise RuntimeError(
                "insightface is required. Install dependencies from requirements.txt"
            ) from exc

        providers = ["CPUExecutionProvider"]
        model_root = ensure_dir(INSIGHTFACE_ROOT_DIR)
        self.app = FaceAnalysis(name=model_name, root=str(model_root), providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.logger = logging.getLogger(f"{__name__}.FaceDetector")

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        faces = self.app.get(image)
        detected_faces: list[DetectedFace] = []
        for face in faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is not None:
                embedding = l2_normalize(np.asarray(embedding, dtype=np.float32))
            bbox = np.asarray(face.bbox, dtype=np.float32)
            kps = None
            if getattr(face, "kps", None) is not None:
                kps = np.asarray(face.kps, dtype=np.float32)
            detected_faces.append(
                DetectedFace(
                    bbox=bbox,
                    kps=kps,
                    det_score=float(getattr(face, "det_score", 0.0)),
                    embedding=embedding,
                )
            )
        return detected_faces

    def detect_from_path(self, path: Path) -> tuple[np.ndarray, list[DetectedFace]]:
        image = read_image(path)
        return image, self.detect(image)

    @staticmethod
    def select_best_face(faces: list[DetectedFace]) -> DetectedFace:
        if not faces:
            raise ValueError("No detected faces available")
        bboxes = np.stack([face.bbox for face in faces], axis=0)
        det_scores = np.asarray([face.det_score for face in faces], dtype=np.float32)
        index = largest_face_index(bboxes, det_scores)
        return faces[index]
