from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectedFace, FaceDetector
from .gallery import GalleryIndex, load_gallery_index
from .utils import cosine_similarity, l2_normalize, read_image
from trainable.load_model import TrainedFaceClassifier

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RecognitionResult:
    bbox: np.ndarray
    kps: np.ndarray | None
    det_score: float
    identity: str
    similarity: float
    classifier_confidence: float
    embedding: np.ndarray
    recognition_time: float = 0.0


class FaceRecognizer:
    def __init__(self, embeddings_dir: Path, threshold: float | None = None, models_dir: Path | None = None) -> None:
        self.embeddings_dir = embeddings_dir
        self.threshold = threshold
        self.detector = FaceDetector()
        self.model: TrainedFaceClassifier | None = None
        try:
            self.model = TrainedFaceClassifier(models_dir, threshold=threshold) if models_dir else TrainedFaceClassifier(threshold=threshold)
        except FileNotFoundError:
            self.logger = logging.getLogger(f"{__name__}.FaceRecognizer")
            self.logger.warning("No trained classifier found; using legacy cosine gallery matching.")
        self.gallery = load_gallery_index(embeddings_dir) if self.model is None else None
        self.logger = logging.getLogger(f"{__name__}.FaceRecognizer")
        self.last_detection_time = 0.0
        self.last_recognition_time = 0.0
        self.last_processing_time = 0.0

    def recognize_embedding(self, embedding: np.ndarray) -> tuple[str, float]:
        if self.model is not None:
            prediction = self.model.predict_embedding(embedding)
            return prediction.identity, prediction.confidence
        assert self.gallery is not None
        candidate = l2_normalize(embedding)
        similarities = self.gallery.embeddings @ candidate
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        best_identity = self.gallery.identities[best_index]
        if best_similarity < (self.threshold or 0.35):
            return "NOT RECOGNIZED", best_similarity
        return best_identity, best_similarity

    def recognize_image(self, image: np.ndarray) -> tuple[np.ndarray, list[RecognitionResult]]:
        started = time.perf_counter()
        detection_started = time.perf_counter()
        faces = self.detector.detect(image)
        self.last_detection_time = time.perf_counter() - detection_started
        recognition_started = time.perf_counter()
        results: list[RecognitionResult] = []
        for face in faces:
            if face.embedding is None:
                continue
            identity, confidence = self.recognize_embedding(face.embedding)
            results.append(
                RecognitionResult(
                    bbox=face.bbox,
                    kps=face.kps,
                    det_score=face.det_score,
                    identity=identity,
                    similarity=confidence,
                    classifier_confidence=confidence,
                    embedding=face.embedding,
                )
            )
        self.last_recognition_time = time.perf_counter() - recognition_started
        self.last_processing_time = time.perf_counter() - started
        per_face_time = self.last_recognition_time / max(1, len(results))
        for result in results:
            result.recognition_time = per_face_time
        return image, results

    def recognize_path(self, path: Path) -> tuple[np.ndarray, list[RecognitionResult]]:
        image = read_image(path)
        return self.recognize_image(image)
