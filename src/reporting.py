"""Reusable logging, face-crop, metric, and report helpers."""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from .recognizer import RecognitionResult
from .config import load_config
from .utils import EMBEDDINGS_DIR, GALLERY_DIR, ROOT_DIR, ensure_dir, iter_image_files, load_json, write_image

LOGS_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"


def output_directory() -> Path:
    """Return the configured project-relative output root."""
    return ROOT_DIR / load_config().output_directory


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def append_recognition_log(image_name: str, results: Iterable[RecognitionResult], processing_time: float) -> None:
    """Append one CSV row per recognized face."""
    if not load_config().logging_enabled:
        return
    path = ensure_dir(LOGS_DIR) / "recognition_log.csv"
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["timestamp", "image_name", "prediction", "classifier_confidence", "detection_confidence", "bbox", "processing_time_seconds"])
        for result in results:
            writer.writerow([timestamp(), image_name, result.identity, f"{result.similarity:.6f}", f"{result.det_score:.6f}", json.dumps(result.bbox.astype(float).tolist()), f"{processing_time:.6f}"])


def save_face_crops(image: np.ndarray, results: Iterable[RecognitionResult]) -> list[Path]:
    """Save clipped face crops with prediction and confidence in their filenames."""
    output_dir = ensure_dir(output_directory() / "cropped_faces")
    paths: list[Path] = []
    height, width = image.shape[:2]
    for result in results:
        x1, y1, x2, y2 = result.bbox.astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        safe_identity = "".join(char if char.isalnum() or char in "_-" else "_" for char in result.identity)
        path = output_dir / f"{safe_identity}_{timestamp()}_conf{result.det_score:.2f}.jpg"
        write_image(path, image[y1:y2, x1:x2])
        paths.append(path)
    return paths


def performance_summary(samples: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-image timing measurements into reportable metrics."""
    if not samples:
        return {key: 0.0 for key in ("average_detection_time", "average_recognition_time", "average_processing_time", "fps", "average_similarity")}
    processing = np.mean([item["processing_time"] for item in samples])
    return {
        "average_detection_time": float(np.mean([item["detection_time"] for item in samples])),
        "average_recognition_time": float(np.mean([item["recognition_time"] for item in samples])),
        "average_processing_time": float(processing),
        "fps": float(1.0 / processing) if processing > 0 else 0.0,
        "average_similarity": float(np.mean([item["similarity"] for item in samples])),
    }


def save_performance_report(metrics: dict[str, float]) -> Path:
    path = ensure_dir(REPORTS_DIR) / "performance_report.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows((key, f"{value:.6f}") for key, value in metrics.items())
    return path


def gallery_statistics() -> dict[str, Any]:
    metadata = load_json(EMBEDDINGS_DIR / "gallery_index.json", default={}) or {}
    embedding_files = sorted(EMBEDDINGS_DIR.glob("*.npy"))
    dimension = int(np.load(embedding_files[0]).size) if embedding_files else 0
    images_per_identity = {name: int(info.get("image_count", 0)) for name, info in metadata.items()}
    return {"number_of_identities": len(images_per_identity), "images_per_identity": images_per_identity, "embedding_dimension": dimension, "total_gallery_images": sum(images_per_identity.values())}


def save_gallery_report() -> Path:
    path = ROOT_DIR / "gallery_report.json"
    path.write_text(json.dumps(gallery_statistics(), indent=2), encoding="utf-8")
    return path


def classification_metrics(y_true: list[str], y_pred: list[str]) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    unknown_actual = np.asarray([value == "UNKNOWN" for value in y_true])
    unknown_predicted = np.asarray([value == "UNKNOWN" for value in y_pred])
    false_positive_rate = float(np.sum(unknown_actual & (~unknown_predicted)) / max(1, np.sum(unknown_actual)))
    false_negative_rate = float(np.sum((~unknown_actual) & unknown_predicted) / max(1, np.sum(~unknown_actual)))
    return ({"accuracy": correct / max(1, len(y_true)), "precision": float(precision), "recall": float(recall), "f1_score": float(f1), "false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate, "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0)}, matrix, labels)


def save_evaluation_report(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    metrics, matrix, labels = classification_metrics(y_true, y_pred)
    directory = ensure_dir(REPORTS_DIR)
    (directory / "classification_report.txt").write_text(metrics.pop("classification_report"), encoding="utf-8")
    (directory / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plt.figure(figsize=(max(7, len(labels)), max(6, len(labels))))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title("Recognition Confusion Matrix")
    for row, col in np.ndindex(matrix.shape):
        plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(directory / "confusion_matrix.png", dpi=160); plt.close()
    return metrics
