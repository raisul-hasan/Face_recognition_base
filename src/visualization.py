from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .recognizer import RecognitionResult
from .config import load_config
from .utils import ensure_dir, write_image


def draw_results(image: np.ndarray, results: list[RecognitionResult], threshold: float | None = None) -> np.ndarray:
    """Draw professional labels, boxes, and the five SCRFD facial landmarks."""
    config = load_config()
    threshold = config.similarity_threshold if threshold is None else threshold
    annotated = image.copy()
    for result in results:
        bbox = result.bbox.astype(int)
        x1, y1, x2, y2 = bbox.tolist()
        is_unknown = result.identity in {"UNKNOWN", "NOT RECOGNIZED"}
        color = (0, 0, 255) if is_unknown else (0, 255, 0)
        thickness = config.bounding_box_thickness
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{result.identity} | Cls: {result.classifier_confidence:.2f} | Det: {result.det_score:.2f} | {result.recognition_time * 1000:.1f} ms"
        if is_unknown:
            label = f"NOT RECOGNIZED | Cls: {result.classifier_confidence:.2f} | Det: {result.det_score:.2f} | {result.recognition_time * 1000:.1f} ms"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = config.font_scale
        text_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        top = max(y1, text_height + 8)
        cv2.rectangle(annotated, (x1, top - text_height - baseline - 6), (x1 + text_width + 8, top), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, top - 4),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
        if result.kps is not None:
            for point in result.kps[:5].astype(int):
                cv2.circle(annotated, tuple(point.tolist()), config.landmark_radius, color, -1, cv2.LINE_AA)
    return annotated


def save_annotated(image: np.ndarray, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    write_image(output_path, image)
