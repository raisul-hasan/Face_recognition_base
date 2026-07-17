"""Recognize all test images and create annotated images plus summary metrics."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.recognizer import FaceRecognizer
from src.reporting import append_recognition_log, performance_summary, save_face_crops, save_performance_report
from src.utils import EMBEDDINGS_DIR, OUTPUTS_DIR, TEST_IMAGES_DIR, ensure_dir, iter_image_files, read_image
from src.visualization import draw_results, save_annotated


LOGGER = logging.getLogger(__name__)


def recognize_folder(input_dir: Path, output_dir: Path, threshold: float) -> dict[str, float]:
    """Process every image below ``input_dir`` and return performance metrics."""
    recognizer = FaceRecognizer(EMBEDDINGS_DIR, threshold=threshold)
    image_paths = iter_image_files(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    samples: list[dict[str, float]] = []
    for image_path in image_paths:
        image = read_image(image_path)
        _, results = recognizer.recognize_image(image)
        relative = image_path.relative_to(input_dir)
        output_path = ensure_dir(output_dir / relative.parent) / f"{image_path.stem}_recognized{image_path.suffix}"
        save_annotated(draw_results(image, results, threshold), output_path)
        save_face_crops(image, results)
        append_recognition_log(str(relative), results, recognizer.last_processing_time)
        samples.append({"detection_time": recognizer.last_detection_time, "recognition_time": recognizer.last_recognition_time, "processing_time": recognizer.last_processing_time, "similarity": sum(item.similarity for item in results) / max(1, len(results))})
    metrics = performance_summary(samples)
    save_performance_report(metrics)
    LOGGER.info("Processed %d images. Performance: %s", len(image_paths), {key: round(value, 4) for key, value in metrics.items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize every image in a folder")
    parser.add_argument("--input", type=Path, default=TEST_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path(load_config().output_directory) / "annotated")
    parser.add_argument("--threshold", type=float, default=None, help="Optional classifier confidence override")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    recognize_folder(args.input, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
