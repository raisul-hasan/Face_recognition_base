from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recognizer import FaceRecognizer
from src.config import load_config
from src.reporting import append_recognition_log, performance_summary, save_face_crops, save_performance_report
from src.utils import EMBEDDINGS_DIR, OUTPUTS_DIR, TEST_IMAGES_DIR, ensure_dir, iter_image_files, read_image
from src.visualization import draw_results, save_annotated


LOGGER = logging.getLogger(__name__)


def recognize_path(input_path: Path, output_dir: Path, threshold: float) -> None:
    """Recognize all faces in one image or an image directory."""
    recognizer = FaceRecognizer(EMBEDDINGS_DIR, threshold=threshold)
    ensure_dir(output_dir)

    image_paths = [input_path] if input_path.is_file() else iter_image_files(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found at {input_path}")

    samples: list[dict[str, float]] = []
    for image_path in image_paths:
        image = read_image(image_path)
        _, results = recognizer.recognize_image(image)
        annotated = draw_results(image, results, threshold)
        output_path = output_dir / f"{image_path.stem}_recognized{image_path.suffix or '.png'}"
        save_annotated(annotated, output_path)
        save_face_crops(image, results)
        append_recognition_log(image_path.name, results, recognizer.last_processing_time)
        samples.append({"detection_time": recognizer.last_detection_time, "recognition_time": recognizer.last_recognition_time, "processing_time": recognizer.last_processing_time, "similarity": sum(result.similarity for result in results) / max(1, len(results))})
        LOGGER.info("Processed %s -> %s (%d faces)", image_path, output_path, len(results))
    metrics = performance_summary(samples)
    save_performance_report(metrics)
    LOGGER.info("Performance: %s", {key: round(value, 4) for key, value in metrics.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run face recognition on images")
    parser.add_argument("--input", type=Path, default=TEST_IMAGES_DIR, help="Image file or directory")
    parser.add_argument("--output-dir", type=Path, default=Path(load_config().output_directory))
    parser.add_argument("--threshold", type=float, default=None, help="Optional classifier confidence override")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    recognize_path(args.input, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
