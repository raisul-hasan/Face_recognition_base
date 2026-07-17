from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.recognizer import FaceRecognizer
from src.config import load_config
from src.reporting import append_recognition_log, save_face_crops
from src.utils import EMBEDDINGS_DIR
from src.visualization import draw_results


LOGGER = logging.getLogger(__name__)


def run_webcam(threshold: float, camera_index: int, width: int, height: int, max_frames: int | None = None) -> None:
    try:
        recognizer = FaceRecognizer(EMBEDDINGS_DIR, threshold=threshold)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        LOGGER.error("Unable to start face recognition: %s", exc)
        return

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        LOGGER.warning("Camera %s could not be opened. Exiting gracefully.", camera_index)
        return

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame_count = 0
    while True:
        success, frame = capture.read()
        if not success:
            LOGGER.warning("Failed to read frame from camera")
            break
        _, results = recognizer.recognize_image(frame)
        annotated = draw_results(frame, results, threshold)
        save_face_crops(frame, results)
        append_recognition_log(f"webcam_{frame_count:06d}", results, recognizer.last_processing_time)
        cv2.imshow("Face Recognition Baseline", annotated)
        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if max_frames is not None and frame_count >= max_frames:
            break

    capture.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam face recognition")
    parser.add_argument("--threshold", type=float, default=None, help="Optional classifier confidence override")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for validation")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    run_webcam(args.threshold, args.camera_index, args.width, args.height, args.max_frames)


if __name__ == "__main__":
    main()
