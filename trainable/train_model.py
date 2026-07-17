"""Create ArcFace embeddings, train classifiers, evaluate them, and save the best model."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.training import (
    calibrate_threshold_and_evaluate,
    generate_embedding_dataset,
    save_best_model,
    train_and_select_model,
)
from src.utils import GALLERY_DIR, MODELS_DIR, REPORTS_DIR, TRAINING_DIR, configure_logging


def main() -> None:
    """Execute the reproducible train/evaluate/save workflow."""
    parser = argparse.ArgumentParser(description="Train the face-recognition classifier on ArcFace embeddings")
    parser.add_argument("--gallery-dir", type=Path, default=GALLERY_DIR)
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--test-size", type=float, default=load_config().training_test_size)
    parser.add_argument("--random-state", type=int, default=load_config().random_state)
    args = parser.parse_args()
    configure_logging()
    x_values, y_values = generate_embedding_dataset(args.gallery_dir, args.training_dir)
    classifier, scaler, encoder, name, accuracy = train_and_select_model(
        x_values, y_values, args.test_size, args.random_state, args.reports_dir
    )
    threshold = calibrate_threshold_and_evaluate(classifier, scaler, encoder, args.reports_dir)
    artifacts = save_best_model(classifier, scaler, encoder, name, threshold, accuracy, args.models_dir)
    logging.getLogger(__name__).info(
        "Saved %s (test accuracy %.3f, threshold %.2f) to %s", name, accuracy, threshold, artifacts.classifier_path
    )


if __name__ == "__main__":
    main()
