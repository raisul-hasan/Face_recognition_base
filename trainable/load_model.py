"""Load and use the saved scikit-learn face classifier for inference."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np

from src.config import load_config
from src.detector import FaceDetector
from src.utils import MODELS_DIR, l2_normalize, load_json, read_image


@dataclass(slots=True)
class ModelPrediction:
    """A classifier decision with calibrated confidence."""
    identity: str
    confidence: float
    accepted: bool


class TrainedFaceClassifier:
    """Thin reusable loader around the classifier, scaler, encoder, and threshold."""

    def __init__(self, models_dir: Path = MODELS_DIR, threshold: float | None = None) -> None:
        self.models_dir = models_dir
        try:
            self.classifier: Any = joblib.load(models_dir / "best_classifier.joblib")
            self.scaler: Any = joblib.load(models_dir / "scaler.joblib")
            self.label_encoder: Any = joblib.load(models_dir / "label_encoder.joblib")
        except FileNotFoundError as exc:
            raise FileNotFoundError("Trained model artifacts are missing. Run `python train_model.py` first.") from exc
        saved = load_json(models_dir / "threshold.json", default={}) or {}
        self.threshold = float(saved.get("confidence_threshold", load_config().classifier_confidence_threshold)) if threshold is None else threshold

    def predict_embedding(self, embedding: np.ndarray) -> ModelPrediction:
        """Predict an identity, rejecting low-probability classifications."""
        scaled = self.scaler.transform([l2_normalize(embedding)])
        probabilities = self.classifier.predict_proba(scaled)[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        identity = str(self.label_encoder.inverse_transform([index])[0])
        accepted = confidence >= self.threshold
        return ModelPrediction(identity if accepted else "NOT RECOGNIZED", confidence, accepted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained classifier inference on one image")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    model, detector = TrainedFaceClassifier(args.models_dir, args.threshold), FaceDetector()
    image = read_image(args.input)
    faces = detector.detect(image)
    if not faces:
        print("Not recognized (no face detected)")
        return
    face = detector.select_best_face(faces)
    if face.embedding is None:
        print("Not recognized (no embedding generated)")
        return
    prediction = model.predict_embedding(face.embedding)
    print(f"{prediction.identity} | confidence={prediction.confidence:.3f} | detection={face.det_score:.3f}")


if __name__ == "__main__":
    main()
