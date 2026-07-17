"""Training and evaluation helpers for the ArcFace embedding classifier."""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .detector import FaceDetector
from .utils import (
    GALLERY_DIR,
    KNOWN_TEST_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TRAINING_DIR,
    UNKNOWN_TEST_DIR,
    ensure_dir,
    iter_image_files,
    l2_normalize,
    load_json,
    save_json,
)

LOGGER = logging.getLogger(__name__)
THRESHOLD_CANDIDATES = (0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass(slots=True)
class TrainingArtifacts:
    classifier_name: str
    classifier_path: Path
    label_encoder_path: Path
    scaler_path: Path
    threshold_path: Path
    threshold: float
    test_accuracy: float


def generate_embedding_dataset(
    gallery_dir: Path = GALLERY_DIR,
    training_dir: Path = TRAINING_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one aligned ArcFace embedding for every usable gallery image."""
    ensure_dir(training_dir)
    detector = FaceDetector()
    cache_dir = ensure_dir(training_dir / "per_image_embeddings")
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    records: list[dict[str, str]] = []
    identity_map = load_json(gallery_dir / "identity_map.json", default={}) or {}

    for identity_dir in sorted(path for path in gallery_dir.iterdir() if path.is_dir()):
        for image_path in iter_image_files(identity_dir):
            try:
                cache_path = ensure_dir(cache_dir / identity_dir.name) / f"{image_path.stem}.npy"
                if cache_path.exists():
                    embedding = np.load(cache_path)
                else:
                    _, faces = detector.detect_from_path(image_path)
                    if not faces:
                        LOGGER.warning("No face detected: %s", image_path)
                        continue
                    face = detector.select_best_face(faces)
                    if face.embedding is None:
                        LOGGER.warning("No embedding produced: %s", image_path)
                        continue
                    embedding = l2_normalize(face.embedding)
                    # Checkpoint every sample so an interrupted CPU extraction resumes safely.
                    np.save(cache_path, embedding.astype(np.float32))
                embeddings.append(l2_normalize(embedding))
                labels.append(identity_dir.name)
                records.append({
                    "path": str(image_path),
                    "label": identity_dir.name,
                    "display_name": str(identity_map.get(identity_dir.name, identity_dir.name)),
                })
            except (OSError, ValueError, RuntimeError) as exc:
                LOGGER.warning("Skipping %s: %s", image_path, exc)

    if not embeddings:
        raise RuntimeError("No gallery embeddings were generated; check the gallery images and InsightFace setup.")
    x_values = np.stack(embeddings).astype(np.float32)
    y_values = np.asarray(labels, dtype=str)
    if len(set(y_values)) < 2:
        raise RuntimeError("At least two identities are required to train a classifier.")
    np.save(training_dir / "embeddings.npy", x_values)
    np.save(training_dir / "labels.npy", y_values)
    save_json(training_dir / "embedding_records.json", records)
    save_json(training_dir / "embedding_dataset_summary.json", {
        "samples": int(len(y_values)), "embedding_dimension": int(x_values.shape[1]),
        "identities": sorted(set(y_values.tolist())),
    })
    return x_values, y_values


def _plot_confusion(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    plt.figure(figsize=(max(7, len(labels) * .85), max(6, len(labels) * .75)))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title); plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for row, col in np.ndindex(matrix.shape):
        plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def _save_classifier_evaluation(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray,
    labels: list[str], report_dir: Path,
) -> dict[str, float]:
    """Persist metrics, a confusion matrix, classification report and OVR ROC plot."""
    model_dir = ensure_dir(report_dir / name.lower().replace(" ", "_"))
    precision, recall, f1_value, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision),
        "recall": float(recall), "f1_score": float(f1_value),
    }
    save_json(model_dir / "metrics.json", metrics)
    (model_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, labels=labels, zero_division=0), encoding="utf-8"
    )
    _plot_confusion(confusion_matrix(y_true, y_pred, labels=labels), labels,
                    f"{name} Confusion Matrix", model_dir / "confusion_matrix.png")
    if len(labels) > 1 and len(y_true) > 1:
        encoded = label_binarize(y_true, classes=labels)
        if encoded.ndim == 1 or encoded.shape[1] == 1:
            encoded = np.column_stack([1 - encoded.ravel(), encoded.ravel()])
        try:
            metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(encoded, probabilities, multi_class="ovr", average="weighted"))
            plt.figure(figsize=(7, 5))
            for index, label in enumerate(labels):
                if len(np.unique(encoded[:, index])) < 2:
                    continue
                fpr, tpr, _ = roc_curve(encoded[:, index], probabilities[:, index])
                plt.plot(fpr, tpr, label=label)
            plt.plot([0, 1], [0, 1], "k--", linewidth=1); plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate"); plt.title(f"{name} One-vs-Rest ROC")
            plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(model_dir / "roc_curve.png", dpi=160); plt.close()
            save_json(model_dir / "metrics.json", metrics)
        except ValueError as exc:
            LOGGER.warning("ROC unavailable for %s: %s", name, exc)
    return metrics


def train_and_select_model(
    x_values: np.ndarray, y_values: np.ndarray, test_size: float, random_state: int,
    report_dir: Path = REPORTS_DIR,
) -> tuple[ClassifierMixin, StandardScaler, LabelEncoder, str, float]:
    """Train the required classifiers and return the highest-accuracy candidate."""
    ensure_dir(report_dir)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_values)
    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    np.savez(TRAINING_DIR / "train_test_split.npz", x_train=x_train, x_test=x_test,
             y_train=y_train, y_test=y_test)
    scaler = StandardScaler().fit(x_train)
    x_train_scaled, x_test_scaled = scaler.transform(x_train), scaler.transform(x_test)
    neighbor_count = min(5, len(x_train))
    candidates: dict[str, ClassifierMixin] = {
        "Linear SVM": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=random_state),
        "RBF SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=neighbor_count, weights="distance"),
    }
    labels = list(range(len(encoder.classes_)))
    comparison: list[dict[str, Any]] = []
    best_name, best_model, best_accuracy = "", None, -1.0
    for name, model in candidates.items():
        model.fit(x_train_scaled, y_train)
        predicted = model.predict(x_test_scaled)
        probabilities = model.predict_proba(x_test_scaled)
        metrics = _save_classifier_evaluation(name, y_test, predicted, probabilities, labels, report_dir)
        row = {"classifier": name, **metrics}
        comparison.append(row)
        if metrics["accuracy"] > best_accuracy:
            best_name, best_model, best_accuracy = name, model, metrics["accuracy"]
    assert best_model is not None
    with (report_dir / "classifier_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in comparison for key in row}))
        writer.writeheader(); writer.writerows(comparison)
    save_json(report_dir / "classifier_comparison.json", comparison)
    plt.figure(figsize=(8, 4)); plt.bar([item["classifier"] for item in comparison], [item["accuracy"] for item in comparison])
    plt.ylim(0, 1); plt.ylabel("Test accuracy"); plt.title("Classifier comparison")
    plt.tight_layout(); plt.savefig(report_dir / "classifier_comparison.png", dpi=160); plt.close()
    return best_model, scaler, encoder, best_name, float(best_accuracy)


def _test_image_predictions(
    classifier: ClassifierMixin, scaler: StandardScaler, encoder: LabelEncoder,
    root: Path, is_known: bool, detector: FaceDetector,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    cache_root = ensure_dir(TRAINING_DIR / "test_embedding_cache" / ("known" if is_known else "unknown"))
    for path in iter_image_files(root):
        try:
            cache_path = ensure_dir(cache_root / path.parent.name) / f"{path.stem}.npy"
            if cache_path.exists():
                embedding = np.load(cache_path)
            else:
                _, faces = detector.detect_from_path(path)
                if not faces:
                    continue
                face = detector.select_best_face(faces)
                if face.embedding is None:
                    continue
                embedding = l2_normalize(face.embedding)
                np.save(cache_path, embedding.astype(np.float32))
            probabilities = classifier.predict_proba(scaler.transform([l2_normalize(embedding)]))[0]
            predicted = encoder.inverse_transform([int(np.argmax(probabilities))])[0]
            values.append({"path": str(path), "known": is_known, "actual": path.parent.name if is_known else "UNKNOWN",
                           "predicted": predicted, "confidence": float(np.max(probabilities))})
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.warning("Unable to evaluate %s: %s", path, exc)
    return values


def calibrate_threshold_and_evaluate(
    classifier: ClassifierMixin, scaler: StandardScaler, encoder: LabelEncoder,
    report_dir: Path = REPORTS_DIR,
) -> float:
    """Choose a threshold using separate known/unknown images and save open-set metrics."""
    detector = FaceDetector()
    records = _test_image_predictions(classifier, scaler, encoder, KNOWN_TEST_DIR, True, detector)
    records.extend(_test_image_predictions(classifier, scaler, encoder, UNKNOWN_TEST_DIR, False, detector))
    if not records:
        raise RuntimeError("No known/unknown test embeddings were available for threshold calibration.")
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLD_CANDIDATES:
        correct_known = sum(item["known"] and item["predicted"] == item["actual"] and item["confidence"] >= threshold for item in records)
        known_total = sum(item["known"] for item in records)
        unknown_total = sum(not item["known"] for item in records)
        false_accepts = sum((not item["known"]) and item["confidence"] >= threshold for item in records)
        false_rejects = sum(item["known"] and (item["confidence"] < threshold or item["predicted"] != item["actual"]) for item in records)
        y_true = ["KNOWN" if item["known"] else "UNKNOWN" for item in records]
        y_pred = ["KNOWN" if item["known"] and item["predicted"] == item["actual"] and item["confidence"] >= threshold else "UNKNOWN" for item in records]
        rows.append({"threshold": threshold, "recognition_accuracy": correct_known / max(1, known_total),
                     "false_acceptance_rate": false_accepts / max(1, unknown_total),
                     "false_rejection_rate": false_rejects / max(1, known_total),
                     "decision_f1": float(f1_score(y_true, y_pred, pos_label="KNOWN", zero_division=0))})
    best = max(rows, key=lambda row: (row["decision_f1"], row["recognition_accuracy"], -row["false_acceptance_rate"]))
    with (report_dir / "threshold_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plt.figure(figsize=(7, 4))
    for key in ("recognition_accuracy", "false_acceptance_rate", "false_rejection_rate", "decision_f1"):
        plt.plot([row["threshold"] for row in rows], [row[key] for row in rows], marker="o", label=key)
    plt.ylim(0, 1); plt.xlabel("Confidence threshold"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(report_dir / "classifier_threshold_analysis.png", dpi=160); plt.close()
    final_true = ["KNOWN" if item["known"] else "UNKNOWN" for item in records]
    final_pred = ["KNOWN" if item["known"] and item["predicted"] == item["actual"] and item["confidence"] >= best["threshold"] else "UNKNOWN" for item in records]
    matrix = confusion_matrix(final_true, final_pred, labels=["KNOWN", "UNKNOWN"])
    _plot_confusion(matrix, ["KNOWN", "UNKNOWN"], "Open-set Recognition Confusion Matrix", report_dir / "open_set_confusion_matrix.png")
    report = {"selected_threshold": best["threshold"], **best,
              "known_samples": sum(item["known"] for item in records), "unknown_samples": sum(not item["known"] for item in records),
              "classification_report": classification_report(final_true, final_pred, zero_division=0)}
    (report_dir / "open_set_classification_report.txt").write_text(report.pop("classification_report"), encoding="utf-8")
    save_json(report_dir / "open_set_evaluation.json", report)
    save_json(report_dir / "open_set_predictions.json", records)
    return float(best["threshold"])


def save_best_model(classifier: ClassifierMixin, scaler: StandardScaler, encoder: LabelEncoder,
                    name: str, threshold: float, accuracy: float, models_dir: Path = MODELS_DIR) -> TrainingArtifacts:
    """Save all inference artifacts independently with joblib and a manifest."""
    ensure_dir(models_dir)
    classifier_path, scaler_path = models_dir / "best_classifier.joblib", models_dir / "scaler.joblib"
    encoder_path, threshold_path = models_dir / "label_encoder.joblib", models_dir / "threshold.json"
    joblib.dump(classifier, classifier_path); joblib.dump(scaler, scaler_path); joblib.dump(encoder, encoder_path)
    save_json(threshold_path, {"confidence_threshold": threshold, "classifier": name, "test_accuracy": accuracy})
    return TrainingArtifacts(name, classifier_path, encoder_path, scaler_path, threshold_path, threshold, accuracy)
