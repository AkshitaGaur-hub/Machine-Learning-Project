"""Metrics, confusion matrices, and open-set evaluation reporting."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)


def evaluate_classifier(name, model, X, y, classes, output_dir: Path, split_name: str) -> dict:
    pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=classes, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y, pred, labels=classes)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{name} - {split_name} confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name.lower().replace(' ', '_')}_{split_name}_confusion_matrix.png", dpi=140)
    plt.close(fig)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": classification_report(
            y, pred, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm.tolist()
    }


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def open_set_metrics(known_accept: np.ndarray, unknown_accept: np.ndarray) -> dict:
    return {
        "known_false_rejection_rate": float(np.mean(~known_accept)) if len(known_accept) else None,
        "unknown_false_acceptance_rate": float(np.mean(unknown_accept)) if len(unknown_accept) else None,
        "known_acceptance_rate": float(np.mean(known_accept)) if len(known_accept) else None,
        "unknown_rejection_rate": float(np.mean(~unknown_accept)) if len(unknown_accept) else None
    }
