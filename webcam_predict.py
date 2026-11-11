"""Live webcam recognition using the exact trained MobileNetV2 + six-model bundle."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf

from preprocessing import IMAGE_SIZE, extract_embeddings
from ensemble import calibrated_probabilities, decide_one
from unknown_detection import nearest_centroid_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    models_dir = Path(args.models_dir)
    extractor_path = models_dir / "mobilenetv2_feature_extractor.keras"
    bundle_path = models_dir / "six_model_bundle.joblib"
    if not extractor_path.exists() or not bundle_path.exists():
        raise SystemExit("Models missing. Run `python train_models.py` first.")

    print("Loading feature extractor and six classifiers...")
    extractor = tf.keras.models.load_model(extractor_path, compile=False)
    bundle = joblib.load(bundle_path)
    models, classes = bundle["models"], bundle["classes"]
    weights = bundle["weights"]
    # Warm-up to avoid first-capture latency spike.
    extractor.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera permissions or --camera index.")
    print("SPACE = capture and predict | Q = quit")
    last_lines = ["Place one object inside the rectangle.", "Press SPACE to capture."]
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam frame could not be read.")
            break
        h, w = frame.shape[:2]
        # Fixed centred guide. The same centre-square crop is applied at prediction.
        side = int(min(h, w) * 0.68)
        x1, y1 = (w - side) // 2, (h - side) // 2
        x2, y2 = x1 + side, y1 + side
        display = frame.copy()
        cv2.rectangle(display, (x1, y1), (x2, y2), (60, 220, 90), 2)
        cv2.putText(display, "Place ONE object here | SPACE: capture | Q: quit",
                    (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        for i, line in enumerate(last_lines[-5:]):
            cv2.putText(display, line[:75], (18, h - 110 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 240, 255), 1, cv2.LINE_AA)
        cv2.imshow("Real-world object recognition", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key == 32:
            # Inference timer starts after the user has positioned the object.
            captured = frame[y1:y2, x1:x2]
            rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
            feature_start = time.perf_counter()
            embedding = extract_embeddings([rgb], extractor, batch_size=1)
            feature_time = time.perf_counter() - feature_start

            model_start = time.perf_counter()
            probabilities, predictions = {}, {}
            for name, model in models.items():
                p = calibrated_probabilities(model, embedding, classes)[0]
                probabilities[name] = {c: float(p[j]) for j, c in enumerate(classes)}
                predictions[name] = str(model.predict(embedding)[0])
            model_time = time.perf_counter() - model_start

            ensemble_start = time.perf_counter()
            distance = float(nearest_centroid_distance(embedding, bundle["centroids"])[0])
            result = decide_one(
                probabilities, predictions, weights, distance,
                bundle["distance_threshold"], bundle["confidence_threshold"],
                bundle.get("min_agreement", 4)
            )
            ensemble_time = time.perf_counter() - ensemble_start
            total = feature_time + model_time + ensemble_time

            print("\nINDIVIDUAL MODEL PREDICTIONS")
            for name in models:
                print(f"{name:22s}: {predictions[name]}")
            print("\nENSEMBLE ANALYSIS")
            print(f"Predicted class: {result['predicted_class']}")
            print(f"Model agreement: {result['agreement']}/6")
            print(f"Ensemble confidence: {result['ensemble_confidence']:.4f}")
            print(f"Nearest-centroid cosine distance: {distance:.4f} (limit {bundle['distance_threshold']:.4f})")
            print(f"Novelty check: {'PASS' if result['novelty_pass'] else 'FAIL'}")
            print(f"Feature extraction: {feature_time:.3f}s | Six models: {model_time:.3f}s | Decision: {ensemble_time:.3f}s")
            print(f"Total inference: {total:.3f}s")
            if result["accepted"]:
                final = f"OBJECT FOUND: {result['predicted_class'].upper()}"
            else:
                final = "OBJECT NOT FOUND"
            print(f"\nFINAL RESULT:\n{final}\n")
            last_lines = [
                f"Result: {final}",
                f"Agreement: {result['agreement']}/6  Confidence: {result['ensemble_confidence']:.2f}",
                f"Novelty: {'PASS' if result['novelty_pass'] else 'FAIL'}  Time: {total:.2f}s"
            ]
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
