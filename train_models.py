"""Train MobileNetV2 embeddings + six traditional classifiers.

Run from project root:
    python train_models.py --dataset dataset --augmentation-versions 2

Original images are split BEFORE augmentation. Unknown images are never a known class.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

from preprocessing import read_rgb, list_images, make_augmentations, extract_embeddings
from model_evaluation import evaluate_classifier, save_json, open_set_metrics
from unknown_detection import class_centroids, nearest_centroid_distance, calibrate_distance_threshold
from ensemble import normalized_weights, calibrated_probabilities, decide_one

SEED = 42
CLASSIFIERS = ["Random Forest", "SVM", "Logistic Regression", "KNN", "Naive Bayes", "Decision Tree"]


def load_partition(paths, labels):
    images, good_labels, good_paths = [], [], []
    for path, label in zip(paths, labels):
        image = read_rgb(path)
        if image is not None:
            images.append(image)
            good_labels.append(label)
            good_paths.append(path)
    return images, np.asarray(good_labels), good_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--augmentation-versions", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    dataset = Path(args.dataset)
    models_dir, reports_dir = Path(args.models_dir), Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(p for p in dataset.iterdir() if p.is_dir() and p.name.lower() != "unknown")
    if len(class_dirs) != 10:
        raise SystemExit(f"Expected 10 known class folders; found {len(class_dirs)}: {[p.name for p in class_dirs]}")
    classes = [p.name for p in class_dirs]
    paths, labels = [], []
    for folder in class_dirs:
        found = list_images(folder)
        print(f"{folder.name}: {len(found)} images")
        paths.extend(found)
        labels.extend([folder.name] * len(found))
    if not paths:
        raise SystemExit("No known images found.")
    print(f"Total known images: {len(paths)}")

    # Split source/original images first: 70/15/15, stratified.
    p_train, p_temp, y_train, y_temp = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    p_val, p_test, y_val, y_test = train_test_split(
        p_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    print(f"Original split: train={len(p_train)}, validation={len(p_val)}, test={len(p_test)}")

    train_images, y_train_arr, p_train = load_partition(p_train, y_train)
    val_images, y_val_arr, _ = load_partition(p_val, y_val)
    test_images, y_test_arr, _ = load_partition(p_test, y_test)
    if len(set(y_train_arr)) != 10 or len(set(y_val_arr)) != 10 or len(set(y_test_arr)) != 10:
        raise SystemExit("At least one split is missing a class. Add more original images per class.")

    unknown_dir = dataset / "unknown"
    unknown_paths = list_images(unknown_dir)
    print(f"Unknown reference images: {len(unknown_paths)} (not a known class)")
    # Unknowns are split separately and never included in classifier fitting.
    if len(unknown_paths) >= 4:
        u_val_paths, u_test_paths = train_test_split(unknown_paths, test_size=0.5, random_state=SEED)
    else:
        u_val_paths, u_test_paths = unknown_paths, []

    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    outputs = base(inputs, training=False)
    extractor = tf.keras.Model(inputs, outputs, name="mobilenetv2_feature_extractor")
    # Save architecture/weights so webcam uses the same exact feature extractor.
    extractor.save(models_dir / "mobilenetv2_feature_extractor.keras")
    extractor.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)

    rng = np.random.default_rng(SEED)
    augmented_images = list(train_images)
    augmented_labels = list(y_train_arr)
    for image, label in zip(train_images, y_train_arr):
        for aug in make_augmentations(image, args.augmentation_versions, rng):
            augmented_images.append(aug)
            augmented_labels.append(label)
    # Concatenation order and labels are appended together, preserving alignment.
    print(f"Training images after augmentation: {len(augmented_images)}")
    print("Extracting training embeddings...")
    X_train = extract_embeddings(augmented_images, extractor, args.batch_size)
    y_train_aug = np.asarray(augmented_labels)
    print("Extracting validation/test embeddings...")
    X_val = extract_embeddings(val_images, extractor, args.batch_size)
    X_test = extract_embeddings(test_images, extractor, args.batch_size)
    X_u_val = extract_embeddings([im for p in u_val_paths if (im := read_rgb(p)) is not None], extractor, args.batch_size)
    X_u_test = extract_embeddings([im for p in u_test_paths if (im := read_rgb(p)) is not None], extractor, args.batch_size)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1),
        "SVM": make_pipeline(StandardScaler(), SVC(C=5.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced", random_state=SEED)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_train) // 10)), weights="distance", metric="minkowski")),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=2, random_state=SEED)
    }
    bundle_models, reports = {}, {"classes": classes, "seed": SEED, "split": {"train_original": len(p_train), "validation": len(y_val_arr), "test": len(y_test_arr)}, "unknown_reference_count": len(unknown_paths), "models": {}}
    validation_scores = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train_aug)
        # Calibrate probability scores on validation set where possible.
        # SVC already exposes probabilities; sklearn pipeline models expose predict_proba.
        bundle_models[name] = model
        val_report = evaluate_classifier(name, model, X_val, y_val_arr, classes, reports_dir, "validation")
        test_report = evaluate_classifier(name, model, X_test, y_test_arr, classes, reports_dir, "test")
        validation_scores[name] = val_report["balanced_accuracy"]
        reports["models"][name] = {"validation": val_report, "test": test_report}
        print(f"{name}: val acc={val_report['accuracy']:.4f}, test acc={test_report['accuracy']:.4f}, test F1={test_report['f1_weighted']:.4f}")

    weights = normalized_weights(validation_scores)
    centroids = class_centroids(X_train, y_train_aug, classes)
    known_val_dist = nearest_centroid_distance(X_val, centroids)
    unknown_val_dist = nearest_centroid_distance(X_u_val, centroids) if len(X_u_val) else None
    distance_threshold = calibrate_distance_threshold(known_val_dist, unknown_val_dist)
    # Calibrate ensemble confidence threshold from correctly predicted known validation examples,
    # with a conservative quantile. Unknown validation scores also inform threshold if available.
    def ensemble_rows(X):
        probs_by_model, preds_by_model = {}, {}
        for n, m in bundle_models.items():
            p = calibrated_probabilities(m, X, classes)
            probs_by_model[n] = p
            preds_by_model[n] = m.predict(X)
        rows = []
        for i in range(len(X)):
            probs = {n: {c: float(probs_by_model[n][i, j]) for j, c in enumerate(classes)} for n in bundle_models}
            preds = {n: str(preds_by_model[n][i]) for n in bundle_models}
            rows.append(decide_one(probs, preds, weights, 0.0, 1.0, 0.0, min_agreement=1))
        return rows
    val_rows = ensemble_rows(X_val)
    known_conf = np.array([r["ensemble_confidence"] for r in val_rows])
    confidence_threshold = float(np.quantile(known_conf, 0.10)) if len(known_conf) else 0.5
    if len(X_u_val):
        u_rows = ensemble_rows(X_u_val)
        # choose threshold to improve balanced known/unknown validation separation,
        # while retaining at least 90% of known validation samples.
        candidates = np.unique(np.concatenate([known_conf, [r["ensemble_confidence"] for r in u_rows]]))
        best_t, best_score = confidence_threshold, -1
        for t in candidates:
            ka = np.mean(known_conf >= t)
            ur = np.mean([r["ensemble_confidence"] < t for r in u_rows])
            if ka >= 0.90 and 0.5 * (ka + ur) > best_score:
                best_t, best_score = float(t), float(0.5 * (ka + ur))
        confidence_threshold = best_t

    # Evaluate ensemble on known test set and unknown test set, using frozen validation thresholds.
    test_rows = ensemble_rows(X_test)
    test_dist = nearest_centroid_distance(X_test, centroids)
    accepted_known, ens_pred = [], []
    for row, dist in zip(test_rows, test_dist):
        final = decide_one(
            {n: {c: row["scores"][c] for c in classes} for n in []}, {}, weights,
            float(dist), distance_threshold, confidence_threshold
        ) if False else None
        accepted = row["ensemble_confidence"] >= confidence_threshold and row["agreement"] >= 4 and dist <= distance_threshold
        accepted_known.append(accepted)
        ens_pred.append(row["predicted_class"] if accepted else "OBJECT NOT FOUND")
    known_accuracy = float(np.mean(np.asarray(ens_pred) == y_test_arr))
    unknown_accept = []
    if len(X_u_test):
        u_rows_test = ensemble_rows(X_u_test)
        u_dist_test = nearest_centroid_distance(X_u_test, centroids)
        unknown_accept = [
            r["ensemble_confidence"] >= confidence_threshold and r["agreement"] >= 4 and d <= distance_threshold
            for r, d in zip(u_rows_test, u_dist_test)
        ]
    reports["ensemble"] = {
        "validation_weights": weights,
        "validation_balanced_accuracies": validation_scores,
        "confidence_threshold": confidence_threshold,
        "distance_threshold": distance_threshold,
        "minimum_agreement": 4,
        "known_test_accuracy_including_rejections": known_accuracy,
        "open_set_test_metrics": open_set_metrics(np.asarray(accepted_known), np.asarray(unknown_accept)),
        "note": "Thresholds/weights calibrated only on validation data. Test set used only for final evaluation."
    }
    bundle = {
        "models": bundle_models, "classes": classes, "weights": weights,
        "confidence_threshold": confidence_threshold,
        "distance_threshold": distance_threshold, "min_agreement": 4,
        "centroids": centroids, "feature_dim": int(X_train.shape[1]),
        "preprocessing": "RGB 224x224, MobileNetV2 preprocess_input, pooled embeddings, L2 normalized"
    }
    joblib.dump(bundle, models_dir / "six_model_bundle.joblib", compress=3)
    save_json(reports, models_dir / "training_report.json")
    save_json(reports, reports_dir / "training_report.json")
    print("\nENSEMBLE")
    print(f"Validation weights: {weights}")
    print(f"Confidence threshold: {confidence_threshold:.4f}")
    print(f"Cosine-distance threshold: {distance_threshold:.4f}")
    print(f"Known test accuracy incl. rejections: {known_accuracy:.4f}")
    if len(X_u_test):
        print(f"Unknown rejection rate: {reports['ensemble']['open_set_test_metrics']['unknown_rejection_rate']:.4f}")
        print(f"Unknown false acceptance rate: {reports['ensemble']['open_set_test_metrics']['unknown_false_acceptance_rate']:.4f}")
    print(f"Saved: {models_dir / 'mobilenetv2_feature_extractor.keras'}")
    print(f"Saved: {models_dir / 'six_model_bundle.joblib'}")
    print(f"Saved: {models_dir / 'training_report.json'}")
    print(f"Total training time: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
