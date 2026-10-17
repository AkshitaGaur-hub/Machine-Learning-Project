
from pathlib import Path
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Paths and settings
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"

IMAGE_SIZE = (128, 128)  # (width, height)
TEST_SIZE = 0.25
RANDOM_STATE = 42

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


# --------------------------------------------------
# Image preprocessing: must match webcam code
# --------------------------------------------------
def resize_with_pad(gray_image, target_size):
    target_w, target_h = target_size
    h, w = gray_image.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Empty image received.")

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        gray_image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=0
    )


def extract_hog_features(gray_image):
    resized = resize_with_pad(gray_image, IMAGE_SIZE)

    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True
    )

    return features.astype(np.float32)


# --------------------------------------------------
# Read dataset folders
# --------------------------------------------------
def load_dataset():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    class_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
        and folder.name.startswith("class_")
        and folder.name != ".ipynb_checkpoints"
    )

    if not class_folders:
        raise RuntimeError(
            f"No class folders found in {DATASET_DIR}. "
            "Expected folders such as class_00_perfume."
        )

    features = []
    labels = []
    class_counts = {}

    print("\nReading dataset images...")

    for folder in class_folders:
        # Example: class_00_perfume -> perfume
        class_name = folder.name.split("_", 2)[-1]

        image_files = sorted(
            path for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        loaded_count = 0

        for image_path in image_files:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            try:
                feature = extract_hog_features(image)
            except Exception as exc:
                print(f"Skipping {image_path}: {exc}")
                continue

            features.append(feature)
            labels.append(class_name)
            loaded_count += 1

        class_counts[class_name] = loaded_count
        print(f"  {class_name}: {loaded_count} images")

    if not features:
        raise RuntimeError("No readable dataset images were found.")

    # A stratified train/test split requires at least two images per class.
    too_small = {
        name: count for name, count in class_counts.items()
        if count < 2
    }

    if too_small:
        raise ValueError(
            "Each class needs at least 2 readable images. "
            f"Classes with too few images: {too_small}"
        )

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=str)
    class_names = sorted(set(labels))

    print(f"\nTotal images: {len(y)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"HOG feature vector size: {X.shape[1]}")

    return X, y, class_names


# --------------------------------------------------
# Define six classifiers
# --------------------------------------------------
def create_models(min_train_class_count):
    knn_neighbors = min(5, min_train_class_count)

    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            )
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "Naive Bayes": GaussianNB(),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=knn_neighbors
            )
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ),

        "SVC": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=10,
                probability=True,
                random_state=RANDOM_STATE
            )
        )
    }


# --------------------------------------------------
# Train and save models
# --------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, class_names = load_dataset()

    # Check that every class has enough images for stratification.
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        raise ValueError("Every class needs at least 2 images.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    _, train_counts = np.unique(y_train, return_counts=True)
    models = create_models(int(train_counts.min()))

    # Remove old model files so obsolete classifiers cannot be loaded.
    for old_file in MODEL_DIR.glob("*.joblib"):
        old_file.unlink()

    results = []

    print("\nTraining six classifiers...\n")

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions) * 100

        safe_name = model_name.lower().replace(" ", "_")
        model_path = MODEL_DIR / f"{safe_name}.joblib"

        # Store metadata for the webcam script.
        bundle = {
            "model": model,
            "model_name": model_name,
            "image_size": IMAGE_SIZE,
            "class_names": class_names,
            "feature_type": "HOG",
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "block_norm": "L2-Hys",
            "transform_sqrt": False
        }

        joblib.dump(bundle, model_path)

        results.append({
            "Model": model_name,
            "Accuracy (%)": round(accuracy, 2),
            "Model File": model_path.name
        })

        print(f"  Test accuracy: {accuracy:.2f}%")
        print(f"  Saved: {model_path.name}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=class_names,
                zero_division=0
            )
        )

    # Highest test accuracy first. This column is read by webcam_predict.py.
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        "Accuracy (%)",
        ascending=False
    ).reset_index(drop=True)

    comparison.to_csv(COMPARISON_CSV, index=False)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)
    print(comparison.to_string(index=False))
    print(f"\nModels saved in: {MODEL_DIR}")
    print(f"Comparison saved in: {COMPARISON_CSV}")
    print("\nNext, run: python webcam_predict.py")


if __name__ == "__main__":
    main()
