
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "results"
MODEL_DIR = OUTPUT_DIR / "models"

IMAGE_SIZE = (128, 128)
TEST_SIZE = 0.20
RANDOM_STATE = 42

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ---------------- LOAD IMAGES ----------------

X_features = []
y_labels = []
class_names = []

print("Loading images and extracting HOG features...\n")

class_dirs = sorted(
    folder for folder in DATASET_DIR.iterdir()
    if folder.is_dir() and folder.name.startswith("class_")
)

for class_dir in class_dirs:
    class_name = class_dir.name.replace("class_", "", 1)
    class_name = class_name.split("_", 1)[1] if "_" in class_name else class_name

    class_names.append(class_name)
    count = 0

    for image_path in sorted(class_dir.iterdir()):
        if image_path.suffix.lower() not in EXTENSIONS:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE)

        features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )

        X_features.append(features)
        y_labels.append(class_name)
        count += 1

    print(f"{class_name}: {count} images processed")

X = np.asarray(X_features, dtype=np.float32)
y = np.asarray(y_labels)

print("\nDataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Total images:", len(y))


# ---------------- TRAIN / TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


# ---------------- DEFINE CLASSIFIERS ----------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        )),
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB()),
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5)),
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "SVC": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf")),
    ]),
}


# ---------------- TRAIN AND EVALUATE ----------------

results = []
all_predictions = {}

for model_name, model in models.items():
    print(f"\n{'=' * 55}")
    print("Training:", model_name)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    all_predictions[model_name] = predictions

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1-score (%)": round(f1 * 100, 2),
    })

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    # Save each trained model and its class labels.
    joblib.dump(
        {
            "model": model,
            "class_names": sorted(np.unique(y).tolist()),
            "image_size": IMAGE_SIZE,
            "feature_type": "HOG",
        },
        MODEL_DIR / (
            model_name.lower().replace(" ", "_") + ".joblib"
        ),
    )


# ---------------- SAVE COMPARISON TABLE ----------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "Accuracy (%)", ascending=False
)

print("\n\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
)


# ---------------- ACCURACY GRAPH ----------------

plt.figure(figsize=(11, 6))
plt.bar(
    results_df["Model"],
    results_df["Accuracy (%)"],
)
plt.title("Accuracy Comparison of ML Classifiers")
plt.xlabel("Classification Algorithm")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "accuracy_comparison.png",
    dpi=200,
)
plt.show()


# ---------------- CONFUSION MATRICES ----------------

fig, axes = plt.subplots(2, 3, figsize=(19, 12))
axes = axes.flatten()

for ax, (model_name, predictions) in zip(
    axes, all_predictions.items()
):
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=sorted(np.unique(y).tolist()),
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(model_name)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrices.png",
    dpi=200,
)
plt.show()

print("\nCompleted successfully!")
print("Results folder:", OUTPUT_DIR)
print("Saved models folder:", MODEL_DIR)
