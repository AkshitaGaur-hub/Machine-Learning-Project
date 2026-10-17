from pathlib import Path
import time
import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "results" / "models"
COMPARISON_CSV = BASE_DIR / "results" / "model_comparison.csv"

IMAGE_SIZE = (128, 128)  # (width, height), same as train_models.py
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SMOOTHING_FRAMES = 5


def resize_with_pad(gray_image, target_size):
    target_w, target_h = target_size
    h, w = gray_image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image received.")
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0
    )


def extract_hog_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = resize_with_pad(gray, IMAGE_SIZE)
    features = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
    )
    return features.astype(np.float32).reshape(1, -1)


def load_models():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Models folder not found: {MODEL_DIR}\n"
            "Run python train_models.py first."
        )

    model_paths = sorted(MODEL_DIR.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(
            f"No .joblib models found in {MODEL_DIR}. Run python train_models.py first."
        )

    # Order models by current run's held-out accuracy, if comparison file exists.
    order = {}
    if COMPARISON_CSV.exists():
        try:
            df = pd.read_csv(COMPARISON_CSV)
            if {"Model", "Accuracy (%)"}.issubset(df.columns):
                order = dict(zip(df["Model"], df["Accuracy (%)"]))
        except Exception as exc:
            print(f"Could not read comparison CSV: {exc}")

    loaded = []
    for path in model_paths:
        try:
            bundle = joblib.load(path)
            if not isinstance(bundle, dict) or "model" not in bundle:
                print(f"Skipping unsupported model bundle: {path.name}")
                continue
            if tuple(bundle.get("image_size", ())) != IMAGE_SIZE:
                print(f"Skipping {path.name}: image size doesn't match webcam settings.")
                continue
            model_name = bundle.get("model_name", path.stem.replace("_", " ").title())
            accuracy = order.get(model_name, None)
            loaded.append((model_name, bundle, accuracy, path.name))
        except Exception as exc:
            print(f"Could not load {path.name}: {exc}")

    loaded.sort(key=lambda item: (
        item[2] is None,
        -(float(item[2]) if item[2] is not None else 0.0),
        item[0]
    ))
    if not loaded:
        raise RuntimeError("No compatible trained models could be loaded.")
    return loaded


def main():
    models = load_models()
    print("Loaded models:")
    for name, _, accuracy, filename in models:
        suffix = f" — test accuracy {accuracy:.2f}%" if accuracy is not None else ""
        print(f"  {name}{suffix} ({filename})")

    # DirectShow works well on many Windows installations; fall back if unavailable.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {CAMERA_INDEX}. "
            "Close other camera apps or try CAMERA_INDEX = 1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    history = {name: [] for name, _, _, _ in models}
    previous_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: webcam frame could not be read.")
                break

            frame = cv2.flip(frame, 1)
            feature = extract_hog_from_frame(frame)

            lines = []
            for model_name, bundle, accuracy, _ in models:
                prediction = bundle["model"].predict(feature)[0]
                label = str(prediction)

                # Smooth brief flicker in predictions.
                history[model_name].append(label)
                if len(history[model_name]) > SMOOTHING_FRAMES:
                    history[model_name].pop(0)
                counts = {item: history[model_name].count(item) for item in set(history[model_name])}
                stable_label = max(counts, key=counts.get)

                suffix = f" ({accuracy:.1f}%)" if accuracy is not None else ""
                lines.append((model_name, stable_label, suffix))

            # Show camera preview and six model predictions.
            y0 = 35
            cv2.rectangle(frame, (10, 10), (min(frame.shape[1] - 10, 660), 40 + 30 * len(lines)), (0, 0, 0), -1)
            cv2.putText(frame, "HOG OBJECT RECOGNITION  |  Press Q to quit",
                        (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            for i, (model_name, label, suffix) in enumerate(lines):
                text = f"{model_name}: {label}{suffix}"
                cv2.putText(
                    frame, text, (20, y0 + 30 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 0), 2, cv2.LINE_AA
                )

            now = time.time()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Object Recognition - Six Classifiers", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()