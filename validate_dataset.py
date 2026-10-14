
from pathlib import Path
import cv2

DATASET_DIR = Path(r"D:\ML_Project\dataset")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_images = 0
valid_images = 0

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir() or class_dir.name.startswith("."):
        continue

    image_files = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ]

    valid_count = 0
    invalid_files = []

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            invalid_files.append(image_path.name)
        else:
            valid_count += 1

    total_images += len(image_files)
    valid_images += valid_count

    print(
        f"{class_dir.name}: "
        f"{len(image_files)} files, "
        f"{valid_count} valid, "
        f"{len(invalid_files)} invalid"
    )

    if invalid_files:
        print("  Invalid files:", invalid_files)

print("\nTotal image files:", total_images)
print("Valid images:", valid_images)
print("Expected images: 500")
