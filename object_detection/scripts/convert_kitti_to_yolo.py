from pathlib import Path
import shutil
import random
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "object_detection"

IMAGE_SOURCE = (
    DATASET_ROOT
    / "data_object_image_2"
    / "training"
    / "image_2"
)

LABEL_SOURCE = (
    DATASET_ROOT
    / "data_object_label_2"
    / "training"
    / "label_2"
)

OUTPUT_ROOT = (
    DATASET_ROOT
    / "dataset"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

TRAIN_IMAGES = OUTPUT_ROOT / "images" / "train"
VAL_IMAGES = OUTPUT_ROOT / "images" / "val"

TRAIN_LABELS = OUTPUT_ROOT / "labels" / "train"
VAL_LABELS = OUTPUT_ROOT / "labels" / "val"


for folder in [
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASSES = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7
}


# ============================================================
# CHECK SOURCE DATA
# ============================================================

if not IMAGE_SOURCE.exists():
    raise FileNotFoundError(
        f"Image folder not found:\n{IMAGE_SOURCE}"
    )

if not LABEL_SOURCE.exists():
    raise FileNotFoundError(
        f"Label folder not found:\n{LABEL_SOURCE}"
    )


# ============================================================
# FIND IMAGES
# ============================================================

images = sorted(
    list(IMAGE_SOURCE.glob("*.png"))
    + list(IMAGE_SOURCE.glob("*.jpg"))
    + list(IMAGE_SOURCE.glob("*.jpeg"))
)

print("=" * 60)
print("KITTI → YOLO DATASET CONVERTER")
print("=" * 60)

print(f"\nImages found: {len(images)}")


if len(images) == 0:
    raise RuntimeError("No images found!")


# ============================================================
# SHUFFLE DATASET
# ============================================================

random.seed(42)

random.shuffle(images)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(len(images) * 0.8)

train_images = images[:train_size]
val_images = images[train_size:]


print(f"Training images:   {len(train_images)}")
print(f"Validation images: {len(val_images)}")


# ============================================================
# CONVERT KITTI LABEL TO YOLO
# ============================================================

def convert_label(
    kitti_label,
    yolo_label,
    image_width,
    image_height
):

    yolo_lines = []

    with open(kitti_label, "r") as file:

        lines = file.readlines()


    for line in lines:

        parts = line.strip().split()

        if len(parts) < 8:
            continue


        class_name = parts[0]


        # Ignore DontCare
        if class_name == "DontCare":
            continue


        # Ignore classes we don't know
        if class_name not in CLASSES:
            continue


        class_id = CLASSES[class_name]


        # ----------------------------------------------------
        # KITTI BOUNDING BOX
        # ----------------------------------------------------

        x1 = float(parts[4])
        y1 = float(parts[5])
        x2 = float(parts[6])
        y2 = float(parts[7])


        # ----------------------------------------------------
        # WIDTH / HEIGHT
        # ----------------------------------------------------

        box_width = x2 - x1
        box_height = y2 - y1


        if box_width <= 0 or box_height <= 0:
            continue


        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2


        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        center_x /= image_width
        center_y /= image_height

        box_width /= image_width
        box_height /= image_height


        # ----------------------------------------------------
        # YOLO FORMAT
        # ----------------------------------------------------

        line = (
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        yolo_lines.append(line)


    # --------------------------------------------------------
    # WRITE YOLO LABEL
    # --------------------------------------------------------

    with open(yolo_label, "w") as file:

        file.write("\n".join(yolo_lines))


# ============================================================
# PROCESS DATA
# ============================================================

def process_dataset(
    image_list,
    image_output,
    label_output
):

    processed = 0
    skipped = 0


    for image_path in image_list:

        label_path = (
            LABEL_SOURCE
            / f"{image_path.stem}.txt"
        )


        if not label_path.exists():

            print(
                f"WARNING: Label missing for "
                f"{image_path.name}"
            )

            skipped += 1

            continue


        # ----------------------------------------------------
        # IMAGE SIZE
        # ----------------------------------------------------

        with Image.open(image_path) as image:

            width, height = image.size


        # ----------------------------------------------------
        # COPY IMAGE
        # ----------------------------------------------------

        destination_image = (
            image_output
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination_image
        )


        # ----------------------------------------------------
        # CREATE YOLO LABEL
        # ----------------------------------------------------

        destination_label = (
            label_output
            / f"{image_path.stem}.txt"
        )


        convert_label(
            label_path,
            destination_label,
            width,
            height
        )


        processed += 1


        if processed % 500 == 0:

            print(
                f"Processed {processed} images..."
            )


    return processed, skipped


# ============================================================
# TRAINING DATA
# ============================================================

print("\nProcessing training dataset...")

train_processed, train_skipped = process_dataset(
    train_images,
    TRAIN_IMAGES,
    TRAIN_LABELS
)


# ============================================================
# VALIDATION DATA
# ============================================================

print("\nProcessing validation dataset...")

val_processed, val_skipped = process_dataset(
    val_images,
    VAL_IMAGES,
    VAL_LABELS
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(f"Training processed:   {train_processed}")
print(f"Training skipped:     {train_skipped}")

print(f"Validation processed: {val_processed}")
print(f"Validation skipped:   {val_skipped}")

print("\nYOLO dataset location:")

print(OUTPUT_ROOT)

print("\nDataset structure:")

print("""
dataset/
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/
""")

print("=" * 60)