from pathlib import Path
import cv2
import random


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "object_detection" / "dataset"

IMAGE_DIR = DATASET_ROOT / "images" / "train"
LABEL_DIR = DATASET_ROOT / "labels" / "train"

OUTPUT_DIR = PROJECT_ROOT / "object_detection" / "visualized"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Car",
    1: "Van",
    2: "Truck",
    3: "Pedestrian",
    4: "Person_sitting",
    5: "Cyclist",
    6: "Tram",
    7: "Misc"
}


# ============================================================
# FIND IMAGES
# ============================================================

images = list(IMAGE_DIR.glob("*.png"))

if not images:
    images = list(IMAGE_DIR.glob("*.jpg"))

if not images:
    images = list(IMAGE_DIR.glob("*.jpeg"))


if len(images) == 0:
    print("❌ No images found!")
    exit()


# ============================================================
# SELECT RANDOM IMAGES
# ============================================================

random.seed(42)

sample_images = random.sample(
    images,
    min(10, len(images))
)


print("=" * 60)
print("DATASET VISUALIZATION")
print("=" * 60)

print(f"\nTotal training images: {len(images)}")
print(f"Visualizing: {len(sample_images)} images")


# ============================================================
# PROCESS IMAGES
# ============================================================

for image_path in sample_images:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"❌ Could not read {image_path.name}")
        continue


    height, width = image.shape[:2]


    # --------------------------------------------------------
    # LABEL PATH
    # --------------------------------------------------------

    label_path = LABEL_DIR / f"{image_path.stem}.txt"


    if not label_path.exists():

        print(
            f"⚠ Label not found: {label_path.name}"
        )

        continue


    # --------------------------------------------------------
    # READ YOLO LABEL
    # --------------------------------------------------------

    with open(label_path, "r") as file:

        lines = file.readlines()


    object_count = 0


    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue


        class_id = int(parts[0])

        center_x = float(parts[1])
        center_y = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])


        # ----------------------------------------------------
        # CONVERT YOLO → PIXEL COORDINATES
        # ----------------------------------------------------

        center_x *= width
        center_y *= height

        box_width *= width
        box_height *= height


        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)

        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)


        # ----------------------------------------------------
        # KEEP INSIDE IMAGE
        # ----------------------------------------------------

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)


        # ----------------------------------------------------
        # CLASS NAME
        # ----------------------------------------------------

        class_name = CLASS_NAMES.get(
            class_id,
            f"Class_{class_id}"
        )


        # ----------------------------------------------------
        # DRAW BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # DRAW LABEL
        # ----------------------------------------------------

        text = f"{class_name} ({class_id})"

        cv2.putText(
            image,
            text,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        object_count += 1


    # --------------------------------------------------------
    # ADD INFORMATION
    # --------------------------------------------------------

    info = (
        f"{image_path.name} | "
        f"Objects: {object_count}"
    )

    cv2.putText(
        image,
        info,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"visual_{image_path.name}"
    )

    cv2.imwrite(
        str(output_path),
        image
    )


    print(
        f"✅ {image_path.name} "
        f"→ {object_count} objects"
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE")
print("=" * 60)

print("\nOutput folder:")

print(OUTPUT_DIR)

print("\nOpen this folder to see the annotated images.")

print("=" * 60)