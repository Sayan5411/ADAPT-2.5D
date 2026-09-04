from ultralytics import YOLO
import os
import glob
import sys


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = r"D:\Lidar_Adaptive_2_5D"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "object_detection",
    "runs",
    "lidar_object_detector-4",
    "weights",
    "best.pt"
)

VAL_IMAGE_DIR = os.path.join(
    PROJECT_DIR,
    "object_detection",
    "dataset",
    "images",
    "val"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "object_detection",
    "predictions"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("LIDAR OBJECT DETECTION")
print("=" * 60)


# ============================================================
# CHECK MODEL
# ============================================================

print("\nChecking trained model...")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Trained model not found!")
    print(MODEL_PATH)
    sys.exit(1)

print("Model found:")
print(MODEL_PATH)


# ============================================================
# FIND VALIDATION IMAGES
# ============================================================

print("\nSearching validation images...")

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.webp"
]

images = []

for extension in extensions:
    images.extend(
        glob.glob(
            os.path.join(
                VAL_IMAGE_DIR,
                extension
            )
        )
    )

if len(images) == 0:
    print("\nERROR: No images found!")
    print("Directory:")
    print(VAL_IMAGE_DIR)
    sys.exit(1)

# Sort images
images.sort()

print(f"Found {len(images)} validation images.")

# Select first image
IMAGE_PATH = images[0]

print("\nSelected image:")
print(IMAGE_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

try:

    model = YOLO(MODEL_PATH)

except Exception as e:

    print("\nERROR loading model:")
    print(e)
    sys.exit(1)

print("Model loaded successfully!")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# RUN DETECTION
# ============================================================

print("\nRunning detection...")

try:

    results = model.predict(
        source=IMAGE_PATH,

        # Confidence threshold
        conf=0.25,

        # Image size
        imgsz=640,

        # NVIDIA GPU
        device=0,

        # Save annotated image
        save=True,

        # Output directory
        project=OUTPUT_DIR,

        name="test",

        # Show class names
        show_labels=True,

        # Show confidence
        show_conf=True
    )

except Exception as e:

    print("\nERROR during detection:")
    print(e)
    sys.exit(1)


# ============================================================
# DISPLAY DETECTIONS
# ============================================================

print("\n" + "=" * 60)
print("DETECTION RESULTS")
print("=" * 60)


total_objects = 0

for result in results:

    if result.boxes is None:
        continue

    number_of_objects = len(result.boxes)

    total_objects += number_of_objects

    print(
        f"\nObjects detected: {number_of_objects}"
    )

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        class_name = model.names[
            class_id
        ]

        print(
            f"  {class_name:<18} "
            f"Confidence: {confidence:.2f}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("DETECTION COMPLETED")
print("=" * 60)

print(
    f"\nTotal objects detected: {total_objects}"
)

print("\nInput image:")
print(IMAGE_PATH)

print("\nPrediction saved in:")
print(
    os.path.join(
        OUTPUT_DIR,
        "test"
    )
)

print("\n" + "=" * 60)