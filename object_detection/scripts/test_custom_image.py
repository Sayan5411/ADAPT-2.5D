from ultralytics import YOLO
from pathlib import Path
import cv2

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Lidar_Adaptive_2_5D")

MODEL_PATH = (
    PROJECT_ROOT
    / "object_detection"
    / "runs"
    / "lidar_object_detector-4"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "object_detection"
    / "predictions"
    / "custom"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL
# ============================================================

print("=" * 60)
print("CUSTOM IMAGE OBJECT DETECTION")
print("=" * 60)

if not MODEL_PATH.exists():
    print("ERROR: Model not found!")
    print(MODEL_PATH)
    exit()

print("\nModel:")
print(MODEL_PATH)

model = YOLO(str(MODEL_PATH))

print("\nModel classes:")

for class_id, class_name in model.names.items():
    print(f"  {class_id}: {class_name}")

# ============================================================
# IMAGE INPUT
# ============================================================

image_path = input(
    "\nEnter image path "
    "(press ENTER to use validation image): "
).strip()

if image_path == "":
    val_dir = PROJECT_ROOT / "object_detection" / "dataset" / "images" / "val"

    images = list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpg"))

    if not images:
        print("No validation images found.")
        exit()

    image_path = str(images[0])

image_path = Path(image_path)

if not image_path.exists():
    print("\nERROR: Image does not exist:")
    print(image_path)
    exit()

# ============================================================
# DETECTION
# ============================================================

print("\nInput:")
print(image_path)

print("\nRunning detection...\n")

results = model.predict(
    source=str(image_path),
    conf=0.30,
    iou=0.50,
    imgsz=640,
    device=0,
    save=False,
    verbose=True
)

# ============================================================
# PROCESS RESULT
# ============================================================

result = results[0]

image = cv2.imread(str(image_path))

detections = []

if result.boxes is not None:

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        class_name = model.names[class_id]

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": (x1, y1, x2, y2)
        })

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# ============================================================
# SAVE
# ============================================================

output_path = OUTPUT_DIR / f"prediction_{image_path.stem}.jpg"

cv2.imwrite(
    str(output_path),
    image
)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("DETECTION RESULTS")
print("=" * 60)

print(f"\nObjects detected: {len(detections)}")

for i, detection in enumerate(detections, 1):

    print(
        f"{i}. "
        f"{detection['class_name']:<18} "
        f"Confidence: "
        f"{detection['confidence']:.2f}"
    )

print("\nPrediction saved:")
print(output_path)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)