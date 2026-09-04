import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import time

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

CONFIDENCE = 0.45
CAMERA_INDEX = 0

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
# CHECK MODEL
# ============================================================

print("=" * 60)
print("LIVE LIDAR OBJECT DETECTION")
print("=" * 60)

if not MODEL_PATH.exists():
    print("\nERROR: Model not found!")
    print(MODEL_PATH)
    exit()

print("\nModel found:")
print(MODEL_PATH)

# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    print("\nUsing NVIDIA GPU")
    print(torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("\nCUDA not available - using CPU")

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")

# ============================================================
# OPEN CAMERA
# ============================================================

print("\nOpening camera...")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("\nERROR: Could not open camera.")
    print("Try changing CAMERA_INDEX = 0 to 1.")
    exit()

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera started.")
print("\nControls:")
print("  Q = Quit")
print("  S = Save current frame")
print()

# ============================================================
# FPS
# ============================================================

prev_time = time.time()

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # --------------------------------------------------------
    # RUN YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        device=DEVICE,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------------

    annotated_frame = result.plot()

    # --------------------------------------------------------
    # COUNT OBJECTS
    # --------------------------------------------------------

    object_counts = {}

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = CLASS_NAMES.get(
                class_id,
                f"Unknown_{class_id}"
            )

            object_counts[class_name] = (
                object_counts.get(class_name, 0) + 1
            )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time = time.time()

    fps = 1.0 / max(
        current_time - prev_time,
        0.001
    )

    prev_time = current_time

    # --------------------------------------------------------
    # DISPLAY FPS
    # --------------------------------------------------------

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # DISPLAY OBJECT COUNTS
    # --------------------------------------------------------

    y = 80

    for name, count in object_counts.items():

        text = f"{name}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y += 30

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        "Adaptive 2.5D - Live Object Detection",
        annotated_frame
    )

    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord("q"):
        break

    # S = save frame
    elif key == ord("s"):

        save_dir = (
            PROJECT_ROOT
            / "object_detection"
            / "predictions"
            / "live"
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            save_dir
            / f"frame_{int(time.time())}.jpg"
        )

        cv2.imwrite(
            str(filename),
            annotated_frame
        )

        print(
            f"\nFrame saved:\n{filename}"
        )

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("LIVE DETECTION STOPPED")
print("=" * 60)