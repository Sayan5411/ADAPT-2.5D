import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

# ============================================================
# CONFIG
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

VIDEO_PATH = (
    PROJECT_ROOT
    / "object_detection"
    / "videos"
    / "input.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "object_detection"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VIDEO = OUTPUT_DIR / "detected_video.mp4"

CONFIDENCE = 0.45

# ============================================================
# DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("VIDEO OBJECT DETECTION")
print("=" * 60)

# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    print("ERROR: Model not found!")
    print(MODEL_PATH)
    exit()

if not VIDEO_PATH.exists():

    print("ERROR: Video not found!")
    print(VIDEO_PATH)
    print("\nPut your video here:")
    print(
        PROJECT_ROOT
        / "object_detection"
        / "videos"
    )

    exit()

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded.")

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():

    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print(f"\nVideo resolution: {width}x{height}")
print(f"FPS: {fps}")

# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

# ============================================================
# PROCESS
# ============================================================

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        device=DEVICE,
        imgsz=640,
        verbose=False
    )

    annotated = results[0].plot()

    writer.write(annotated)

    cv2.imshow(
        "Video Object Detection",
        annotated
    )

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("VIDEO DETECTION COMPLETED")
print("=" * 60)

print(f"\nProcessed frames: {frame_count}")

print("\nOutput:")
print(OUTPUT_VIDEO)