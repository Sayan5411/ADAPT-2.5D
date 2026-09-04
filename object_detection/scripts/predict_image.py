from ultralytics import YOLO
from pathlib import Path

MODEL = r"D:\Lidar_Adaptive_2_5D\object_detection\runs\lidar_object_detector-4\weights\best.pt"

IMAGE = r"D:\Lidar_Adaptive_2_5D\object_detection\dataset\images\val\000000.png"

model = YOLO(MODEL)

results = model.predict(
    source=IMAGE,
    conf=0.40,
    imgsz=640,
    device=0,
    save=True
)

print("Prediction completed.")

for result in results:
    if result.boxes is None:
        continue

    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        print(
            f"Detected: {model.names[cls]} "
            f"| Confidence: {conf:.2f}"
        )