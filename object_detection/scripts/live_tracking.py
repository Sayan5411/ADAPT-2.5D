import cv2
import torch
import time
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(r"D:\Lidar_Adaptive_2_5D")

MODEL_PATH = (
    ROOT
    / "object_detection"
    / "runs"
    / "lidar_object_detector-4"
    / "weights"
    / "best.pt"
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.40
IMAGE_SIZE = 640


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
# START
# ============================================================

print("=" * 70)
print("ADAPTIVE 2.5D - LIVE CAMERA TRACKING")
print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    print("\nERROR: Model not found:")
    print(MODEL_PATH)

    exit()


print("\nModel found:")
print(MODEL_PATH)


# ============================================================
# GPU
# ============================================================

if torch.cuda.is_available():

    DEVICE = 0

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

else:

    DEVICE = "cpu"

    print("\nCUDA not available.")
    print("Using CPU.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")


# ============================================================
# FIND CAMERAS
# ============================================================

print("\nSearching for cameras...")

available_cameras = []

for camera_index in range(10):

    cap_test = cv2.VideoCapture(
        camera_index,
        cv2.CAP_DSHOW
    )

    if cap_test.isOpened():

        ret, frame = cap_test.read()

        if ret:

            available_cameras.append(camera_index)

            print(
                f"Camera {camera_index} detected"
            )

    cap_test.release()


# ============================================================
# NO CAMERA
# ============================================================

if not available_cameras:

    print("\nERROR: No camera detected.")

    print("\nMake sure your phone is connected")
    print("and available to Windows as a camera.")

    exit()


# ============================================================
# SELECT CAMERA
# ============================================================

print("\nAvailable cameras:")

for index in available_cameras:

    print(
        f"  [{index}] Camera {index}"
    )


if len(available_cameras) == 1:

    CAMERA_INDEX = available_cameras[0]

    print(
        f"\nUsing Camera {CAMERA_INDEX}"
    )

else:

    print("\nSelect the camera you want to use.")

    try:

        CAMERA_INDEX = int(
            input("Enter camera number: ")
        )

    except ValueError:

        print("Invalid camera number.")

        exit()


# ============================================================
# OPEN CAMERA
# ============================================================

print(
    f"\nOpening Camera {CAMERA_INDEX}..."
)

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print("\nERROR: Could not open camera.")

    exit()


# ============================================================
# CAMERA RESOLUTION
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print("\nCamera started successfully.")

print("\nControls:")
print("  Q = Quit")
print("  S = Save frame")

print("\nStarting LIVE TRACKING...\n")


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print(
            "\nERROR: Could not read camera frame."
        )

        break


    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = model.track(

        source=frame,

        conf=CONFIDENCE,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        tracker="bytetrack.yaml",

        persist=True,

        verbose=False
    )


    result = results[0]


    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    annotated_frame = result.plot()


    # ========================================================
    # OBJECT INFORMATION
    # ========================================================

    object_count = 0


    if result.boxes is not None:

        for box in result.boxes:

            object_count += 1


            # ------------------------------------------------
            # CLASS
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )


            class_name = CLASS_NAMES.get(
                class_id,
                f"Unknown_{class_id}"
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # TRACK ID
            # ------------------------------------------------

            if box.id is not None:

                track_id = int(
                    box.id[0]
                )

            else:

                track_id = -1


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (
                f"ID {track_id} | "
                f"{class_name} | "
                f"{confidence:.2f}"
            )


            cv2.putText(

                annotated_frame,

                label,

                (
                    x1,
                    max(y1 - 10, 20)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2
            )


            # ------------------------------------------------
            # CENTER POINT
            # ------------------------------------------------

            cv2.circle(

                annotated_frame,

                (
                    center_x,
                    center_y
                ),

                4,

                (255, 255, 255),

                -1
            )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1.0 / max(
        current_time - previous_time,
        0.001
    )

    previous_time = current_time


    # ========================================================
    # DASHBOARD
    # ========================================================

    cv2.rectangle(

        annotated_frame,

        (10, 10),

        (290, 115),

        (0, 0, 0),

        -1
    )


    cv2.putText(

        annotated_frame,

        f"FPS: {fps:.1f}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    cv2.putText(

        annotated_frame,

        f"Objects: {object_count}",

        (20, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    cv2.putText(

        annotated_frame,

        "LIVE TRACKING",

        (20, 100),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(

        "Adaptive 2.5D - Live Camera Tracking",

        annotated_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    elif key == ord("s"):

        save_dir = (
            ROOT
            / "object_detection"
            / "predictions"
            / "live_tracking"
        )


        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        filename = (
            save_dir
            / f"tracking_{int(time.time())}.jpg"
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


print("\n" + "=" * 70)
print("LIVE TRACKING STOPPED")
print("=" * 70)