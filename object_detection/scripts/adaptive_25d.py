import cv2
import torch
import time
import math
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(r"D:\Lidar_Adaptive_2_5D")

FOCAL_LENGTH_FILE = ROOT / "object_detection" / "focal_length.txt"

if FOCAL_LENGTH_FILE.exists():
    FOCAL_LENGTH = float(FOCAL_LENGTH_FILE.read_text().strip())
    print(f"\nLoaded calibrated focal length: {FOCAL_LENGTH:.2f}")
else:
    FOCAL_LENGTH = 700.0
    print("\nNo calibration file found — using default focal length (700.0)")

MODEL_PATH = (
    ROOT / "object_detection" / "runs" / "lidar_object_detector-4"
    / "weights" / "best.pt"
)


# ============================================================
# YOLO SETTINGS
# ============================================================

CONFIDENCE = 0.40
NORMAL_IMAGE_SIZE = 640
HIGH_IMAGE_SIZE = 960

CAMERA_HFOV_DEGREES = 60.0   # approximate horizontal field of view of your camera


# ============================================================
# APPROXIMATE OBJECT WIDTHS (meters)
# ============================================================

OBJECT_WIDTHS = {
    "Car": 1.80, "Van": 2.00, "Truck": 2.50, "Pedestrian": 0.50,
    "Person_sitting": 0.50, "Cyclist": 0.60, "Tram": 2.50, "Misc": 1.00
}

CLASS_NAMES = {
    0: "Car", 1: "Van", 2: "Truck", 3: "Pedestrian",
    4: "Person_sitting", 5: "Cyclist", 6: "Tram", 7: "Misc"
}


# ============================================================
# RISK THRESHOLDS (meters)
# ============================================================

DANGER_DISTANCE = 4.0
WARNING_DISTANCE = 10.0


# ============================================================
# TRACK STABILITY / TTC SETTINGS
# ============================================================

SMOOTHING_ALPHA = 0.35
GRACE_FRAMES = 15
APPROACH_SPEED_THRESHOLD = 0.5
TTC_DANGER_SECONDS = 3.0
SPEED_SMOOTHING_ALPHA = 0.4


# ============================================================
# COLORS (BGR)
# ============================================================

COLOR_SAFE = (90, 200, 90)
COLOR_WARNING = (0, 210, 255)
COLOR_DANGER = (60, 60, 255)
COLOR_TEXT = (235, 235, 235)
COLOR_DIM = (150, 150, 150)
COLOR_PANEL_BG = (25, 22, 20)
COLOR_HEADER_BG = (55, 40, 20)
COLOR_ACCENT = (255, 170, 60)
COLOR_MAP_BG = (18, 16, 15)
COLOR_GRID = (45, 40, 38)
COLOR_RING = (60, 55, 50)

RISK_COLORS = {
    "SAFE": COLOR_SAFE,
    "WARNING": COLOR_WARNING,
    "DANGER": COLOR_DANGER
}

track_memory = {}
frame_number = 0


# ============================================================
# MAP WINDOW SETTINGS
# ============================================================

MAP_WINDOW_SIZE = 700          # square window, pixels
MAP_MAX_RANGE = 30.0           # meters shown edge-to-edge
MAP_RING_STEP = 5.0            # meters between distance rings


def build_map_frame(track_memory):
    """Builds a standalone bird's-eye 2.5D perception map."""

    size = MAP_WINDOW_SIZE
    map_img = cv2.copyMakeBorder(
        cv2.UMat(size, size, cv2.CV_8UC3).get() if False else
        cv2.cvtColor(cv2.UMat(1, 1).get(), cv2.COLOR_GRAY2BGR) if False else
        None, 0, 0, 0, 0, cv2.BORDER_CONSTANT
    ) if False else None

    map_img = cv2.rectangle(
        cv2.UMat(size, size, cv2.CV_8UC3).get() if False else
        __import__("numpy").full((size, size, 3), COLOR_MAP_BG, dtype="uint8"),
        (0, 0), (size, size), COLOR_MAP_BG, -1
    )

    apex_x = size // 2
    apex_y = size - 60

    max_range = MAP_MAX_RANGE
    scale = (size - 100) / max_range  # pixels per meter

    # ------------------------------------------------------
    # DISTANCE RINGS
    # ------------------------------------------------------

    ring_distance = MAP_RING_STEP
    while ring_distance <= max_range:
        radius = int(ring_distance * scale)
        cv2.ellipse(
            map_img, (apex_x, apex_y), (radius, radius),
            0, 180, 360, COLOR_RING, 1
        )
        cv2.putText(
            map_img, f"{int(ring_distance)}m",
            (apex_x + 6, apex_y - radius + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DIM, 1
        )
        ring_distance += MAP_RING_STEP

    # ------------------------------------------------------
    # FIELD OF VIEW CONE
    # ------------------------------------------------------

    half_fov = math.radians(CAMERA_HFOV_DEGREES / 2)
    edge_len = max_range * scale

    left_x = int(apex_x - edge_len * math.sin(half_fov))
    left_y = int(apex_y - edge_len * math.cos(half_fov))
    right_x = int(apex_x + edge_len * math.sin(half_fov))
    right_y = int(apex_y - edge_len * math.cos(half_fov))

    cv2.line(map_img, (apex_x, apex_y), (left_x, left_y), COLOR_GRID, 1)
    cv2.line(map_img, (apex_x, apex_y), (right_x, right_y), COLOR_GRID, 1)

    # ------------------------------------------------------
    # CAMERA / "YOU" MARKER
    # ------------------------------------------------------

    cv2.circle(map_img, (apex_x, apex_y), 9, COLOR_TEXT, -1)
    cv2.putText(map_img, "YOU", (apex_x - 18, apex_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM, 1)

    # ------------------------------------------------------
    # TITLE
    # ------------------------------------------------------

    cv2.putText(map_img, "2.5D PERCEPTION MAP", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_ACCENT, 2)
    cv2.putText(map_img, f"Range: {int(max_range)}m   FOV: {int(CAMERA_HFOV_DEGREES)} deg",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DIM, 1)

    # ------------------------------------------------------
    # PLOT OBJECTS
    # ------------------------------------------------------

    for tid, data in track_memory.items():
        distance = min(data["distance"], max_range)

        # relative_x in [0,1] across the frame -> angle across FOV
        angle = (data["x"] - 0.5) * math.radians(CAMERA_HFOV_DEGREES)

        obj_x = int(apex_x + distance * scale * math.sin(angle))
        obj_y = int(apex_y - distance * scale * math.cos(angle))

        color = RISK_COLORS[data["risk"]]
        radius = 11 if data["approaching"] else 8

        # Danger objects get a pulsing ring
        if data["risk"] == "DANGER":
            cv2.circle(map_img, (obj_x, obj_y), radius + 6, color, 2)

        cv2.circle(map_img, (obj_x, obj_y), radius, color, -1)

        label = f"#{tid} {data['class']}"
        cv2.putText(map_img, label, (obj_x + 12, obj_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1)
        cv2.putText(map_img, f"{distance:.1f}m", (obj_x + 12, obj_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if data["approaching"] and data.get("ttc"):
            cv2.putText(map_img, f"TTC {data['ttc']:.1f}s", (obj_x + 12, obj_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_ACCENT, 1)

    # ------------------------------------------------------
    # LEGEND
    # ------------------------------------------------------

    legend_y = size - 25
    cv2.circle(map_img, (25, legend_y), 6, COLOR_SAFE, -1)
    cv2.putText(map_img, "Safe", (38, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1)
    cv2.circle(map_img, (110, legend_y), 6, COLOR_WARNING, -1)
    cv2.putText(map_img, "Warning", (123, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1)
    cv2.circle(map_img, (220, legend_y), 6, COLOR_DANGER, -1)
    cv2.putText(map_img, "Danger", (233, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1)

    return map_img


def draw_section_divider(img, x, y, width):
    cv2.line(img, (x, y), (x + width, y), (70, 60, 50), 1)


# ============================================================
# START
# ============================================================

print("=" * 80)
print("ADAPTIVE 2.5D LIVE PERCEPTION SYSTEM")
print("=" * 80)

if not MODEL_PATH.exists():
    print("\nERROR: Model not found:")
    print(MODEL_PATH)
    exit()

print("\nModel:")
print(MODEL_PATH)

if torch.cuda.is_available():
    DEVICE = 0
    print("\nGPU:")
    print(torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("\nUsing CPU")

print("\nLoading YOLO model...")
model = YOLO(str(MODEL_PATH))
print("Model loaded!")


# ============================================================
# AUTO-DETECT CAMERA
# ============================================================

print("\nSearching for available cameras...")

available_cameras = []
for index in range(6):
    cap_test = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap_test.isOpened():
        ret, frame = cap_test.read()
        if ret:
            available_cameras.append(index)
            print(f"  Camera {index} detected")
    cap_test.release()

if not available_cameras:
    print("\nERROR: No camera detected.")
    exit()

if len(available_cameras) == 1:
    CAMERA_INDEX = available_cameras[0]
    print(f"\nUsing the only available camera: {CAMERA_INDEX}")
else:
    print(f"\nMultiple cameras found: {available_cameras}")
    try:
        CAMERA_INDEX = int(input("Enter the camera number to use: "))
    except ValueError:
        print("Invalid input.")
        exit()

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("\nERROR: Could not open selected camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Camera started!")


# ============================================================
# STATE
# ============================================================

previous_time = time.time()
current_image_size = NORMAL_IMAGE_SIZE

cv2.namedWindow("Adaptive 2.5D Live Perception", cv2.WINDOW_NORMAL)
cv2.namedWindow("2.5D Perception Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("2.5D Perception Map", MAP_WINDOW_SIZE, MAP_WINDOW_SIZE)


# ============================================================
# MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("\nCamera frame unavailable.")
        break

    frame_number += 1
    now = time.time()

    results = model.track(
        source=frame, conf=CONFIDENCE, imgsz=current_image_size,
        device=DEVICE, tracker="bytetrack.yaml", persist=True, verbose=False
    )

    result = results[0]
    display = frame.copy()

    danger_count = 0
    warning_count = 0
    safe_count = 0
    approaching_count = 0

    frame_height, frame_width = frame.shape[:2]

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES.get(class_id, "Unknown")
            track_id = int(box.id[0]) if box.id is not None else -1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            pixel_width = max(x2 - x1, 1)

            known_width = OBJECT_WIDTHS.get(class_name, 1.0)
            raw_distance = known_width * FOCAL_LENGTH / pixel_width
            raw_distance = max(min(raw_distance, 100.0), 0.5)

            previous_entry = track_memory.get(track_id)

            if previous_entry is not None:
                smoothed_distance = (
                    SMOOTHING_ALPHA * raw_distance
                    + (1 - SMOOTHING_ALPHA) * previous_entry["distance"]
                )
            else:
                smoothed_distance = raw_distance

            speed = 0.0
            ttc = None

            if previous_entry is not None:
                dt = max(now - previous_entry["last_seen_time"], 1e-3)
                raw_speed = (previous_entry["distance"] - smoothed_distance) / dt
                previous_speed = previous_entry.get("speed", 0.0)
                speed = (
                    SPEED_SMOOTHING_ALPHA * raw_speed
                    + (1 - SPEED_SMOOTHING_ALPHA) * previous_speed
                )
                if speed >= APPROACH_SPEED_THRESHOLD:
                    ttc = smoothed_distance / speed

            approaching = speed >= APPROACH_SPEED_THRESHOLD
            if approaching:
                approaching_count += 1

            if smoothed_distance <= DANGER_DISTANCE:
                risk = "DANGER"
            elif smoothed_distance <= WARNING_DISTANCE:
                risk = "WARNING"
            else:
                risk = "SAFE"

            if ttc is not None and ttc <= TTC_DANGER_SECONDS:
                risk = "DANGER"

            if risk == "DANGER":
                danger_count += 1
            elif risk == "WARNING":
                warning_count += 1
            else:
                safe_count += 1

            relative_x = center_x / frame_width
            relative_y = center_y / frame_height

            track_memory[track_id] = {
                "distance": smoothed_distance, "speed": speed, "class": class_name,
                "x": relative_x, "y": relative_y, "risk": risk,
                "approaching": approaching, "ttc": ttc,
                "last_seen_frame": frame_number, "last_seen_time": now
            }

            box_color = RISK_COLORS[risk]

            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(display, (center_x, center_y), 4, box_color, -1)

            label = f"ID {track_id}  {class_name}  {smoothed_distance:.1f}m"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display, (x1, y1 - lh - 14), (x1 + lw + 10, y1 - 4), box_color, -1)
            cv2.putText(display, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)

            if approaching:
                risk_label = f"{risk}  |  APPROACHING  TTC {ttc:.1f}s" if ttc else f"{risk}  |  APPROACHING"
            else:
                risk_label = risk

            cv2.putText(display, risk_label, (x1, min(y2 + 25, frame_height - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    stale_ids = [tid for tid, d in track_memory.items()
                 if frame_number - d["last_seen_frame"] > GRACE_FRAMES]
    for tid in stale_ids:
        del track_memory[tid]

    object_count = len(track_memory)
    current_image_size = HIGH_IMAGE_SIZE if (danger_count > 0 or warning_count > 0) else NORMAL_IMAGE_SIZE

    current_time = time.time()
    fps = 1.0 / max(current_time - previous_time, 0.001)
    previous_time = current_time

    # ========================================================
    # DASHBOARD PANEL (camera window, no embedded map anymore)
    # ========================================================

    panel_width = 340
    dashboard = display.copy()

    cv2.rectangle(dashboard, (0, 0), (panel_width, frame.shape[0]), COLOR_PANEL_BG, -1)

    cv2.rectangle(dashboard, (0, 0), (panel_width, 70), COLOR_HEADER_BG, -1)
    cv2.putText(dashboard, "ADAPTIVE 2.5D", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_ACCENT, 2)
    cv2.putText(dashboard, "LIVE PERCEPTION SYSTEM", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    y = 100
    cv2.putText(dashboard, f"FPS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM, 1)
    cv2.putText(dashboard, f"{fps:.1f}", (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    y += 28
    cv2.putText(dashboard, f"Objects", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM, 1)
    cv2.putText(dashboard, f"{object_count}", (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    y += 28
    cv2.putText(dashboard, f"Inference", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM, 1)
    cv2.putText(dashboard, f"{current_image_size}px", (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)

    y += 25
    draw_section_divider(dashboard, 20, y, panel_width - 40)
    y += 35

    cv2.putText(dashboard, "RISK STATUS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 2)
    y += 32
    cv2.circle(dashboard, (28, y - 5), 6, COLOR_DANGER, -1)
    cv2.putText(dashboard, f"Danger      {danger_count}", (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
    y += 28
    cv2.circle(dashboard, (28, y - 5), 6, COLOR_WARNING, -1)
    cv2.putText(dashboard, f"Warning     {warning_count}", (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
    y += 28
    cv2.circle(dashboard, (28, y - 5), 6, COLOR_SAFE, -1)
    cv2.putText(dashboard, f"Safe        {safe_count}", (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)
    y += 28
    cv2.putText(dashboard, f"Approaching  {approaching_count}", (28, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)

    y += 25
    draw_section_divider(dashboard, 20, y, panel_width - 40)
    y += 35

    cv2.putText(dashboard, "TRACKED OBJECTS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 2)
    y += 30

    sorted_tracks = sorted(track_memory.items(), key=lambda item: item[1]["distance"])

    for tid, data in sorted_tracks[:8]:
        color = RISK_COLORS[data["risk"]]
        marker = "  ->" if data["approaching"] else ""
        text = f"#{tid}  {data['class']}  {data['distance']:.1f}m{marker}"
        cv2.putText(dashboard, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
        y += 24
        if y > frame.shape[0] - 20:
            break

    cv2.putText(dashboard, f"CAM {CAMERA_INDEX}",
                (frame.shape[1] - 100, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM, 1)

    # ========================================================
    # SEPARATE 2.5D MAP WINDOW
    # ========================================================

    map_frame = build_map_frame(track_memory)

    cv2.imshow("Adaptive 2.5D Live Perception", dashboard)
    cv2.imshow("2.5D Perception Map", map_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        save_dir = ROOT / "object_detection" / "predictions" / "live"
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        cv2.imwrite(str(save_dir / f"perception_{timestamp}.jpg"), dashboard)
        cv2.imwrite(str(save_dir / f"map_{timestamp}.jpg"), map_frame)
        print(f"\nSaved dashboard and map screenshots (timestamp {timestamp})")

cap.release()
cv2.destroyAllWindows()
print("\n" + "=" * 80)
print("ADAPTIVE 2.5D SYSTEM STOPPED")
print("=" * 80)