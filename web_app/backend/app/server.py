import base64
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

from .config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    HIGH_IMAGE_SIZE,
    MODEL_PATH,
    NORMAL_IMAGE_SIZE,
)

from .distance import estimate_distance
from .lidar_pipeline import LiDARPipeline
from .risk import calculate_risk


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Adaptive 2.5D LiDAR Mapping API",
    version="1.0.0",
)


# ============================================================
# GLOBAL STATE
# ============================================================

current_inference_size = NORMAL_IMAGE_SIZE

track_history = {}

lidar_pipeline = LiDARPipeline()


# ============================================================
# STARTUP INFORMATION
# ============================================================

print("=" * 70)
print("ADAPTIVE 2.5D LiDAR MAPPING BACKEND")
print("=" * 70)

print("Loading YOLO model...")
print("Model path:")
print(MODEL_PATH)

print("=" * 70)


# ============================================================
# CHECK MODEL PATH
# ============================================================

model_path = Path(MODEL_PATH)

if not model_path.exists():

    print("WARNING: MODEL PATH DOES NOT EXIST")

    print(
        "The configured MODEL_PATH does not exist on this system:"
    )

    print(model_path)

    print(
        "Make sure config.py points to the correct best.pt."
    )


# ============================================================
# LOAD YOLO
# ============================================================

try:

    model = YOLO(str(model_path))

    print("YOLO model loaded successfully.")

except Exception as error:

    print("=" * 70)
    print("FATAL ERROR: YOLO MODEL COULD NOT BE LOADED")
    print("=" * 70)

    print(repr(error))

    raise


# ============================================================
# DEVICE
# ============================================================

print("Inference device:", DEVICE)

if torch.cuda.is_available():

    print(
        "CUDA GPU detected:",
        torch.cuda.get_device_name(0)
    )

else:

    print("CUDA GPU not detected - using CPU")


# ============================================================
# MODEL INFORMATION
# ============================================================

try:

    print("Model classes:")

    print(model.names)

except Exception:

    print("Could not read model class names.")


print("=" * 70)
print("BACKEND READY")
print("=" * 70)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": (
            "Adaptive 2.5D LiDAR Mapping "
            "Backend is working!"
        ),
        "model": "YOLO",
        "tracking": "ByteTrack",
        "device": str(DEVICE),
        "endpoint": "/ws/detection",
        "lidar_endpoint": "/ws/lidar",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "tracker": "ByteTrack",
    }


# ============================================================
# LiDAR WEBSOCKET
# ============================================================

@app.websocket("/ws/lidar")
async def lidar_websocket(
    websocket: WebSocket,
):

    await websocket.accept()

    print("=" * 50)
    print("LiDAR client connected")
    print("=" * 50)

    try:

        while True:

            payload = await websocket.receive_json()

            if (
                not isinstance(payload, dict)
                or "points" not in payload
            ):

                await websocket.send_json(
                    {
                        "error": (
                            "Expected a JSON object "
                            "with a points array"
                        )
                    }
                )

                continue

            try:

                result = lidar_pipeline.process_frame(
                    payload["points"],
                    payload.get("labels"),
                )

            except (
                TypeError,
                ValueError,
            ) as error:

                print(
                    "LiDAR processing error:",
                    repr(error),
                )

                await websocket.send_json(
                    {
                        "error": str(error)
                    }
                )

                continue

            await websocket.send_json(result)

    except WebSocketDisconnect:

        lidar_pipeline.disconnect()

        print("LiDAR client disconnected")

    except Exception as error:

        lidar_pipeline.disconnect()

        print(
            "LiDAR websocket error:",
            repr(error),
        )


# ============================================================
# DETECTION WEBSOCKET
# ============================================================

@app.websocket("/ws/detection")
async def detection_websocket(
    websocket: WebSocket,
):

    global current_inference_size

    await websocket.accept()

    client_host = (
        websocket.client.host
        if websocket.client
        else "unknown"
    )

    print("=" * 70)
    print("DETECTION CLIENT CONNECTED")
    print("Client:", client_host)
    print("=" * 70)

    previous_frame_time = time.perf_counter()

    frame_number = 0

    try:

        while True:

            # ====================================================
            # RECEIVE FRAME
            # ====================================================

            data = await websocket.receive_text()

            frame_number += 1

            receive_time = time.perf_counter()

            print(
                f"[Frame {frame_number}] "
                f"Received image"
            )

            # ====================================================
            # REMOVE DATA URL PREFIX
            # ====================================================

            if "," in data:

                data = data.split(
                    ",",
                    1
                )[1]

            # ====================================================
            # BASE64 DECODE
            # ====================================================

            try:

                image_bytes = base64.b64decode(
                    data,
                    validate=True
                )

            except Exception as error:

                print(
                    f"[Frame {frame_number}] "
                    f"Invalid base64:",
                    repr(error),
                )

                await websocket.send_json(
                    {
                        "error": "Invalid base64 image"
                    }
                )

                continue

            # ====================================================
            # BASE64 → NUMPY
            # ====================================================

            try:

                np_array = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8
                )

                frame = cv2.imdecode(
                    np_array,
                    cv2.IMREAD_COLOR
                )

            except Exception as error:

                print(
                    f"[Frame {frame_number}] "
                    f"OpenCV decode error:",
                    repr(error),
                )

                await websocket.send_json(
                    {
                        "error": (
                            "Could not decode image"
                        )
                    }
                )

                continue

            # ====================================================
            # FRAME VALIDATION
            # ====================================================

            if frame is None:

                print(
                    f"[Frame {frame_number}] "
                    f"OpenCV returned None"
                )

                await websocket.send_json(
                    {
                        "error": (
                            "OpenCV could not "
                            "decode frame"
                        )
                    }
                )

                continue

            height, width = frame.shape[:2]

            print(
                f"[Frame {frame_number}] "
                f"Image: {width}x{height}"
            )

            # ====================================================
            # YOLO + BYTE TRACK
            # ====================================================

            inference_start = time.perf_counter()

            try:

                with torch.inference_mode():

                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=CONFIDENCE_THRESHOLD,
                        imgsz=current_inference_size,
                        device=DEVICE,
                        verbose=False,
                    )

            except Exception as error:

                print("=" * 70)
                print(
                    f"[Frame {frame_number}] "
                    f"YOLO INFERENCE ERROR"
                )
                print(
                    "ERROR:",
                    repr(error)
                )
                print("=" * 70)

                await websocket.send_json(
                    {
                        "objects": [],
                        "counts": {
                            "danger": 0,
                            "warning": 0,
                            "safe": 0,
                            "total": 0,
                        },
                        "fps": 0,
                        "inference_size": (
                            current_inference_size
                        ),
                        "map": [],
                        "camera": {
                            "width": width,
                            "height": height,
                            "connected": True,
                        },
                        "lidar": lidar_pipeline.latest,
                        "system": {
                            "device": str(DEVICE),
                            "model": "YOLO",
                            "tracker": "ByteTrack",
                            "adaptive_resolution": True,
                        },
                        "error": str(error),
                    }
                )

                continue

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            print(
                f"[Frame {frame_number}] "
                f"YOLO time: "
                f"{inference_time:.3f}s"
            )

            # ====================================================
            # OBJECT LIST
            # ====================================================

            objects = []

            current_track_ids = set()

            current_time = time.time()

            # ====================================================
            # PROCESS YOLO RESULTS
            # ====================================================

            if (
                results
                and len(results) > 0
                and results[0].boxes is not None
            ):

                boxes = results[0].boxes

                box_count = len(boxes)

                print(
                    f"[Frame {frame_number}] "
                    f"Detections: {box_count}"
                )

                for i in range(box_count):

                    box = boxes[i]

                    # ------------------------------------------------
                    # BOUNDING BOX
                    # ------------------------------------------------

                    try:

                        coordinates = (
                            box.xyxy[0]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        x1, y1, x2, y2 = coordinates

                        x1 = max(
                            0,
                            min(
                                width - 1,
                                int(x1)
                            )
                        )

                        y1 = max(
                            0,
                            min(
                                height - 1,
                                int(y1)
                            )
                        )

                        x2 = max(
                            0,
                            min(
                                width - 1,
                                int(x2)
                            )
                        )

                        y2 = max(
                            0,
                            min(
                                height - 1,
                                int(y2)
                            )
                        )

                    except Exception as error:

                        print(
                            "Bounding box error:",
                            repr(error)
                        )

                        continue

                    # ------------------------------------------------
                    # CONFIDENCE
                    # ------------------------------------------------

                    try:

                        confidence = float(
                            box.conf[0]
                            .detach()
                            .cpu()
                            .item()
                        )

                    except Exception:

                        confidence = 0.0

                    # ------------------------------------------------
                    # CLASS ID
                    # ------------------------------------------------

                    try:

                        class_id = int(
                            box.cls[0]
                            .detach()
                            .cpu()
                            .item()
                        )

                    except Exception:

                        class_id = -1

                    # ------------------------------------------------
                    # CLASS NAME
                    # ------------------------------------------------

                    if (
                        class_id >= 0
                        and class_id < len(CLASS_NAMES)
                    ):

                        class_name = (
                            CLASS_NAMES[class_id]
                        )

                    else:

                        # Fallback to YOLO model names.
                        try:

                            class_name = str(
                                model.names[class_id]
                            )

                        except Exception:

                            class_name = "Unknown"

                    # ------------------------------------------------
                    # TRACK ID
                    # ------------------------------------------------

                    track_id = None

                    if boxes.id is not None:

                        try:

                            track_id = int(
                                boxes.id[i]
                                .detach()
                                .cpu()
                                .item()
                            )

                            current_track_ids.add(
                                track_id
                            )

                        except Exception:

                            track_id = None

                    # ------------------------------------------------
                    # PIXEL WIDTH
                    # ------------------------------------------------

                    pixel_width = max(
                        1,
                        x2 - x1
                    )

                    # ------------------------------------------------
                    # DISTANCE
                    # ------------------------------------------------

                    try:

                        distance = float(
                            estimate_distance(
                                class_name,
                                pixel_width,
                            )
                        )

                    except Exception as error:

                        print(
                            "Distance error:",
                            repr(error)
                        )

                        distance = 0.0

                    # ------------------------------------------------
                    # SPEED / APPROACHING / TTC
                    # ------------------------------------------------

                    speed = 0.0

                    approaching = False

                    ttc = None

                    if track_id is not None:

                        previous = track_history.get(
                            track_id
                        )

                        if previous is not None:

                            previous_distance = float(
                                previous["distance"]
                            )

                            previous_time = float(
                                previous["time"]
                            )

                            delta_time = (
                                current_time
                                - previous_time
                            )

                            if delta_time > 0:

                                distance_change = (
                                    previous_distance
                                    - distance
                                )

                                speed = (
                                    distance_change
                                    / delta_time
                                )

                                # Ignore very small movement.
                                if speed > 0.15:

                                    approaching = True

                                # Time-to-collision.
                                if (
                                    approaching
                                    and speed > 0.1
                                ):

                                    ttc = (
                                        distance
                                        / speed
                                    )

                        track_history[
                            track_id
                        ] = {
                            "distance": distance,
                            "time": current_time,
                        }

                    # ------------------------------------------------
                    # RISK
                    # ------------------------------------------------

                    try:

                        risk = calculate_risk(
                            distance,
                            ttc,
                        )

                    except Exception as error:

                        print(
                            "Risk calculation error:",
                            repr(error)
                        )

                        risk = "SAFE"

                    # ------------------------------------------------
                    # CENTER
                    # ------------------------------------------------

                    center_x = (
                        x1 + x2
                    ) / 2.0

                    center_y = (
                        y1 + y2
                    ) / 2.0

                    # ------------------------------------------------
                    # RELATIVE POSITION
                    # ------------------------------------------------

                    relative_x = (
                        center_x / width
                        if width > 0
                        else 0
                    )

                    relative_y = (
                        center_y / height
                        if height > 0
                        else 0
                    )

                    # ------------------------------------------------
                    # 2.5D MAP X
                    # ------------------------------------------------

                    map_x = (
                        relative_x * 2.0
                    ) - 1.0

                    # ------------------------------------------------
                    # OBJECT
                    # ------------------------------------------------

                    object_data = {
                        "id": track_id,

                        "class": class_name,

                        "class_id": class_id,

                        "confidence": round(
                            confidence,
                            3
                        ),

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,

                        "width": max(
                            0,
                            x2 - x1
                        ),

                        "height": max(
                            0,
                            y2 - y1
                        ),

                        "center_x": round(
                            center_x,
                            1
                        ),

                        "center_y": round(
                            center_y,
                            1
                        ),

                        "distance": round(
                            distance,
                            2
                        ),

                        "speed": round(
                            speed,
                            2
                        ),

                        "ttc": (
                            round(ttc, 2)
                            if ttc is not None
                            else None
                        ),

                        "approaching": (
                            approaching
                        ),

                        "risk": risk,

                        "rel_x": round(
                            relative_x,
                            3
                        ),

                        "rel_y": round(
                            relative_y,
                            3
                        ),

                        "map_x": round(
                            map_x,
                            3
                        ),

                        "map_distance": round(
                            distance,
                            2
                        ),
                    }

                    objects.append(
                        object_data
                    )

            else:

                print(
                    f"[Frame {frame_number}] "
                    f"No detections"
                )

            # ====================================================
            # COUNTS
            # ====================================================

            danger_count = sum(
                1
                for obj in objects
                if obj["risk"] == "DANGER"
            )

            warning_count = sum(
                1
                for obj in objects
                if obj["risk"] == "WARNING"
            )

            safe_count = sum(
                1
                for obj in objects
                if obj["risk"] == "SAFE"
            )

            # ====================================================
            # FPS
            # ====================================================

            now = time.perf_counter()

            elapsed = (
                now
                - previous_frame_time
            )

            fps = (
                1.0 / elapsed
                if elapsed > 0
                else 0.0
            )

            previous_frame_time = now

            # ====================================================
            # ADAPTIVE RESOLUTION
            # ====================================================

            if danger_count > 0:

                current_inference_size = (
                    HIGH_IMAGE_SIZE
                )

            elif warning_count > 0:

                current_inference_size = (
                    HIGH_IMAGE_SIZE
                )

            elif len(objects) >= 6:

                current_inference_size = (
                    HIGH_IMAGE_SIZE
                )

            else:

                current_inference_size = (
                    NORMAL_IMAGE_SIZE
                )

            # ====================================================
            # 2.5D MAP
            # ====================================================

            map_objects = []

            for obj in objects:

                map_objects.append(
                    {
                        "id": obj["id"],

                        "class": obj["class"],

                        "x": obj["map_x"],

                        "distance": (
                            obj["map_distance"]
                        ),

                        "risk": obj["risk"],

                        "approaching": (
                            obj["approaching"]
                        ),
                    }
                )

            # ====================================================
            # RESPONSE
            # ====================================================

            response = {

                "objects": objects,

                "counts": {
                    "danger": danger_count,
                    "warning": warning_count,
                    "safe": safe_count,
                    "total": len(objects),
                },

                "fps": round(
                    fps,
                    1
                ),

                "inference_size": (
                    current_inference_size
                ),

                "image": {
                    "width": width,
                    "height": height,
                },

                "map": map_objects,

                "camera": {
                    "width": width,
                    "height": height,
                    "connected": True,
                },

                "lidar": (
                    lidar_pipeline.latest
                ),

                "system": {
                    "device": str(DEVICE),
                    "model": "YOLO",
                    "tracker": "ByteTrack",
                    "adaptive_resolution": True,
                },

                "debug": {
                    "frame": frame_number,
                    "receive_ms": round(
                        (
                            receive_time
                            - previous_frame_time
                        )
                        * 1000,
                        2,
                    ),
                    "inference_ms": round(
                        inference_time * 1000,
                        2,
                    ),
                },
            }

            # ====================================================
            # SEND RESULT
            # ====================================================

            try:

                await websocket.send_json(
                    response
                )

                print(
                    f"[Frame {frame_number}] "
                    f"Response sent | "
                    f"Objects={len(objects)} | "
                    f"Danger={danger_count} | "
                    f"Warning={warning_count} | "
                    f"Safe={safe_count} | "
                    f"FPS={fps:.1f}"
                )

            except Exception as error:

                print(
                    "Could not send response:",
                    repr(error)
                )

                break

    except WebSocketDisconnect:

        print("=" * 70)
        print(
            "DETECTION CLIENT DISCONNECTED"
        )
        print(
            "Client:",
            client_host
        )
        print(
            "Frames processed:",
            frame_number
        )
        print("=" * 70)

    except Exception as error:

        print("=" * 70)
        print("WEBSOCKET ERROR")
        print("Client:", client_host)
        print(
            "Frames processed:",
            frame_number
        )
        print(
            "ERROR:",
            repr(error)
        )
        print("=" * 70)