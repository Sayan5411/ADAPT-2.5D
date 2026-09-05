import base64
import time
from typing import Any, Dict, List, Optional

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
    description="Dynamic Environment Perception using YOLO, ByteTrack and LiDAR",
    version="1.0.0",
)


# ============================================================
# YOLO MODEL
# ============================================================

print("=" * 70)
print("STARTING ADAPTIVE 2.5D LIDAR MAPPING BACKEND")
print("=" * 70)

print("Loading YOLO model...")
print("Model path:", MODEL_PATH)

try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully")
except Exception as error:
    print("ERROR loading YOLO model:")
    print(repr(error))
    raise


print("Inference device:", DEVICE)

if torch.cuda.is_available():
    print("CUDA GPU detected")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA GPU not detected - using CPU")


# ============================================================
# CLASS NAMES
# ============================================================

# Use configured class names first.
# If unavailable, fall back to the names stored in the YOLO model.

try:
    MODEL_CLASS_NAMES = model.names
except Exception:
    MODEL_CLASS_NAMES = {}


def get_class_name(class_id: int) -> str:
    """
    Safely convert YOLO class ID into a class name.
    """

    # First use project configuration.
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]

    # Then try YOLO model names.
    try:
        if isinstance(MODEL_CLASS_NAMES, dict):
            return str(MODEL_CLASS_NAMES.get(class_id, "Unknown"))

        if isinstance(MODEL_CLASS_NAMES, list):
            if 0 <= class_id < len(MODEL_CLASS_NAMES):
                return str(MODEL_CLASS_NAMES[class_id])

    except Exception:
        pass

    return "Unknown"


# ============================================================
# ADAPTIVE RESOLUTION
# ============================================================

current_inference_size = NORMAL_IMAGE_SIZE


# ============================================================
# TRACK HISTORY
# ============================================================

# track_id -> {
#     "distance": float,
#     "time": float
# }

track_history: Dict[int, Dict[str, float]] = {}

MAX_TRACK_HISTORY = 500


# ============================================================
# LIDAR
# ============================================================

try:
    lidar_pipeline = LiDARPipeline()
    print("LiDAR pipeline initialized successfully")
except Exception as error:
    print("WARNING: LiDAR pipeline initialization failed:")
    print(repr(error))
    lidar_pipeline = None


def get_latest_lidar() -> Dict[str, Any]:
    """
    Safely return the latest LiDAR state.
    """

    default_lidar = {
        "connected": False,
        "points": [],
        "point_count": 0,
        "grid": [],
        "elevation": [],
        "traversability": [],
    }

    if lidar_pipeline is None:
        return default_lidar

    try:
        latest = getattr(lidar_pipeline, "latest", None)

        if latest is None:
            return default_lidar

        return latest

    except Exception as error:
        print("LiDAR latest-state error:", repr(error))
        return default_lidar


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Adaptive 2.5D LiDAR Mapping Backend is working!",
        "model": "YOLO",
        "tracking": "ByteTrack",
        "device": str(DEVICE),
        "endpoint": "/ws/detection",
        "lidar_endpoint": "/ws/lidar",
        "confidence": CONFIDENCE_THRESHOLD,
        "normal_resolution": NORMAL_IMAGE_SIZE,
        "high_resolution": HIGH_IMAGE_SIZE,
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
        "cuda_available": torch.cuda.is_available(),
        "tracking": "ByteTrack",
        "adaptive_resolution": True,
        "lidar_available": lidar_pipeline is not None,
    }


# ============================================================
# LIDAR WEBSOCKET
# ============================================================

@app.websocket("/ws/lidar")
async def lidar_websocket(websocket: WebSocket):

    await websocket.accept()

    print("=" * 70)
    print("LiDAR client connected")
    print("=" * 70)

    try:

        while True:

            payload = await websocket.receive_json()

            if not isinstance(payload, dict):
                await websocket.send_json(
                    {
                        "error": "Expected a JSON object"
                    }
                )
                continue

            if "points" not in payload:

                await websocket.send_json(
                    {
                        "error": "Expected a JSON object with a points array"
                    }
                )

                continue

            if lidar_pipeline is None:

                await websocket.send_json(
                    {
                        "error": "LiDAR pipeline is unavailable"
                    }
                )

                continue

            try:

                result = lidar_pipeline.process_frame(
                    payload["points"],
                    payload.get("labels"),
                )

            except (TypeError, ValueError) as error:

                print(
                    "LiDAR processing validation error:",
                    repr(error)
                )

                await websocket.send_json(
                    {
                        "error": str(error)
                    }
                )

                continue

            except Exception as error:

                print(
                    "LiDAR processing error:",
                    repr(error)
                )

                await websocket.send_json(
                    {
                        "error": "LiDAR processing failed"
                    }
                )

                continue

            await websocket.send_json(result)

    except WebSocketDisconnect:

        if lidar_pipeline is not None:

            try:
                lidar_pipeline.disconnect()
            except Exception:
                pass

        print("LiDAR client disconnected")

    except Exception as error:

        if lidar_pipeline is not None:

            try:
                lidar_pipeline.disconnect()
            except Exception:
                pass

        print("LiDAR websocket error:", repr(error))


# ============================================================
# DETECTION WEBSOCKET
# ============================================================

@app.websocket("/ws/detection")
async def detection_websocket(
    websocket: WebSocket
):

    global current_inference_size

    # --------------------------------------------------------
    # ACCEPT CONNECTION
    # --------------------------------------------------------

    await websocket.accept()

    client_host = "unknown"

    try:

        if websocket.client is not None:
            client_host = websocket.client.host

    except Exception:
        pass

    print("=" * 70)
    print("Frontend connected")
    print("Client:", client_host)
    print("WebSocket endpoint: /ws/detection")
    print("=" * 70)

    previous_frame_time = time.time()

    frame_number = 0

    try:

        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            frame_number += 1

            frame_start_time = time.time()

            # ====================================================
            # RECEIVE IMAGE
            # ====================================================

            try:

                data = await websocket.receive_text()

            except WebSocketDisconnect:

                raise

            except Exception as error:

                print(
                    "ERROR receiving frame:",
                    repr(error)
                )

                continue

            print(
                f"[FRAME {frame_number}] FRAME RECEIVED | "
                f"data_length={len(data)}"
            )

            # ====================================================
            # VALIDATE RECEIVED DATA
            # ====================================================

            if not data:

                print(
                    f"[FRAME {frame_number}] Empty frame received"
                )

                continue

            # ====================================================
            # REMOVE DATA URL PREFIX
            # ====================================================

            if "," in data:

                data = data.split(",", 1)[1]

            # ====================================================
            # BASE64 -> BYTES
            # ====================================================

            try:

                image_bytes = base64.b64decode(
                    data,
                    validate=True
                )

            except Exception as error:

                print(
                    f"[FRAME {frame_number}] "
                    f"Invalid base64 image: {repr(error)}"
                )

                continue

            if not image_bytes:

                print(
                    f"[FRAME {frame_number}] "
                    "Decoded image is empty"
                )

                continue

            print(
                f"[FRAME {frame_number}] "
                f"Base64 decoded | bytes={len(image_bytes)}"
            )

            # ====================================================
            # BYTES -> NUMPY
            # ====================================================

            try:

                np_array = np.frombuffer(
                    image_bytes,
                    np.uint8
                )

            except Exception as error:

                print(
                    f"[FRAME {frame_number}] "
                    f"NumPy conversion failed: {repr(error)}"
                )

                continue

            # ====================================================
            # NUMPY -> OPENCV IMAGE
            # ====================================================

            try:

                frame = cv2.imdecode(
                    np_array,
                    cv2.IMREAD_COLOR
                )

            except Exception as error:

                print(
                    f"[FRAME {frame_number}] "
                    f"OpenCV decode error: {repr(error)}"
                )

                continue

            if frame is None:

                print(
                    f"[FRAME {frame_number}] "
                    "OpenCV returned None"
                )

                continue

            height, width = frame.shape[:2]

            print(
                f"[FRAME {frame_number}] "
                f"IMAGE DECODED | {width}x{height}"
            )

            # ====================================================
            # YOLO + BYTE TRACK
            # ====================================================

            print(
                f"[FRAME {frame_number}] "
                f"RUNNING YOLO | imgsz={current_inference_size}"
            )

            yolo_start_time = time.time()

            try:

                results = model.track(

                    frame,

                    persist=True,

                    tracker="bytetrack.yaml",

                    conf=CONFIDENCE_THRESHOLD,

                    imgsz=current_inference_size,

                    device=DEVICE,

                    verbose=False,

                )

            except Exception as error:

                print(
                    f"[FRAME {frame_number}] "
                    "YOLO ERROR:",
                    repr(error)
                )

                # Keep WebSocket alive instead of killing it.
                try:

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
                            "inference_size": current_inference_size,
                            "image": {
                                "width": width,
                                "height": height,
                            },
                            "map": [],
                            "camera": {
                                "width": width,
                                "height": height,
                                "connected": True,
                            },
                            "lidar": get_latest_lidar(),
                            "system": {
                                "device": str(DEVICE),
                                "model": "YOLO",
                                "tracker": "ByteTrack",
                                "adaptive_resolution": True,
                                "error": "YOLO inference failed",
                            },
                        }
                    )

                except Exception:
                    pass

                continue

            yolo_time = time.time() - yolo_start_time

            print(
                f"[FRAME {frame_number}] "
                f"YOLO FINISHED | "
                f"time={yolo_time:.3f}s"
            )

            # ====================================================
            # DETECTION RESULT CONTAINER
            # ====================================================

            objects: List[Dict[str, Any]] = []

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

                detection_count = len(boxes)

                print(
                    f"[FRAME {frame_number}] "
                    f"YOLO DETECTIONS: {detection_count}"
                )

                # ------------------------------------------------
                # PROCESS EACH OBJECT
                # ------------------------------------------------

                for i in range(len(boxes)):

                    try:

                        box = boxes[i]

                        # ========================================
                        # BOUNDING BOX
                        # ========================================

                        x1, y1, x2, y2 = (
                            box.xyxy[0]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)

                        # ========================================
                        # CONFIDENCE
                        # ========================================

                        confidence = float(
                            box.conf[0]
                            .detach()
                            .cpu()
                            .item()
                        )

                        # ========================================
                        # CLASS
                        # ========================================

                        class_id = int(
                            box.cls[0]
                            .detach()
                            .cpu()
                            .item()
                        )

                        class_name = get_class_name(
                            class_id
                        )

                        # ========================================
                        # TRACK ID
                        # ========================================

                        track_id: Optional[int] = None

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

                        # ========================================
                        # OBJECT WIDTH
                        # ========================================

                        pixel_width = max(
                            1,
                            x2 - x1
                        )

                        # ========================================
                        # DISTANCE
                        # ========================================

                        try:

                            distance = float(
                                estimate_distance(
                                    class_name,
                                    pixel_width
                                )
                            )

                        except Exception as error:

                            print(
                                f"[FRAME {frame_number}] "
                                f"Distance estimation error "
                                f"for {class_name}:",
                                repr(error)
                            )

                            distance = 0.0

                        # ========================================
                        # SPEED
                        # ========================================

                        speed = 0.0

                        approaching = False

                        ttc = None

                        # ========================================
                        # TRACK HISTORY
                        # ========================================

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

                                    # Positive means object
                                    # is getting closer.

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

                                    # =================================
                                    # TIME TO COLLISION
                                    # =================================

                                    if (
                                        approaching
                                        and speed > 0.1
                                        and distance > 0
                                    ):

                                        ttc = (
                                            distance
                                            / speed
                                        )

                            # Save current track state.

                            track_history[track_id] = {

                                "distance": distance,

                                "time": current_time,

                            }

                        # ========================================
                        # RISK
                        # ========================================

                        try:

                            risk = calculate_risk(
                                distance,
                                ttc
                            )

                        except Exception as error:

                            print(
                                f"[FRAME {frame_number}] "
                                f"Risk calculation error:",
                                repr(error)
                            )

                            risk = "SAFE"

                        # ========================================
                        # CENTER
                        # ========================================

                        center_x = (
                            x1 + x2
                        ) / 2

                        center_y = (
                            y1 + y2
                        ) / 2

                        # ========================================
                        # RELATIVE POSITION
                        # ========================================

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

                        # ========================================
                        # 2.5D MAP X
                        # ========================================

                        map_x = (
                            (
                                center_x / width
                            ) * 2
                            - 1
                            if width > 0
                            else 0
                        )

                        # ========================================
                        # OBJECT DATA
                        # ========================================

                        objects.append({

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

                            "width": x2 - x1,

                            "height": y2 - y1,

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

                            "approaching": approaching,

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

                        })

                    except Exception as error:

                        print(
                            f"[FRAME {frame_number}] "
                            f"Object processing error "
                            f"at index {i}:",
                            repr(error)
                        )

                        continue

            else:

                print(
                    f"[FRAME {frame_number}] "
                    "YOLO DETECTIONS: 0"
                )

            # ====================================================
            # LIMIT TRACK HISTORY
            # ====================================================

            if len(track_history) > MAX_TRACK_HISTORY:

                active_ids = set(
                    current_track_ids
                )

                old_ids = list(
                    track_history.keys()
                )

                for track_id in old_ids:

                    if track_id not in active_ids:

                        try:
                            del track_history[track_id]
                        except KeyError:
                            pass

                        if len(track_history) <= (
                            MAX_TRACK_HISTORY * 0.8
                        ):
                            break

            # ====================================================
            # COUNTS
            # ====================================================

            danger_count = sum(

                1

                for obj in objects

                if str(
                    obj["risk"]
                ).upper() == "DANGER"

            )

            warning_count = sum(

                1

                for obj in objects

                if str(
                    obj["risk"]
                ).upper() == "WARNING"

            )

            safe_count = sum(

                1

                for obj in objects

                if str(
                    obj["risk"]
                ).upper() == "SAFE"

            )

            # ====================================================
            # FPS
            # ====================================================

            now = time.time()

            elapsed = (
                now
                - previous_frame_time
            )

            fps = (

                1 / elapsed

                if elapsed > 0

                else 0

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
            # ENVIRONMENT MAP
            # ====================================================

            map_objects = []

            for obj in objects:

                map_objects.append({

                    "id": obj["id"],

                    "class": obj["class"],

                    "x": obj["map_x"],

                    "distance": obj[
                        "map_distance"
                    ],

                    "risk": obj["risk"],

                    "approaching": obj[
                        "approaching"
                    ],

                })

            # ====================================================
            # LIDAR
            # ====================================================

            lidar_data = get_latest_lidar()

            # ====================================================
            # TOTAL PROCESSING TIME
            # ====================================================

            total_processing_time = (
                time.time()
                - frame_start_time
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

                "lidar": lidar_data,

                "system": {

                    "device": str(DEVICE),

                    "model": "YOLO",

                    "tracker": "ByteTrack",

                    "adaptive_resolution": True,

                    "frame": frame_number,

                    "yolo_time": round(
                        yolo_time,
                        3
                    ),

                    "processing_time": round(
                        total_processing_time,
                        3
                    ),

                },

            }

            # ====================================================
            # DEBUG SUMMARY
            # ====================================================

            print(
                f"[FRAME {frame_number}] "
                f"RESULT | "
                f"objects={len(objects)} | "
                f"danger={danger_count} | "
                f"warning={warning_count} | "
                f"safe={safe_count} | "
                f"fps={fps:.1f} | "
                f"next_imgsz={current_inference_size} | "
                f"total_time={total_processing_time:.3f}s"
            )

            # ====================================================
            # SEND RESPONSE
            # ====================================================

            try:

                await websocket.send_json(
                    response
                )

                print(
                    f"[FRAME {frame_number}] "
                    "SENDING RESPONSE: "
                    f"{len(objects)} objects"
                )

            except WebSocketDisconnect:

                raise

            except Exception as error:

                print(
                    f"[FRAME {frame_number}] "
                    f"Response send error:",
                    repr(error)
                )

                break

    # ============================================================
    # FRONTEND DISCONNECT
    # ============================================================

    except WebSocketDisconnect:

        print("=" * 70)
        print("Frontend disconnected")
        print("Client:", client_host)
        print("Frames processed:", frame_number)
        print("=" * 70)

    # ============================================================
    # UNEXPECTED WEBSOCKET ERROR
    # ============================================================

    except Exception as error:

        print("=" * 70)
        print("WebSocket error")
        print("Client:", client_host)
        print("Frames processed:", frame_number)
        print("ERROR:", repr(error))
        print("=" * 70)