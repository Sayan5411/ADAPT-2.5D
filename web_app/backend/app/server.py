# ============================================================
# ADAPTIVE 2.5D LiDAR MAPPING
# REAL-TIME DETECTION WEBSOCKET BACKEND
# ============================================================
#
# FastAPI + WebSocket
# YOLO11n + ByteTrack
# Distance Estimation
# Speed / Approach Detection
# TTC Calculation
# Risk Classification
# Adaptive Inference Resolution
# 2.5D Environment Mapping
# LiDAR Pipeline
#
# ============================================================

import asyncio
import base64
import binascii
import time
from typing import Any, Dict, Optional

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
# APPLICATION
# ============================================================

app = FastAPI(
    title="Adaptive 2.5D LiDAR Mapping API",
    version="1.0.0",
    description=(
        "Real-time adaptive 2.5D environment perception "
        "using YOLO, ByteTrack and LiDAR."
    ),
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SERVER_START_TIME = time.time()

# Current inference resolution.
#
# Starts at normal resolution.
# It can automatically increase when the environment becomes
# more complex or dangerous.
current_inference_size = NORMAL_IMAGE_SIZE


# ============================================================
# YOLO MODEL
# ============================================================

print()
print("=" * 70)
print("STARTING ADAPTIVE 2.5D LIDAR MAPPING BACKEND")
print("=" * 70)

print()
print("Loading YOLO model...")
print("Model path:", MODEL_PATH)

try:
    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully")

except Exception as error:
    print()
    print("=" * 70)
    print("ERROR: YOLO MODEL FAILED TO LOAD")
    print("=" * 70)
    print("Model path:", MODEL_PATH)
    print("Error:", repr(error))
    print("=" * 70)
    raise


# ============================================================
# DEVICE
# ============================================================

print()
print("Inference device:", DEVICE)

if torch.cuda.is_available():
    print("CUDA GPU detected")
else:
    print("CUDA GPU not detected - using CPU")


# ============================================================
# LIDAR
# ============================================================

try:
    lidar_pipeline = LiDARPipeline()

    print("LiDAR pipeline initialized successfully")

except Exception as error:
    print("WARNING: LiDAR pipeline initialization failed")
    print("Error:", repr(error))

    lidar_pipeline = None


# ============================================================
# TRACK HISTORY
# ============================================================
#
# track_id -> {
#     "distance": float,
#     "time": float,
# }
#
# Used for:
#
# distance change
#       ↓
# speed
#       ↓
# approaching
#       ↓
# TTC
#       ↓
# risk
#
# ============================================================

track_history: Dict[int, Dict[str, float]] = {}


# ============================================================
# YOLO INFERENCE LOCK
# ============================================================
#
# Important for Render.
#
# Multiple browser/WebSocket connections can arrive at the
# backend. YOLO inference is CPU-heavy and the same model should
# not be executed simultaneously by multiple connections.
#
# ============================================================

inference_lock = asyncio.Lock()


# ============================================================
# ACTIVE CONNECTION COUNTER
# ============================================================

active_detection_connections = 0
active_lidar_connections = 0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int.
    """

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def decode_base64_image(data: str) -> Optional[np.ndarray]:
    """
    Convert a base64/data-URL image into an OpenCV frame.
    """

    if not data:
        return None

    try:

        # --------------------------------------------------------
        # Remove data URL prefix
        # --------------------------------------------------------

        if "," in data:
            data = data.split(",", 1)[1]

        # --------------------------------------------------------
        # Decode base64
        # --------------------------------------------------------

        image_bytes = base64.b64decode(
            data,
            validate=True,
        )

        if not image_bytes:
            return None

        # --------------------------------------------------------
        # Bytes -> NumPy
        # --------------------------------------------------------

        np_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        # --------------------------------------------------------
        # NumPy -> OpenCV
        # --------------------------------------------------------

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR,
        )

        return frame

    except (
        ValueError,
        TypeError,
        binascii.Error,
        cv2.error,
    ):
        return None


def calculate_object_motion(
    track_id: Optional[int],
    distance: float,
    current_time: float,
):
    """
    Calculate speed, approaching state and TTC.

    Positive speed means the object is getting closer.
    """

    speed = 0.0
    approaching = False
    ttc = None

    if track_id is None:
        return speed, approaching, ttc

    previous = track_history.get(track_id)

    if previous is not None:

        previous_distance = safe_float(
            previous.get("distance"),
            distance,
        )

        previous_time = safe_float(
            previous.get("time"),
            current_time,
        )

        delta_time = current_time - previous_time

        if delta_time > 0:

            # Positive value means the object is approaching.
            distance_change = (
                previous_distance - distance
            )

            speed = (
                distance_change / delta_time
            )

            # Ignore tiny movements/noise.
            if speed > 0.15:
                approaching = True

            # TTC only makes sense for an approaching object.
            if approaching and speed > 0.1:

                ttc = distance / speed

                # Protect against invalid values.
                if ttc < 0:
                    ttc = None

                # Ignore unrealistic values.
                elif ttc > 999:
                    ttc = None

    # ------------------------------------------------------------
    # Save current state
    # ------------------------------------------------------------

    track_history[track_id] = {
        "distance": distance,
        "time": current_time,
    }

    return speed, approaching, ttc


def cleanup_track_history(
    current_track_ids: set,
    current_time: float,
    max_age: float = 5.0,
):
    """
    Remove old tracking information.

    This prevents track_history from growing forever.
    """

    expired_ids = []

    for track_id, state in track_history.items():

        last_time = safe_float(
            state.get("time"),
            current_time,
        )

        # Remove tracks that have not appeared recently.
        if (
            track_id not in current_track_ids
            and current_time - last_time > max_age
        ):
            expired_ids.append(track_id)

    for track_id in expired_ids:
        track_history.pop(track_id, None)


def choose_next_inference_size(
    danger_count: int,
    warning_count: int,
    object_count: int,
) -> int:
    """
    Adaptive inference resolution.

    High resolution is used when:
    - danger exists
    - warning exists
    - many objects are present

    Otherwise normal resolution is used.
    """

    if danger_count > 0:
        return HIGH_IMAGE_SIZE

    if warning_count > 0:
        return HIGH_IMAGE_SIZE

    if object_count >= 6:
        return HIGH_IMAGE_SIZE

    return NORMAL_IMAGE_SIZE


def get_lidar_state() -> Dict[str, Any]:
    """
    Return the latest LiDAR information.

    Keeps the API stable even if the LiDAR pipeline is not
    connected or has no data yet.
    """

    default_state = {
        "connected": False,
        "points": [],
        "point_count": 0,
        "grid": [],
        "elevation": [],
        "traversability": [],
    }

    if lidar_pipeline is None:
        return default_state

    try:

        latest = lidar_pipeline.latest

        if latest is None:
            return default_state

        if isinstance(latest, dict):

            state = default_state.copy()
            state.update(latest)

            return state

        return default_state

    except Exception:
        return default_state


# ============================================================
# ROOT ENDPOINT
# ============================================================


@app.get("/")
def root():

    uptime = time.time() - SERVER_START_TIME

    return {
        "status": "running",
        "message": (
            "Adaptive 2.5D LiDAR Mapping Backend "
            "is working!"
        ),
        "model": "YOLO11n",
        "tracking": "ByteTrack",
        "device": str(DEVICE),
        "endpoint": "/ws/detection",
        "lidar_endpoint": "/ws/lidar",
        "active_detection_connections": (
            active_detection_connections
        ),
        "active_lidar_connections": (
            active_lidar_connections
        ),
        "inference_size": current_inference_size,
        "uptime_seconds": round(uptime, 1),
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
        "lidar_available": lidar_pipeline is not None,
        "active_detection_connections": (
            active_detection_connections
        ),
    }


# ============================================================
# LIDAR WEBSOCKET
# ============================================================


@app.websocket("/ws/lidar")
async def lidar_websocket(
    websocket: WebSocket,
):

    global active_lidar_connections

    await websocket.accept()

    active_lidar_connections += 1

    client_host = (
        websocket.client.host
        if websocket.client
        else "unknown"
    )

    print()
    print("=" * 70)
    print("LiDAR WebSocket connected")
    print("Client:", client_host)
    print(
        "Active LiDAR connections:",
        active_lidar_connections,
    )
    print("=" * 70)

    try:

        while True:

            # ----------------------------------------------------
            # Receive JSON
            # ----------------------------------------------------

            payload = await websocket.receive_json()

            if not isinstance(payload, dict):

                await websocket.send_json({
                    "error": (
                        "Expected a JSON object"
                    )
                })

                continue

            # ----------------------------------------------------
            # Check points
            # ----------------------------------------------------

            if "points" not in payload:

                await websocket.send_json({
                    "error": (
                        "Expected a 'points' array"
                    )
                })

                continue

            if lidar_pipeline is None:

                await websocket.send_json({
                    "error": (
                        "LiDAR pipeline unavailable"
                    ),
                    "connected": False,
                })

                continue

            # ----------------------------------------------------
            # Process LiDAR
            # ----------------------------------------------------

            try:

                result = (
                    lidar_pipeline.process_frame(
                        payload["points"],
                        payload.get("labels"),
                    )
                )

            except (
                TypeError,
                ValueError,
            ) as error:

                await websocket.send_json({
                    "error": str(error),
                })

                continue

            # ----------------------------------------------------
            # Send result
            # ----------------------------------------------------

            await websocket.send_json(result)

    except WebSocketDisconnect:

        print("LiDAR client disconnected")

    except Exception as error:

        print()
        print("=" * 70)
        print("LiDAR WebSocket error")
        print("Client:", client_host)
        print("ERROR:", repr(error))
        print("=" * 70)

    finally:

        active_lidar_connections = max(
            0,
            active_lidar_connections - 1,
        )

        if lidar_pipeline is not None:

            try:
                lidar_pipeline.disconnect()

            except Exception:
                pass

        print(
            "Active LiDAR connections:",
            active_lidar_connections,
        )


# ============================================================
# DETECTION WEBSOCKET
# ============================================================


@app.websocket("/ws/detection")
async def detection_websocket(
    websocket: WebSocket,
):

    global current_inference_size
    global active_detection_connections

    # ------------------------------------------------------------
    # Accept WebSocket
    # ------------------------------------------------------------

    await websocket.accept()

    active_detection_connections += 1

    client_host = (
        websocket.client.host
        if websocket.client
        else "unknown"
    )

    connection_start = time.time()

    frame_number = 0

    previous_frame_time = time.time()

    print()
    print("=" * 70)
    print("FRONTEND DETECTION CLIENT CONNECTED")
    print("Client:", client_host)
    print(
        "Active detection connections:",
        active_detection_connections,
    )
    print("=" * 70)

    try:

        while True:

            # ====================================================
            # RECEIVE FRAME
            # ====================================================

            receive_start = time.time()

            try:

                message = await websocket.receive()

            except WebSocketDisconnect:

                raise

            except Exception as error:

                print(
                    "Receive error:",
                    repr(error),
                )

                break

            # ----------------------------------------------------
            # Determine message type
            # ----------------------------------------------------

            data = None

            if "text" in message:

                data = message["text"]

            elif "bytes" in message:

                # ------------------------------------------------
                # We also support raw image bytes.
                # ------------------------------------------------

                raw_bytes = message["bytes"]

                try:

                    np_array = np.frombuffer(
                        raw_bytes,
                        dtype=np.uint8,
                    )

                    frame = cv2.imdecode(
                        np_array,
                        cv2.IMREAD_COLOR,
                    )

                except Exception:

                    frame = None

            else:

                frame = None

            # ====================================================
            # TEXT / BASE64 IMAGE
            # ====================================================

            if data is not None:

                frame = decode_base64_image(data)

            # ====================================================
            # VALIDATE FRAME
            # ====================================================

            if frame is None:

                print(
                    f"[FRAME {frame_number + 1}] "
                    "Invalid image received"
                )

                try:

                    await websocket.send_json({
                        "error": "Invalid image frame",
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
                        "lidar": get_lidar_state(),
                    })

                except Exception:
                    pass

                continue

            # ====================================================
            # FRAME NUMBER
            # ====================================================

            frame_number += 1

            receive_time = time.time()

            # ====================================================
            # IMAGE INFORMATION
            # ====================================================

            height, width = frame.shape[:2]

            print(
                f"[FRAME {frame_number}] "
                f"received {width}x{height} "
                f"in "
                f"{(receive_time - receive_start) * 1000:.1f} ms"
            )

            # ====================================================
            # SAVE THE RESOLUTION ACTUALLY USED
            # ====================================================

            inference_size_used = (
                current_inference_size
            )

            # ====================================================
            # YOLO + BYTE TRACK
            # ====================================================
            #
            # The lock prevents multiple WebSocket clients from
            # running the same YOLO model simultaneously.
            #
            # This is particularly important on Render CPU.
            #
            # ====================================================

            inference_start = time.time()

            async with inference_lock:

                try:

                    results = model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=CONFIDENCE_THRESHOLD,
                        imgsz=inference_size_used,
                        device=DEVICE,
                        verbose=False,
                    )

                except Exception as error:

                    print()
                    print("=" * 70)
                    print(
                        f"YOLO ERROR - FRAME {frame_number}"
                    )
                    print("ERROR:", repr(error))
                    print("=" * 70)

                    try:

                        await websocket.send_json({
                            "error": (
                                "YOLO inference failed"
                            ),
                            "objects": [],
                            "counts": {
                                "danger": 0,
                                "warning": 0,
                                "safe": 0,
                                "total": 0,
                            },
                            "fps": 0,
                            "inference_size": (
                                inference_size_used
                            ),
                            "map": [],
                            "lidar": (
                                get_lidar_state()
                            ),
                        })

                    except Exception:
                        pass

                    continue

            inference_time = time.time() - inference_start

            # ====================================================
            # DETECTION PROCESSING
            # ====================================================

            objects = []

            current_track_ids = set()

            current_time = time.time()

            # ----------------------------------------------------
            # Check YOLO result
            # ----------------------------------------------------

            if (
                results
                and len(results) > 0
                and results[0].boxes is not None
            ):

                boxes = results[0].boxes

                # ------------------------------------------------
                # Iterate detections
                # ------------------------------------------------

                for i in range(len(boxes)):

                    try:

                        box = boxes[i]

                        # =================================================
                        # BOUNDING BOX
                        # =================================================

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
                                int(x1),
                                width - 1,
                            ),
                        )

                        y1 = max(
                            0,
                            min(
                                int(y1),
                                height - 1,
                            ),
                        )

                        x2 = max(
                            0,
                            min(
                                int(x2),
                                width - 1,
                            ),
                        )

                        y2 = max(
                            0,
                            min(
                                int(y2),
                                height - 1,
                            ),
                        )

                        # Ignore invalid boxes.
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # =================================================
                        # CONFIDENCE
                        # =================================================

                        confidence = safe_float(
                            box.conf[0]
                            .detach()
                            .cpu()
                            .item()
                        )

                        # =================================================
                        # CLASS ID
                        # =================================================

                        class_id = safe_int(
                            box.cls[0]
                            .detach()
                            .cpu()
                            .item()
                        )

                        # =================================================
                        # CLASS NAME
                        # =================================================

                        if (
                            0 <= class_id
                            < len(CLASS_NAMES)
                        ):

                            class_name = (
                                CLASS_NAMES[class_id]
                            )

                        else:

                            class_name = "Unknown"

                        # =================================================
                        # TRACK ID
                        # =================================================

                        track_id = None

                        if boxes.id is not None:

                            try:

                                track_id = safe_int(
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

                        # =================================================
                        # OBJECT WIDTH
                        # =================================================

                        pixel_width = max(
                            1,
                            x2 - x1,
                        )

                        # =================================================
                        # DISTANCE
                        # =================================================

                        try:

                            distance = safe_float(
                                estimate_distance(
                                    class_name,
                                    pixel_width,
                                ),
                                0.0,
                            )

                        except Exception as error:

                            print(
                                "Distance estimation error:",
                                repr(error),
                            )

                            distance = 0.0

                        # =================================================
                        # SPEED / APPROACHING / TTC
                        # =================================================

                        (
                            speed,
                            approaching,
                            ttc,
                        ) = calculate_object_motion(
                            track_id,
                            distance,
                            current_time,
                        )

                        # =================================================
                        # RISK
                        # =================================================

                        try:

                            risk = calculate_risk(
                                distance,
                                ttc,
                            )

                        except Exception as error:

                            print(
                                "Risk calculation error:",
                                repr(error),
                            )

                            risk = "SAFE"

                        # =================================================
                        # OBJECT CENTER
                        # =================================================

                        center_x = (
                            x1 + x2
                        ) / 2.0

                        center_y = (
                            y1 + y2
                        ) / 2.0

                        # =================================================
                        # RELATIVE POSITION
                        # =================================================

                        relative_x = (
                            center_x / width
                            if width > 0
                            else 0.5
                        )

                        relative_y = (
                            center_y / height
                            if height > 0
                            else 0.5
                        )

                        # =================================================
                        # 2.5D MAP X
                        # =================================================
                        #
                        # -1 = left
                        #  0 = center
                        # +1 = right
                        #
                        # =================================================

                        map_x = (
                            relative_x * 2.0
                        ) - 1.0

                        # =================================================
                        # OBJECT DATA
                        # =================================================

                        object_data = {

                            "id": track_id,

                            "class": class_name,

                            "class_id": class_id,

                            "confidence": round(
                                confidence,
                                3,
                            ),

                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,

                            "width": (
                                x2 - x1
                            ),

                            "height": (
                                y2 - y1
                            ),

                            "center_x": round(
                                center_x,
                                1,
                            ),

                            "center_y": round(
                                center_y,
                                1,
                            ),

                            "distance": round(
                                distance,
                                2,
                            ),

                            "speed": round(
                                speed,
                                2,
                            ),

                            "ttc": (
                                round(
                                    ttc,
                                    2,
                                )
                                if ttc is not None
                                else None
                            ),

                            "approaching": (
                                approaching
                            ),

                            "risk": risk,

                            "rel_x": round(
                                relative_x,
                                3,
                            ),

                            "rel_y": round(
                                relative_y,
                                3,
                            ),

                            "map_x": round(
                                map_x,
                                3,
                            ),

                            "map_distance": round(
                                distance,
                                2,
                            ),
                        }

                        objects.append(
                            object_data
                        )

                    except Exception as error:

                        print(
                            f"Object processing error "
                            f"at index {i}:",
                            repr(error),
                        )

                        continue

            # ====================================================
            # CLEAN OLD TRACKS
            # ====================================================

            cleanup_track_history(
                current_track_ids,
                current_time,
            )

            # ====================================================
            # COUNTS
            # ====================================================

            danger_count = sum(
                1
                for obj in objects
                if obj.get("risk") == "DANGER"
            )

            warning_count = sum(
                1
                for obj in objects
                if obj.get("risk") == "WARNING"
            )

            safe_count = sum(
                1
                for obj in objects
                if obj.get("risk") == "SAFE"
            )

            total_objects = len(objects)

            # ====================================================
            # FPS
            # ====================================================

            now = time.time()

            elapsed = (
                now - previous_frame_time
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

            next_inference_size = (
                choose_next_inference_size(
                    danger_count,
                    warning_count,
                    total_objects,
                )
            )

            current_inference_size = (
                next_inference_size
            )

            # ====================================================
            # ENVIRONMENT MAP
            # ====================================================

            map_objects = []

            for obj in objects:

                map_objects.append({

                    "id": obj.get("id"),

                    "class": obj.get(
                        "class",
                        "Unknown",
                    ),

                    "x": obj.get(
                        "map_x",
                        0,
                    ),

                    "distance": obj.get(
                        "map_distance",
                        0,
                    ),

                    "risk": obj.get(
                        "risk",
                        "SAFE",
                    ),

                    "approaching": obj.get(
                        "approaching",
                        False,
                    ),
                })

            # ====================================================
            # LIDAR STATE
            # ====================================================

            lidar_state = get_lidar_state()

            # ====================================================
            # TOTAL FRAME PROCESSING TIME
            # ====================================================

            total_processing_time = (
                time.time() - receive_time
            )

            # ====================================================
            # RESPONSE
            # ====================================================

            response = {

                # ------------------------------------------------
                # Objects
                # ------------------------------------------------

                "objects": objects,

                # ------------------------------------------------
                # Counts
                # ------------------------------------------------

                "counts": {

                    "danger": danger_count,

                    "warning": warning_count,

                    "safe": safe_count,

                    "total": total_objects,
                },

                # ------------------------------------------------
                # Performance
                # ------------------------------------------------

                "fps": round(
                    fps,
                    1,
                ),

                "inference_size": (
                    inference_size_used
                ),

                # ------------------------------------------------
                # Image
                # ------------------------------------------------

                "image": {

                    "width": width,

                    "height": height,
                },

                # ------------------------------------------------
                # Camera
                # ------------------------------------------------

                "camera": {

                    "width": width,

                    "height": height,

                    "connected": True,
                },

                # ------------------------------------------------
                # 2.5D map
                # ------------------------------------------------

                "map": map_objects,

                # ------------------------------------------------
                # LiDAR
                # ------------------------------------------------

                "lidar": lidar_state,

                # ------------------------------------------------
                # Performance diagnostics
                # ------------------------------------------------

                "performance": {

                    "inference_ms": round(
                        inference_time * 1000,
                        1,
                    ),

                    "processing_ms": round(
                        total_processing_time
                        * 1000,
                        1,
                    ),

                    "frame": frame_number,
                },

                # ------------------------------------------------
                # System
                # ------------------------------------------------

                "system": {

                    "device": str(
                        DEVICE
                    ),

                    "model": "YOLO11n",

                    "tracker": "ByteTrack",

                    "adaptive_resolution": True,

                    "backend": "FastAPI",

                    "websocket": True,
                },
            }

            # ====================================================
            # SEND RESULT
            # ====================================================

            try:

                await websocket.send_json(
                    response
                )

            except WebSocketDisconnect:

                raise

            except Exception as error:

                print(
                    "Failed to send response:",
                    repr(error),
                )

                break

            # ====================================================
            # SERVER LOG
            # ====================================================

            if (
                frame_number <= 5
                or frame_number % 10 == 0
            ):

                print(
                    f"[FRAME {frame_number}] "
                    f"objects={total_objects} | "
                    f"D={danger_count} | "
                    f"W={warning_count} | "
                    f"S={safe_count} | "
                    f"FPS={fps:.1f} | "
                    f"YOLO={inference_time * 1000:.0f}ms | "
                    f"size={inference_size_used}"
                )

    # ============================================================
    # CLIENT DISCONNECT
    # ============================================================

    except WebSocketDisconnect:

        connection_duration = (
            time.time()
            - connection_start
        )

        print()
        print("=" * 70)
        print("FRONTEND DISCONNECTED")
        print("Client:", client_host)
        print(
            "Frames processed:",
            frame_number,
        )
        print(
            "Connection duration:",
            round(connection_duration, 1),
            "seconds",
        )
        print("=" * 70)

    # ============================================================
    # GENERAL ERROR
    # ============================================================

    except Exception as error:

        print()
        print("=" * 70)
        print("WEBSOCKET ERROR")
        print("Client:", client_host)
        print(
            "Frames processed:",
            frame_number,
        )
        print(
            "ERROR:",
            repr(error),
        )
        print("=" * 70)

    # ============================================================
    # FINAL CLEANUP
    # ============================================================

    finally:

        active_detection_connections = max(
            0,
            active_detection_connections - 1,
        )

        print(
            "Active detection connections:",
            active_detection_connections,
        )


# ============================================================
# STARTUP / SHUTDOWN EVENTS
# ============================================================


@app.on_event("startup")
async def startup_event():

    print()
    print("=" * 70)
    print("APPLICATION STARTUP COMPLETE")
    print("=" * 70)

    print(
        "Detection endpoint:",
        "/ws/detection",
    )

    print(
        "LiDAR endpoint:",
        "/ws/lidar",
    )

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Normal inference size:",
        NORMAL_IMAGE_SIZE,
    )

    print(
        "High inference size:",
        HIGH_IMAGE_SIZE,
    )

    print("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():

    print()
    print("=" * 70)
    print("ADAPTIVE 2.5D LIDAR MAPPING BACKEND SHUTDOWN")
    print("=" * 70)

    print(
        "Active detection connections:",
        active_detection_connections,
    )

    print(
        "Active LiDAR connections:",
        active_lidar_connections,
    )

    print("=" * 70)