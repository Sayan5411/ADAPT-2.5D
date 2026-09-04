import base64
import time

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
    title="Adaptive 2.5D LiDAR Mapping API"
)


# ============================================================
# YOLO MODEL
# ============================================================

print("=" * 60)
print("Loading YOLO model...")
print(MODEL_PATH)
print("=" * 60)

model = YOLO(MODEL_PATH)

print("Inference device:", DEVICE)

if torch.cuda.is_available():
    print("CUDA GPU detected")
else:
    print("CUDA GPU not detected - using CPU")


# ============================================================
# CLASS NAMES
# ============================================================

# ============================================================
# ADAPTIVE RESOLUTION
# ============================================================

# Current resolution used by inference
current_inference_size = NORMAL_IMAGE_SIZE


# ============================================================
# TRACK HISTORY
# ============================================================

# Stores:
# track_id -> {
#     distance,
#     time
# }
track_history = {}
lidar_pipeline = LiDARPipeline()


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
    }


@app.websocket("/ws/lidar")
async def lidar_websocket(websocket: WebSocket):
    await websocket.accept()
    print("LiDAR client connected")

    try:
        while True:
            payload = await websocket.receive_json()
            if not isinstance(payload, dict) or "points" not in payload:
                await websocket.send_json({"error": "Expected a JSON object with a points array"})
                continue

            try:
                result = lidar_pipeline.process_frame(
                    payload["points"],
                    payload.get("labels"),
                )
            except (TypeError, ValueError) as error:
                await websocket.send_json({"error": str(error)})
                continue

            await websocket.send_json(result)
    except WebSocketDisconnect:
        lidar_pipeline.disconnect()
        print("LiDAR client disconnected")
    except Exception as error:
        lidar_pipeline.disconnect()
        print("LiDAR websocket error:", repr(error))


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws/detection")
async def detection_websocket(
    websocket: WebSocket
):

    global current_inference_size

    await websocket.accept()

    print("Frontend connected")

    previous_frame_time = time.time()

    try:

        while True:

            # ====================================================
            # RECEIVE IMAGE
            # ====================================================

            data = await websocket.receive_text()

            # Remove data URL prefix
            if "," in data:
                data = data.split(",", 1)[1]

            try:

                image_bytes = base64.b64decode(data)

            except Exception:

                print("Invalid base64 image")
                continue

            # ====================================================
            # BASE64 -> OPENCV IMAGE
            # ====================================================

            np_array = np.frombuffer(
                image_bytes,
                np.uint8
            )

            frame = cv2.imdecode(
                np_array,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            height, width = frame.shape[:2]


            # ====================================================
            # YOLO + BYTE TRACK
            # ====================================================

            results = model.track(

                frame,

                persist=True,

                tracker="bytetrack.yaml",

                conf=CONFIDENCE_THRESHOLD,

                imgsz=current_inference_size,

                device=DEVICE,

                verbose=False
            )


            objects = []

            current_track_ids = set()

            current_time = time.time()


            # ====================================================
            # PROCESS DETECTIONS
            # ====================================================

            if (
                results
                and results[0].boxes is not None
            ):

                boxes = results[0].boxes


                for i in range(len(boxes)):

                    box = boxes[i]


                    # ------------------------------------------------
                    # BOUNDING BOX
                    # ------------------------------------------------

                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)


                    # ------------------------------------------------
                    # CONFIDENCE
                    # ------------------------------------------------

                    confidence = float(
                        box.conf[0]
                    )


                    # ------------------------------------------------
                    # CLASS
                    # ------------------------------------------------

                    class_id = int(
                        box.cls[0]
                    )

                    if (
                        class_id >= 0
                        and class_id < len(CLASS_NAMES)
                    ):

                        class_name = (
                            CLASS_NAMES[class_id]
                        )

                    else:

                        class_name = "Unknown"


                    # ------------------------------------------------
                    # TRACK ID
                    # ------------------------------------------------

                    track_id = None

                    if boxes.id is not None:

                        track_id = int(
                            boxes.id[i]
                            .item()
                        )

                        current_track_ids.add(
                            track_id
                        )


                    # ------------------------------------------------
                    # OBJECT WIDTH
                    # ------------------------------------------------

                    pixel_width = max(
                        1,
                        x2 - x1
                    )


                    # ------------------------------------------------
                    # DISTANCE
                    # ------------------------------------------------

                    distance = estimate_distance(

                        class_name,

                        pixel_width

                    )


                    # =================================================
                    # SPEED / APPROACHING / TTC
                    # =================================================

                    speed = 0.0

                    approaching = False

                    ttc = None


                    if track_id is not None:

                        previous = (
                            track_history.get(
                                track_id
                            )
                        )


                        if previous is not None:

                            previous_distance = (
                                previous["distance"]
                            )

                            previous_time = (
                                previous["time"]
                            )

                            delta_time = (
                                current_time
                                - previous_time
                            )


                            if delta_time > 0:

                                # Positive = approaching
                                distance_change = (
                                    previous_distance
                                    - distance
                                )

                                speed = (
                                    distance_change
                                    / delta_time
                                )


                                # Ignore tiny movements
                                if speed > 0.15:

                                    approaching = True


                                # TTC
                                if (
                                    approaching
                                    and speed > 0.1
                                ):

                                    ttc = (
                                        distance
                                        / speed
                                    )


                        # Save current state
                        track_history[
                            track_id
                        ] = {

                            "distance": distance,

                            "time": current_time

                        }


                    # =================================================
                    # RISK
                    # =================================================

                    risk = calculate_risk(

                        distance,

                        ttc

                    )


                    # =================================================
                    # OBJECT CENTER
                    # =================================================

                    center_x = (
                        x1 + x2
                    ) / 2

                    center_y = (
                        y1 + y2
                    ) / 2


                    # Relative horizontal position
                    relative_x = (
                        center_x / width
                    )


                    # Relative vertical position
                    relative_y = (
                        center_y / height
                    )


                    # =================================================
                    # 2.5D POSITION
                    # =================================================

                    # Normalize X:
                    #
                    # -1 = far left
                    #  0 = center
                    # +1 = far right

                    map_x = (
                        (center_x / width)
                        * 2
                        - 1
                    )


                    # =================================================
                    # OBJECT DATA
                    # =================================================

                    objects.append({

                        "id": track_id,

                        "class": class_name,

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
                        )
                    })


            # ====================================================
            # REMOVE OLD TRACKS
            # ====================================================

            old_ids = list(
                track_history.keys()
            )

            for track_id in old_ids:

                if (
                    track_id
                    not in current_track_ids
                ):

                    # Keep only recent tracks
                    # rather than deleting immediately.
                    pass


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

            #
            # If dangerous objects are detected,
            # increase the next inference resolution.
            #

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
                    ]

                })


            # ====================================================
            # RESPONSE
            # ====================================================

            response = {

                "objects": objects,

                "counts": {

                    "danger": danger_count,

                    "warning": warning_count,

                    "safe": safe_count,

                    "total": len(objects)

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

                    "height": height

                },

                "map": map_objects,

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

                    "adaptive_resolution": True

                }

            }


            # ====================================================
            # SEND TO FRONTEND
            # ====================================================

            await websocket.send_json(
                response
            )


    except WebSocketDisconnect:

        print(
            "Frontend disconnected"
        )


    except Exception as e:

        print(
            "WebSocket error:",
            repr(e)
        )