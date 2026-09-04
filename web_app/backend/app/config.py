import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent

PROJECT_ROOT = BASE_DIR.parent.parent

MODEL_PATH = Path(
    os.getenv(
        "LIDAR_MODEL_PATH",
        str(BASE_DIR / "models" / "best.pt")
    )
)

FOCAL_LENGTH_FILE = Path(
    os.getenv(
        "LIDAR_FOCAL_LENGTH_FILE",
        str(PROJECT_ROOT / "object_detection" / "focal_length.txt")
    )
)

DEVICE = 0 if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc"
]

CLASS_WIDTHS = {
    "Car": 1.80,
    "Van": 2.00,
    "Truck": 2.50,
    "Pedestrian": 0.50,
    "Person_sitting": 0.50,
    "Cyclist": 0.60,
    "Tram": 2.50,
    "Misc": 1.00,
}

CONFIDENCE_THRESHOLD = 0.40

DEFAULT_FOCAL_LENGTH = 700.0


def load_focal_length():

    try:
        return float(FOCAL_LENGTH_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return DEFAULT_FOCAL_LENGTH


FOCAL_LENGTH = load_focal_length()

NORMAL_IMAGE_SIZE = 640
HIGH_IMAGE_SIZE = 960

DANGER_DISTANCE = 4.0
WARNING_DISTANCE = 10.0