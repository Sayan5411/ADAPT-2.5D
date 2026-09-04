from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mapping.adaptive_resolution import adaptive_resolution
from mapping.map_2_5d import create_2_5d_map
from mapping.semantic_map import build_semantic_2_5d_map
from preprocessing.pointcloud import filter_point_cloud


EMPTY_LIDAR = {
    "connected": False,
    "points": [],
    "point_count": 0,
    "labels_available": False,
    "grid": [],
    "elevation": [],
    "traversability": [],
}


class LiDARPipeline:
    def __init__(self, min_range: float = 0.5, max_range: float = 100.0):
        self.min_range = min_range
        self.max_range = max_range
        self.latest = dict(EMPTY_LIDAR)

    def process_frame(self, raw_points: Any, raw_labels: Any = None):
        points = np.asarray(raw_points, dtype=np.float32)
        if points.size == 0:
            raise ValueError("LiDAR frame contains no points")
        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError("LiDAR points must have shape Nx3 or Nx4")

        points = points[:, :3]
        labels = self._normalize_labels(raw_labels, len(points))
        distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        valid = (
            np.isfinite(points).all(axis=1)
            & (distances >= self.min_range)
            & (distances <= self.max_range)
            & (points[:, 2] >= -2.0)
            & (points[:, 2] <= 10.0)
        )
        points = points[valid]
        if labels is not None:
            labels = labels[valid]

        points = filter_point_cloud(
            points,
            self.min_range,
            self.max_range,
        )

        points, labels = self._apply_adaptive_resolution(points, labels)
        cells = build_semantic_2_5d_map(points, labels)
        elevation = create_2_5d_map(points)

        result = {
            "connected": True,
            "points": self._serialize_points(points),
            "point_count": int(len(points)),
            "labels_available": labels is not None,
            "grid": self._serialize_cells(cells),
            "elevation": self._serialize_elevation(elevation),
            "traversability": self._serialize_traversability(cells),
        }
        self.latest = result
        return result

    def disconnect(self):
        self.latest = dict(EMPTY_LIDAR)

    @staticmethod
    def _normalize_labels(raw_labels, point_count):
        if raw_labels is None:
            return None
        labels = np.asarray(raw_labels, dtype=np.int64)
        if labels.ndim != 1 or len(labels) != point_count:
            raise ValueError("LiDAR labels must have one value per point")
        return labels

    @staticmethod
    def _apply_adaptive_resolution(points, labels):
        if labels is None:
            return adaptive_resolution(points), None

        distances = np.sqrt(np.sum(points ** 2, axis=1))
        selected = np.concatenate([
            np.flatnonzero(distances < 20),
            np.flatnonzero((distances >= 20) & (distances < 40))[::2],
            np.flatnonzero((distances >= 40) & (distances < 80))[::4],
        ])
        return points[selected], labels[selected]

    @staticmethod
    def _serialize_points(points):
        return np.asarray(points, dtype=np.float32).round(4).tolist()

    @staticmethod
    def _serialize_cells(cells):
        serialized = []
        for cell in cells.values():
            serialized.append({
                "x": round(float(cell["center_x"]), 4),
                "y": round(float(cell["center_y"]), 4),
                "z": round(float(cell["mean_z"]), 4),
                "resolution": float(cell["resolution"]),
                "count": int(cell["count"]),
                "dominant_label": int(cell["dominant_label"]),
                "roughness": round(float(cell["roughness"]), 4),
                "obstacle": bool(cell["obstacle"]),
                "traversability": round(float(cell["traversability"]), 2),
                "slope_deg": round(float(cell["slope_deg"]), 2),
            })
        return serialized

    @staticmethod
    def _serialize_elevation(elevation):
        if elevation is None:
            return []
        rows, columns = np.where(np.isfinite(elevation))
        return [
            {"row": int(row), "column": int(column), "z": round(float(elevation[row, column]), 4)}
            for row, column in zip(rows, columns)
        ]

    @staticmethod
    def _serialize_traversability(cells):
        return [
            {
                "x": round(float(cell["center_x"]), 4),
                "y": round(float(cell["center_y"]), 4),
                "z": round(float(cell["mean_z"]), 4),
                "obstacle": bool(cell["obstacle"]),
                "score": round(float(cell["traversability"]), 2),
                "slope_deg": round(float(cell["slope_deg"]), 2),
                "roughness": round(float(cell["roughness"]), 4),
            }
            for cell in cells.values()
        ]
