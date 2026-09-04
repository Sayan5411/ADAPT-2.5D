import numpy as np


def load_lidar_bin(file_path):
    """
    Load KITTI LiDAR .bin file.

    Each point:
    x, y, z, reflectance
    """

    points = np.fromfile(file_path, dtype=np.float32)

    points = points.reshape(-1, 4)

    return points


def remove_invalid_points(points):
    """
    Remove NaN and infinite points.
    """

    mask = np.isfinite(points).all(axis=1)

    return points[mask]


def filter_distance(points, min_distance=2.0, max_distance=80.0):
    """
    Keep points within useful detection range.
    """

    distance = np.sqrt(
        points[:, 0] ** 2 +
        points[:, 1] ** 2 +
        points[:, 2] ** 2
    )

    mask = (
        (distance >= min_distance) &
        (distance <= max_distance)
    )

    return points[mask]