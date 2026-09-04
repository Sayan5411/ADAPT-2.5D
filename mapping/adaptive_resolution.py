import numpy as np


def calculate_distance(points):
    return np.sqrt(
        points[:, 0] ** 2 +
        points[:, 1] ** 2 +
        points[:, 2] ** 2
    )


def adaptive_resolution(points):
    """
    Adaptive point-cloud resolution.

    Near:
        0-20m -> keep all points

    Medium:
        20-40m -> keep every 2nd point

    Far:
        40-80m -> keep every 4th point
    """

    if len(points) == 0:
        return points

    distances = calculate_distance(points)

    near = points[distances < 20]

    medium = points[
        (distances >= 20) &
        (distances < 40)
    ]

    far = points[
        (distances >= 40) &
        (distances < 80)
    ]

    medium = medium[::2]
    far = far[::4]

    result = np.vstack([
        near,
        medium,
        far
    ])

    return result