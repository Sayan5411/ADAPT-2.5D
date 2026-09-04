import numpy as np


def create_2_5d_map(
    points,
    resolution=0.20,
    x_range=(-40, 40),
    y_range=(0, 80)
):
    """
    Creates a bird's-eye-view 2.5D elevation map.

    X = left/right
    Y = forward distance
    Z = height
    """

    if len(points) == 0:
        return None

    mask = (
        (points[:, 0] >= x_range[0]) &
        (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) &
        (points[:, 1] <= y_range[1])
    )

    points = points[mask]

    width = int(
        (x_range[1] - x_range[0]) / resolution
    )

    height = int(
        (y_range[1] - y_range[0]) / resolution
    )

    grid = np.full(
        (height, width),
        np.nan,
        dtype=np.float32
    )

    for point in points:

        x, y, z = point[:3]

        col = int(
            (x - x_range[0]) / resolution
        )

        row = int(
            (y - y_range[0]) / resolution
        )

        if 0 <= row < height and 0 <= col < width:

            if np.isnan(grid[row, col]):
                grid[row, col] = z
            else:
                grid[row, col] = max(
                    grid[row, col],
                    z
                )

    return grid