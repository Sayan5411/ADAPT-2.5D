import numpy as np


def filter_point_cloud(
    points,
    min_range=0.5,
    max_range=100.0,
    z_min=-2.0,
    z_max=10.0
):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    distance = np.sqrt(
        points[:, 0] ** 2
        +
        points[:, 1] ** 2
    )

    mask = (
        (distance >= min_range)
        &
        (distance <= max_range)
        &
        (points[:, 2] >= z_min)
        &
        (points[:, 2] <= z_max)
        &
        np.isfinite(points).all(axis=1)
    )

    return points[mask]


def voxel_downsample(
    points,
    voxel_size=0.05
):

    if len(points) == 0:
        return points

    coordinates = np.floor(
        points / voxel_size
    ).astype(np.int64)

    _, indices = np.unique(
        coordinates,
        axis=0,
        return_index=True
    )

    return points[
        np.sort(indices)
    ]