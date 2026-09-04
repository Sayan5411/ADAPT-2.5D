import numpy as np
from config import CFG


def create_box(rng, center, size, number_of_points, label):

    cx, cy, cz = center

    sx, sy, sz = size

    x = rng.uniform(
        cx - sx / 2,
        cx + sx / 2,
        number_of_points
    )

    y = rng.uniform(
        cy - sy / 2,
        cy + sy / 2,
        number_of_points
    )

    z = rng.uniform(
        cz - sz / 2,
        cz + sz / 2,
        number_of_points
    )

    points = np.column_stack((x, y, z))

    labels = np.full(
        number_of_points,
        label,
        dtype=np.int64
    )

    return points, labels


def create_cylinder(
    rng,
    center,
    radius,
    height,
    number_of_points,
    label
):

    cx, cy, cz = center

    angle = rng.uniform(
        0,
        2 * np.pi,
        number_of_points
    )

    radius_values = radius * np.sqrt(
        rng.uniform(
            0,
            1,
            number_of_points
        )
    )

    x = (
        cx +
        radius_values * np.cos(angle)
    )

    y = (
        cy +
        radius_values * np.sin(angle)
    )

    z = rng.uniform(
        cz,
        cz + height,
        number_of_points
    )

    points = np.column_stack(
        (x, y, z)
    )

    labels = np.full(
        number_of_points,
        label,
        dtype=np.int64
    )

    return points, labels


def generate_frame(frame_id=0, seed=None):

    if seed is None:
        seed = CFG.seed

    rng = np.random.default_rng(
        seed + frame_id
    )

    # ==================================
    # GROUND
    # ==================================

    ground_count = int(
        CFG.points_per_frame * 0.72
    )

    x = rng.uniform(
        -CFG.max_range,
        CFG.max_range,
        ground_count
    )

    y = rng.uniform(
        -CFG.max_range,
        CFG.max_range,
        ground_count
    )

    z = (
        0.03 * np.sin(x * 0.08)
        +
        0.04 * np.cos(y * 0.07)
        +
        rng.normal(
            0,
            0.015,
            ground_count
        )
    )

    ground = np.column_stack(
        (x, y, z)
    )

    ground_labels = np.zeros(
        ground_count,
        dtype=np.int64
    )

    parts = [
        (ground, ground_labels)
    ]

    # ==================================
    # WALL
    # ==================================

    wall, wall_labels = create_box(
        rng,
        center=(28, 8, 1.5),
        size=(1, 16, 3),
        number_of_points=2500,
        label=1
    )

    parts.append(
        (wall, wall_labels)
    )

    # ==================================
    # POLES
    # ==================================

    pole_locations = [
        (-12, 18, 0),
        (5, 30, 0),
        (40, -18, 0)
    ]

    for location in pole_locations:

        pole, pole_labels = create_cylinder(
            rng,
            location,
            radius=0.18,
            height=4,
            number_of_points=800,
            label=2
        )

        parts.append(
            (pole, pole_labels)
        )

    # ==================================
    # MOVING VEHICLE
    # ==================================

    vehicle_x = 18 - 0.8 * frame_id

    vehicle, vehicle_labels = create_box(
        rng,
        center=(vehicle_x, -4, 0.8),
        size=(4, 2, 1.6),
        number_of_points=1800,
        label=3
    )

    parts.append(
        (vehicle, vehicle_labels)
    )

    # ==================================
    # MOVING PEDESTRIAN
    # ==================================

    pedestrian_y = -12 + 0.6 * frame_id

    pedestrian, pedestrian_labels = create_box(
        rng,
        center=(12, pedestrian_y, 0.9),
        size=(0.7, 0.7, 1.8),
        number_of_points=1000,
        label=4
    )

    parts.append(
        (pedestrian, pedestrian_labels)
    )

    # ==================================
    # COMBINE
    # ==================================

    points = np.vstack(
        [p for p, _ in parts]
    )

    labels = np.concatenate(
        [l for _, l in parts]
    )

    # ==================================
    # RANGE FILTER
    # ==================================

    distance = np.sqrt(
        points[:, 0] ** 2
        +
        points[:, 1] ** 2
    )

    mask = (
        (distance >= CFG.min_range)
        &
        (distance <= CFG.max_range)
        &
        (points[:, 2] >= -1)
    )

    return (
        points[mask],
        labels[mask]
    )