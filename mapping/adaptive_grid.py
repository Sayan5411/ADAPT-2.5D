import numpy as np

from config import CFG


def resolution_for_distance(distance):

    if distance <= CFG.r0:

        return CFG.res_near

    elif distance <= CFG.r1:

        return CFG.res_10_20

    elif distance <= CFG.r2:

        return CFG.res_20_40

    elif distance <= CFG.r3:

        return CFG.res_40_60

    else:

        return CFG.res_far


def cell_key(x, y, resolution):

    cell_x = int(
        np.floor(x / resolution)
    )

    cell_y = int(
        np.floor(y / resolution)
    )

    return cell_x, cell_y


def build_adaptive_grid(
    points,
    labels=None
):

    points = np.asarray(points)

    if labels is None:

        labels = np.zeros(
            len(points),
            dtype=np.int64
        )

    cells = {}

    for point, label in zip(
        points,
        labels
    ):

        x, y, z = map(
            float,
            point
        )

        distance = np.hypot(
            x,
            y
        )

        resolution = resolution_for_distance(
            distance
        )

        cell_x, cell_y = cell_key(
            x,
            y,
            resolution
        )

        key = (
            resolution,
            cell_x,
            cell_y
        )

        if key not in cells:

            cells[key] = {

                "resolution": resolution,

                "ix": cell_x,

                "iy": cell_y,

                "count": 0,

                "min_z": z,

                "max_z": z,

                "sum_z": 0.0,

                "labels": {}

            }

        cell = cells[key]

        cell["count"] += 1

        cell["min_z"] = min(
            cell["min_z"],
            z
        )

        cell["max_z"] = max(
            cell["max_z"],
            z
        )

        cell["sum_z"] += z

        label = int(label)

        cell["labels"][label] = (
            cell["labels"].get(label, 0)
            + 1
        )

    # ==================================
    # Calculate cell information
    # ==================================

    for cell in cells.values():

        resolution = cell[
            "resolution"
        ]

        cell["mean_z"] = (
            cell["sum_z"]
            /
            max(cell["count"], 1)
        )

        cell["dominant_label"] = max(
            cell["labels"],
            key=cell["labels"].get
        )

        cell["roughness"] = (
            cell["max_z"]
            -
            cell["min_z"]
        )

        cell["center_x"] = (
            cell["ix"] * resolution
            +
            resolution / 2
        )

        cell["center_y"] = (
            cell["iy"] * resolution
            +
            resolution / 2
        )

    return cells