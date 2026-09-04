import numpy as np

from config import CFG


def compute_cell_features(cells):

    for cell in cells.values():

        resolution = cell[
            "resolution"
        ]

        roughness = cell[
            "roughness"
        ]

        label = cell[
            "dominant_label"
        ]

        # Estimate slope
        slope = np.degrees(
            np.arctan2(
                roughness,
                max(resolution, 1e-6)
            )
        )

        # Objects are obstacles
        obstacle = label != 0

        if obstacle:

            score = 0.0

        else:

            slope_penalty = min(
                100,
                (
                    slope
                    /
                    CFG.max_safe_slope_deg
                    *
                    70
                )
            )

            roughness_penalty = min(
                30,
                (
                    roughness
                    /
                    CFG.max_roughness
                    *
                    30
                )
            )

            score = max(
                0,
                100
                -
                slope_penalty
                -
                roughness_penalty
            )

        cell["slope_deg"] = float(
            slope
        )

        cell["traversability"] = float(
            score
        )

        cell["obstacle"] = bool(
            obstacle
        )

    return cells