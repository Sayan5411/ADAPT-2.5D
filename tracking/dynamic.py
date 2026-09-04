from collections import defaultdict

import numpy as np


class CentroidTracker:

    def __init__(
        self,
        max_match_distance=5.0
    ):

        self.previous = {}

        self.max_match_distance = (
            max_match_distance
        )

    def update(self, cells):

        current = defaultdict(list)

        # Vehicle = 3
        # Pedestrian = 4

        for cell in cells.values():

            label = cell[
                "dominant_label"
            ]

            if label in (3, 4):

                position = np.array([
                    cell["center_x"],
                    cell["center_y"]
                ])

                current[label].append(
                    position
                )

        centroids = {}

        for label, points in current.items():

            centroids[label] = np.mean(
                points,
                axis=0
            )

        movement = {}

        for label, current_position in centroids.items():

            if label in self.previous:

                delta = (
                    current_position
                    -
                    self.previous[label]
                )

                movement[label] = {

                    "dx": float(delta[0]),

                    "dy": float(delta[1]),

                    "speed_per_frame":
                        float(
                            np.linalg.norm(
                                delta
                            )
                        ),

                    "dynamic":
                        bool(
                            np.linalg.norm(
                                delta
                            ) > 0.05
                        )
                }

            else:

                movement[label] = {

                    "dx": 0.0,

                    "dy": 0.0,

                    "speed_per_frame": 0.0,

                    "dynamic": False
                }

        self.previous = centroids

        return movement