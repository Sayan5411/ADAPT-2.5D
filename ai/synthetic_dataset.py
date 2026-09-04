import numpy as np

import torch

from torch.utils.data import Dataset


class SyntheticSegmentationDataset(
    Dataset
):

    def __init__(
        self,
        samples=300,
        points_per_sample=1024,
        seed=42
    ):

        self.samples = samples

        self.number_of_points = (
            points_per_sample
        )

        self.rng = np.random.default_rng(
            seed
        )

    def __len__(self):

        return self.samples

    def __getitem__(self, index):

        n = self.number_of_points

        ground_count = int(
            n * 0.70
        )

        wall_count = int(
            n * 0.10
        )

        pole_count = int(
            n * 0.08
        )

        vehicle_count = int(
            n * 0.07
        )

        pedestrian_count = (
            n
            -
            ground_count
            -
            wall_count
            -
            pole_count
            -
            vehicle_count
        )

        points = []

        labels = []

        # =========================
        # Ground
        # =========================

        x = self.rng.uniform(
            -20,
            20,
            ground_count
        )

        y = self.rng.uniform(
            -20,
            20,
            ground_count
        )

        z = self.rng.normal(
            0,
            0.02,
            ground_count
        )

        points.append(
            np.column_stack(
                (x, y, z)
            )
        )

        labels.append(
            np.zeros(
                ground_count,
                dtype=np.int64
            )
        )

        # =========================
        # Wall
        # =========================

        x = self.rng.normal(
            8,
            0.25,
            wall_count
        )

        y = self.rng.uniform(
            -5,
            5,
            wall_count
        )

        z = self.rng.uniform(
            0,
            3,
            wall_count
        )

        points.append(
            np.column_stack(
                (x, y, z)
            )
        )

        labels.append(
            np.ones(
                wall_count,
                dtype=np.int64
            )
        )

        # =========================
        # Pole
        # =========================

        angle = self.rng.uniform(
            0,
            2 * np.pi,
            pole_count
        )

        radius = self.rng.uniform(
            0,
            0.15,
            pole_count
        )

        x = (
            -7
            +
            radius * np.cos(angle)
        )

        y = (
            7
            +
            radius * np.sin(angle)
        )

        z = self.rng.uniform(
            0,
            4,
            pole_count
        )

        points.append(
            np.column_stack(
                (x, y, z)
            )
        )

        labels.append(
            np.full(
                pole_count,
                2,
                dtype=np.int64
            )
        )

        # =========================
        # Vehicle
        # =========================

        x = self.rng.uniform(
            3,
            6,
            vehicle_count
        )

        y = self.rng.uniform(
            -7,
            -4,
            vehicle_count
        )

        z = self.rng.uniform(
            0,
            1.5,
            vehicle_count
        )

        points.append(
            np.column_stack(
                (x, y, z)
            )
        )

        labels.append(
            np.full(
                vehicle_count,
                3,
                dtype=np.int64
            )
        )

        # =========================
        # Pedestrian
        # =========================

        x = self.rng.uniform(
            12,
            12.7,
            pedestrian_count
        )

        y = self.rng.uniform(
            4,
            4.7,
            pedestrian_count
        )

        z = self.rng.uniform(
            0,
            1.8,
            pedestrian_count
        )

        points.append(
            np.column_stack(
                (x, y, z)
            )
        )

        labels.append(
            np.full(
                pedestrian_count,
                4,
                dtype=np.int64
            )
        )

        points = np.vstack(points)

        labels = np.concatenate(labels)

        # Shuffle

        order = self.rng.permutation(
            len(points)
        )

        points = points[order]

        labels = labels[order]

        return (
            torch.from_numpy(
                points.astype(np.float32)
            ),

            torch.from_numpy(
                labels
            )
        )