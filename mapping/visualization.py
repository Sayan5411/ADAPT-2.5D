import matplotlib.pyplot as plt
import numpy as np


def show_2_5d_map(grid):

    plt.figure(figsize=(10, 8))

    plt.imshow(
        grid,
        origin="lower",
        interpolation="nearest"
    )

    plt.xlabel("X position")
    plt.ylabel("Forward distance")

    plt.title("Adaptive 2.5D LiDAR Map")

    plt.colorbar(
        label="Height (Z)"
    )

    plt.show()