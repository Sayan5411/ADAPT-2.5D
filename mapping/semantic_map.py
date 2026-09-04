from mapping.adaptive_grid import (
    build_adaptive_grid
)

from mapping.traversability import (
    compute_cell_features
)


def build_semantic_2_5d_map(
    points,
    labels
):

    cells = build_adaptive_grid(
        points,
        labels
    )

    cells = compute_cell_features(
        cells
    )

    return cells