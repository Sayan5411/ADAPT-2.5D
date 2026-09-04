import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from config import CLASS_NAMES


class Dashboard:

    def __init__(self):

        plt.ion()

        self.figure, self.axis = plt.subplots(
            figsize=(11, 8)
        )

    def update(
        self,
        cells,
        frame_id,
        fps,
        latency_ms
    ):

        self.axis.clear()

        markers = {

            0: "o",

            1: "s",

            2: "^",

            3: "D",

            4: "P"
        }

        for cell in cells.values():

            x = cell["center_x"]

            y = cell["center_y"]

            resolution = cell[
                "resolution"
            ]

            label = cell[
                "dominant_label"
            ]

            if label == 0:

                rectangle = Rectangle(

                    (
                        x - resolution / 2,
                        y - resolution / 2
                    ),

                    resolution,

                    resolution,

                    fill=False,

                    linewidth=0.25,

                    alpha=0.35
                )

                self.axis.add_patch(
                    rectangle
                )

            else:

                self.axis.scatter(

                    [x],

                    [y],

                    s=max(
                        5,
                        min(
                            80,
                            resolution * 120
                        )
                    ),

                    marker=markers.get(
                        label,
                        "x"
                    ),

                    label=CLASS_NAMES.get(
                        label,
                        "unknown"
                    )
                )

        handles, labels = (
            self.axis.get_legend_handles_labels()
        )

        unique = dict(
            zip(labels, handles)
        )

        if unique:

            self.axis.legend(
                unique.values(),
                unique.keys(),
                loc="upper right"
            )

        self.axis.set_aspect(
            "equal",
            adjustable="box"
        )

        self.axis.set_xlabel(
            "X (meters)"
        )

        self.axis.set_ylabel(
            "Y (meters)"
        )

        self.axis.set_title(

            f"Adaptive 2.5D LiDAR Map | "
            f"Frame: {frame_id} | "
            f"FPS: {fps:.1f} | "
            f"Latency: {latency_ms:.1f} ms"
        )

        self.axis.grid(
            alpha=0.15
        )

        self.figure.canvas.draw()

        self.figure.canvas.flush_events()