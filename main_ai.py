import time

from data.synthetic_lidar import (
    generate_frame
)

from preprocessing.pointcloud import (
    filter_point_cloud
)

from mapping.semantic_map import (
    build_semantic_2_5d_map
)

from visualization.dashboard import (
    Dashboard
)

from ai.inference import (
    PointNetPredictor
)

from config import CFG


def main():

    # ============================
    # Load AI
    # ============================

    predictor = PointNetPredictor()

    dashboard = Dashboard()

    for frame_id in range(200):

        start_time = (
            time.perf_counter()
        )

        # =========================
        # LiDAR frame
        # =========================

        points, _ = generate_frame(
            frame_id
        )

        # =========================
        # Preprocessing
        # =========================

        points = filter_point_cloud(
            points,
            CFG.min_range,
            CFG.max_range
        )

        # =========================
        # AI prediction
        # =========================

        labels = predictor.predict(
            points
        )

        # =========================
        # Adaptive map
        # =========================

        cells = (
            build_semantic_2_5d_map(
                points,
                labels
            )
        )

        # =========================
        # Performance
        # =========================

        latency = (
            time.perf_counter()
            -
            start_time
        ) * 1000

        fps = (
            1000 /
            max(latency, 1e-6)
        )

        # =========================
        # Visualization
        # =========================

        dashboard.update(
            cells,
            frame_id,
            fps,
            latency
        )

        if frame_id % 20 == 0:

            print(
                f"\nFrame: {frame_id}"
            )

            print(
                f"Points: {len(points)}"
            )

            print(
                f"Cells: {len(cells)}"
            )

            print(
                f"Latency: {latency:.2f} ms"
            )

            print(
                f"FPS: {fps:.2f}"
            )


if __name__ == "__main__":

    main()