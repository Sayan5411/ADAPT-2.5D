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

from tracking.dynamic import (
    CentroidTracker
)

from visualization.dashboard import (
    Dashboard
)

from config import CFG


def main():

    tracker = CentroidTracker()

    dashboard = Dashboard()

    previous_time = time.perf_counter()

    for frame_id in range(200):

        start_time = (
            time.perf_counter()
        )

        # ==========================
        # 1. Get LiDAR
        # ==========================

        points, labels = (
            generate_frame(
                frame_id
            )
        )

        # ==========================
        # 2. Preprocess
        # ==========================

        points = filter_point_cloud(
            points,
            CFG.min_range,
            CFG.max_range
        )

        # ==========================
        # 3. Adaptive 2.5D Map
        # ==========================

        cells = (
            build_semantic_2_5d_map(
                points,
                labels
            )
        )

        # ==========================
        # 4. Dynamic tracking
        # ==========================

        movement = tracker.update(
            cells
        )

        # ==========================
        # 5. Performance
        # ==========================

        latency = (
            time.perf_counter()
            -
            start_time
        ) * 1000

        current_time = (
            time.perf_counter()
        )

        fps = 1 / max(
            current_time
            -
            previous_time,
            1e-6
        )

        previous_time = current_time

        # ==========================
        # 6. Visualization
        # ==========================

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
                f"Grid cells: {len(cells)}"
            )

            print(
                f"Latency: {latency:.2f} ms"
            )

            print(
                f"FPS: {fps:.2f}"
            )

            print(
                "Movement:",
                movement
            )

        time.sleep(0.01)


if __name__ == "__main__":

    main()