from dataclasses import dataclass


@dataclass
class Config:

    # LiDAR range
    max_range: float = 100.0
    min_range: float = 0.5

    # Distance zones
    r0: float = 10.0
    r1: float = 20.0
    r2: float = 40.0
    r3: float = 60.0

    # Grid resolutions
    res_near: float = 0.05
    res_10_20: float = 0.10
    res_20_40: float = 0.20
    res_40_60: float = 0.25
    res_far: float = 0.50

    # Traversability
    max_safe_slope_deg: float = 18.0
    max_roughness: float = 0.20

    # Synthetic LiDAR
    points_per_frame: int = 25000
    seed: int = 7


CFG = Config()


CLASS_NAMES = {
    0: "ground",
    1: "wall",
    2: "pole",
    3: "vehicle",
    4: "pedestrian",
}