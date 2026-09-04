from .config import CLASS_WIDTHS, FOCAL_LENGTH


OBJECT_WIDTHS = {
    "Car": 1.8,
    "Van": 2.0,
    "Truck": 2.5,
    "Pedestrian": 0.6,
    "Person_sitting": 0.6,
    "Cyclist": 0.7,
    "Tram": 2.5,
    "Misc": 1.0,
}

def estimate_distance(class_name, pixel_width):

    if pixel_width <= 0:
        return 999.0

    real_width = OBJECT_WIDTHS.get(
        class_name,
        1.0
    )

    distance = (
        real_width * FOCAL_LENGTH
    ) / pixel_width

    return distance