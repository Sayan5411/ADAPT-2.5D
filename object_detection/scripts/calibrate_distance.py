import cv2
from pathlib import Path


# ============================================================
# CAMERA CALIBRATION
# ============================================================

print("=" * 70)
print("ADAPTIVE 2.5D - DISTANCE CALIBRATION")
print("=" * 70)

print()
print("This program calculates the camera focal length")
print("for approximate distance estimation.")
print()
print("You need:")
print("1. An object with known real-world width")
print("2. A measured distance from camera to object")
print()
print("Example:")
print("Object width = 0.50 meters")
print("Distance = 2.00 meters")
print()


# ============================================================
# USER INPUT
# ============================================================

try:

    known_width = float(
        input("Enter object width in meters: ")
    )

    known_distance = float(
        input("Enter distance from camera in meters: ")
    )

except ValueError:

    print("Invalid input.")
    exit()


if known_width <= 0 or known_distance <= 0:

    print("Width and distance must be greater than zero.")
    exit()


# ============================================================
# CAMERA
# ============================================================

print("\nOpening camera...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


print("\nCamera started.")
print("Place the calibration object in front of camera.")
print("Press SPACE when ready.")
print("Press Q to quit.")


pixel_width = None


# ============================================================
# CAPTURE
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    cv2.putText(
        frame,
        "Place object and press SPACE",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Distance Calibration",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        cap.release()
        cv2.destroyAllWindows()

        exit()


    if key == 32:

        print("\nEnter the object's pixel width.")
        print("Measure it from the image window.")

        try:

            pixel_width = float(
                input("Pixel width: ")
            )

        except ValueError:

            print("Invalid pixel width.")
            continue

        break


cap.release()

cv2.destroyAllWindows()


# ============================================================
# CALCULATE FOCAL LENGTH
# ============================================================

focal_length = (
    pixel_width
    * known_distance
    / known_width
)


print("\n" + "=" * 70)

print("CALIBRATION COMPLETE")

print("=" * 70)

print(
    f"\nKnown width: "
    f"{known_width:.3f} m"
)

print(
    f"Known distance: "
    f"{known_distance:.3f} m"
)

print(
    f"Pixel width: "
    f"{pixel_width:.1f} px"
)

print(
    f"\nFocal length: "
    f"{focal_length:.2f} px"
)


# ============================================================
# SAVE
# ============================================================

root = Path(
    r"D:\Lidar_Adaptive_2_5D"
)

output_file = (
    root
    / "object_detection"
    / "focal_length.txt"
)


output_file.write_text(
    str(focal_length)
)


print(
    f"\nSaved focal length to:"
)

print(output_file)

print("\nCalibration finished.")