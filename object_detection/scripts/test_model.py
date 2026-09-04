from ultralytics import YOLO
import os
import sys


# ============================================================
# CONFIGURATION
# ============================================================

# Your ACTUAL trained model
MODEL_PATH = r"D:\Lidar_Adaptive_2_5D\object_detection\runs\lidar_object_detector-4\weights\best.pt"

# Dataset configuration
DATA_YAML = r"D:\Lidar_Adaptive_2_5D\object_detection\data.yaml"

# Project directory
PROJECT_DIR = r"D:\Lidar_Adaptive_2_5D\object_detection"

# Results directory
RESULTS_DIR = os.path.join(PROJECT_DIR, "test_results")


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Car",
    1: "Van",
    2: "Truck",
    3: "Pedestrian",
    4: "Person_sitting",
    5: "Cyclist",
    6: "Tram",
    7: "Misc"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("       TESTING TRAINED LIDAR OBJECT DETECTION MODEL")
print("=" * 70)


# ============================================================
# CHECK MODEL
# ============================================================

print("\nChecking trained model...")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: best.pt not found!")
    print("\nExpected model:")
    print(MODEL_PATH)
    print("\nPlease check the model path.")
    sys.exit(1)

print("Model found!")
print(MODEL_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO model...")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("\nERROR while loading model:")
    print(e)
    sys.exit(1)

print("Model loaded successfully!")


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print("\nModel path:")
print(MODEL_PATH)

print("\nNumber of classes:")
print(len(CLASS_NAMES))

print("\nClasses:")

for class_id, class_name in CLASS_NAMES.items():
    print(f"  {class_id}: {class_name}")


# ============================================================
# VALIDATE MODEL
# ============================================================

print("\n" + "=" * 70)
print("RUNNING VALIDATION")
print("=" * 70)

print("\nValidation dataset:")
print(DATA_YAML)

try:

    metrics = model.val(
        data=DATA_YAML,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        project=RESULTS_DIR,
        name="validation"
    )

except Exception as e:

    print("\nERROR during validation:")
    print(e)

    sys.exit(1)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)


try:

    print("\nOverall Performance:")

    print(
        f"Precision     : {metrics.box.mp:.4f}"
    )

    print(
        f"Recall        : {metrics.box.mr:.4f}"
    )

    print(
        f"mAP50         : {metrics.box.map50:.4f}"
    )

    print(
        f"mAP50-95      : {metrics.box.map:.4f}"
    )

except Exception:

    print("\nCould not read all metric values.")


# ============================================================
# CLASS-WISE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE PERFORMANCE")
print("=" * 70)


try:

    for class_id, class_name in CLASS_NAMES.items():

        if class_id >= len(metrics.box.p):

            continue

        precision = metrics.box.p[class_id]
        recall = metrics.box.r[class_id]
        map50 = metrics.box.ap50[class_id]
        map5095 = metrics.box.ap[class_id]

        print(
            f"\n{class_id}: {class_name}"
        )

        print(
            f"    Precision : {precision:.4f}"
        )

        print(
            f"    Recall    : {recall:.4f}"
        )

        print(
            f"    mAP50     : {map50:.4f}"
        )

        print(
            f"    mAP50-95  : {map5095:.4f}"
        )

except Exception as e:

    print("\nCould not display class-wise results.")
    print(e)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("TESTING COMPLETED")
print("=" * 70)

print("\nYour trained model is working.")

print("\nModel used:")
print(MODEL_PATH)

print("\nThe model detects:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"  {class_id}: {class_name}"
    )

print("\nValidation results saved inside:")

print(RESULTS_DIR)

print("\n" + "=" * 70)