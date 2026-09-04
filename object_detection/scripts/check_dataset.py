from pathlib import Path

# --------------------------------------------------
# PROJECT DATASET LOCATION
# --------------------------------------------------

DATASET = Path("object_detection")


# --------------------------------------------------
# DATASET PATHS
# --------------------------------------------------

image_folder = (
    DATASET
    / "data_object_image_2"
    / "training"
    / "image_2"
)

label_folder = (
    DATASET
    / "data_object_label_2"
)


# --------------------------------------------------
# CHECK FOLDERS
# --------------------------------------------------

print("=" * 60)
print("KITTI DATASET CHECK")
print("=" * 60)

print("\nImage folder:")
print(image_folder.resolve())

print("\nLabel folder:")
print(label_folder.resolve())


if not image_folder.exists():
    print("\n❌ IMAGE FOLDER NOT FOUND!")
    exit()

if not label_folder.exists():
    print("\n❌ LABEL FOLDER NOT FOUND!")
    exit()


print("\n✅ Image folder found")
print("✅ Label folder found")


# --------------------------------------------------
# FIND IMAGES
# --------------------------------------------------

images = list(image_folder.glob("*.png"))

if not images:
    images = list(image_folder.glob("*.jpg"))

if not images:
    images = list(image_folder.glob("*.jpeg"))


# --------------------------------------------------
# FIND LABELS
# --------------------------------------------------

labels = list(label_folder.glob("*.txt"))


print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"Images found : {len(images)}")
print(f"Labels found : {len(labels)}")


# --------------------------------------------------
# SHOW SAMPLE FILES
# --------------------------------------------------

print("\nSample images:")

for image in sorted(images)[:5]:
    print("  ", image.name)


print("\nSample labels:")

for label in sorted(labels)[:5]:
    print("  ", label.name)


# --------------------------------------------------
# CHECK IMAGE-LABEL MATCHING
# --------------------------------------------------

image_names = {
    image.stem
    for image in images
}

label_names = {
    label.stem
    for label in labels
}


matched = image_names.intersection(label_names)

images_without_labels = image_names - label_names
labels_without_images = label_names - image_names


print("\n" + "=" * 60)
print("IMAGE / LABEL MATCHING")
print("=" * 60)

print(f"Matched pairs          : {len(matched)}")
print(f"Images without labels  : {len(images_without_labels)}")
print(f"Labels without images  : {len(labels_without_images)}")


# --------------------------------------------------
# SHOW PROBLEMS
# --------------------------------------------------

if images_without_labels:
    print("\n⚠ Images without labels:")

    for name in sorted(images_without_labels)[:10]:
        print("  ", name)


if labels_without_images:
    print("\n⚠ Labels without images:")

    for name in sorted(labels_without_images)[:10]:
        print("  ", name)


# --------------------------------------------------
# FINISH
# --------------------------------------------------

print("\n" + "=" * 60)

if len(matched) > 0:
    print("✅ DATASET CAN BE READ SUCCESSFULLY")
else:
    print("❌ NO MATCHING IMAGE/LABEL PAIRS FOUND")

print("=" * 60)