from ultralytics import YOLO


def main():

    print("=" * 60)
    print("LIDAR OBJECT DETECTION - MODEL TRAINING")
    print("=" * 60)

    # Load pretrained YOLO11 Nano
    model = YOLO("yolo11n.pt")

    # Train
    model.train(
        data=r"D:\Lidar_Adaptive_2_5D\object_detection\data.yaml",

        # Training parameters
        epochs=50,
        imgsz=640,
        batch=4,

        # NVIDIA GPU
        device=0,

        # Windows: avoid multiprocessing error
        workers=0,

        # Output
        project=r"D:\Lidar_Adaptive_2_5D\object_detection\runs",
        name="lidar_object_detector",

        # Validation
        val=True,

        # Save model
        save=True,

        # Stop if validation stops improving
        patience=10,

        # GPU memory optimization
        amp=True,

        # Do not cache entire dataset in RAM
        cache=False,

        verbose=True
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        r"Best model:"
        r" D:\Lidar_Adaptive_2_5D\object_detection\runs"
        r"\lidar_object_detector\weights\best.pt"
    )


if __name__ == "__main__":
    main()