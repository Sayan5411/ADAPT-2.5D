import os

import torch

from torch.utils.data import DataLoader

from ai.pointnet import (
    PointNetSegmentation
)

from ai.synthetic_dataset import (
    SyntheticSegmentationDataset
)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Training device:",
        device
    )

    # ============================
    # Dataset
    # ============================

    train_dataset = (
        SyntheticSegmentationDataset(
            samples=500,
            points_per_sample=1024,
            seed=10
        )
    )

    validation_dataset = (
        SyntheticSegmentationDataset(
            samples=100,
            points_per_sample=1024,
            seed=99
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=8
    )

    # ============================
    # Model
    # ============================

    model = PointNetSegmentation(
        num_classes=5
    ).to(device)

    # ============================
    # Optimizer
    # ============================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # ============================
    # Loss
    # ============================

    criterion = (
        torch.nn.CrossEntropyLoss()
    )

    epochs = 15

    # ============================
    # Training
    # ============================

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for points, labels in train_loader:

            points = points.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            predictions = model(
                points
            )

            loss = criterion(
                predictions.reshape(
                    -1,
                    5
                ),
                labels.reshape(
                    -1
                )
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        # ========================
        # Validation
        # ========================

        model.eval()

        correct = 0

        total = 0

        with torch.no_grad():

            for points, labels in validation_loader:

                points = points.to(device)

                labels = labels.to(device)

                predictions = model(
                    points
                )

                predicted_labels = (
                    predictions.argmax(
                        dim=-1
                    )
                )

                correct += (
                    predicted_labels
                    ==
                    labels
                ).sum().item()

                total += labels.numel()

        accuracy = (
            correct / total
        )

        average_loss = (
            total_loss
            /
            len(train_loader)
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{epochs} "
            f"Loss={average_loss:.4f} "
            f"Accuracy={accuracy:.4f}"
        )

    # ============================
    # Save
    # ============================

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = (
        "models/"
        "pointnet_segmentation.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        "\nModel saved:",
        model_path
    )


if __name__ == "__main__":

    main()