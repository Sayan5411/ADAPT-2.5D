import torch

import torch.nn as nn

import torch.nn.functional as F


class PointNetSegmentation(
    nn.Module
):

    def __init__(
        self,
        num_classes=5
    ):

        super().__init__()

        self.layer1 = nn.Conv1d(
            3,
            64,
            1
        )

        self.layer2 = nn.Conv1d(
            64,
            128,
            1
        )

        self.layer3 = nn.Conv1d(
            128,
            256,
            1
        )

        self.head1 = nn.Conv1d(
            512,
            128,
            1
        )

        self.head2 = nn.Conv1d(
            128,
            64,
            1
        )

        self.output = nn.Conv1d(
            64,
            num_classes,
            1
        )

        self.bn1 = nn.BatchNorm1d(64)

        self.bn2 = nn.BatchNorm1d(128)

        self.bn3 = nn.BatchNorm1d(256)

        self.bn4 = nn.BatchNorm1d(128)

        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, x):

        # Input:

        # [Batch, Points, XYZ]

        x = x.transpose(
            1,
            2
        )

        # [Batch, XYZ, Points]

        local1 = F.relu(
            self.bn1(
                self.layer1(x)
            )
        )

        local2 = F.relu(
            self.bn2(
                self.layer2(local1)
            )
        )

        local3 = F.relu(
            self.bn3(
                self.layer3(local2)
            )
        )

        # Global feature

        global_feature = torch.max(
            local3,
            dim=2,
            keepdim=True
        )[0]

        global_feature = global_feature.repeat(
            1,
            1,
            local3.shape[2]
        )

        # Combine local + global

        features = torch.cat(
            [
                local2,
                local3,
                global_feature
            ],
            dim=1
        )

        features = F.relu(
            self.bn4(
                self.head1(features)
            )
        )

        features = F.relu(
            self.bn5(
                self.head2(features)
            )
        )

        output = self.output(
            features
        )

        return output.transpose(
            1,
            2
        )