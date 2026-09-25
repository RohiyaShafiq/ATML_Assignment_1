import torch
import torch.nn as nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.backbone = resnet18(
            weights=None
        )

        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Remove ImageNet max pooling
        self.backbone.maxpool = nn.Identity()

        feature_dim = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            feature_dim,
            num_classes
        )

        self.feature_dim = feature_dim

    def forward(self, x, return_features=False):

        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Residual blocks
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global average pooling
        x = self.backbone.avgpool(x)

        # Penultimate feature
        features = torch.flatten(x, 1)

        # Logits
        logits = self.backbone.fc(features)

        if return_features:
            return features, logits

        return logits