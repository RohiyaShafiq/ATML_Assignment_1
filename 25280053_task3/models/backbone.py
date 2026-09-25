import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights)


class ResNet18Backbone(nn.Module):

    def __init__(self):

        super().__init__()

        model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        self.feature_dim = model.fc.in_features

        model.fc = nn.Identity()

        self.model = model

    def forward(self, x):

        return self.model(x)

    def freeze_batchnorm_stats(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.BatchNorm2d
            ):

                module.eval()

                # Keep gamma and beta trainable
                if module.weight is not None:
                    module.weight.requires_grad = True

                if module.bias is not None:
                    module.bias.requires_grad = True