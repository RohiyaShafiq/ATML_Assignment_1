import torch
import torch.nn as nn

from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead


class SAMMethod(nn.Module):

    def __init__(
        self,
        num_classes=7,
        rho=0.05
    ):
        super().__init__()

        self.backbone = ResNet18Backbone()

        self.classifier = ClassifierHead(
            self.backbone.feature_dim,
            num_classes
        )

        self.rho = rho
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):

        features = self.backbone(x)
        logits = self.classifier(features)

        return logits, features

    def loss(
        self,
        images,
        labels
    ):

        logits, _ = self.forward(images)

        return self.loss_fn(
            logits,
            labels
        )

    def freeze_bn_stats(self):

        self.backbone.freeze_batchnorm_stats()