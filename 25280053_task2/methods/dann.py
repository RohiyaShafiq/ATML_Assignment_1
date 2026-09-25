import math

import torch
import torch.nn as nn

from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import (
    DomainDiscriminator,
    gradient_reverse
)


class DANNMethod(nn.Module):

    def __init__(
        self,
        num_classes=7,
        lambda_domain=1.0
    ):
        super().__init__()

        self.backbone = ResNet18Backbone()

        self.classifier = ClassifierHead(
            self.backbone.feature_dim,
            num_classes
        )

        self.domain_discriminator = DomainDiscriminator(
            self.backbone.feature_dim
        )

        self.lambda_domain = lambda_domain

        self.classification_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()

    @staticmethod
    def get_alpha(progress):

        return (
            2.0 /
            (1.0 + math.exp(-10.0 * progress))
            - 1.0
        )

    def forward_source(self, x):

        features = self.backbone(x)

        logits = self.classifier(features)

        return logits, features

    def forward_target(self, x):

        features = self.backbone(x)

        return features

    def compute_loss(
        self,
        source_images,
        source_labels,
        target_images,
        progress
    ):

        # ------------------------------------------------
        # Source
        # ------------------------------------------------

        source_features = self.backbone(
            source_images
        )

        source_logits = self.classifier(
            source_features
        )

        # Only source contributes to classification loss
        cls_loss = self.classification_loss(
            source_logits,
            source_labels
        )

        # ------------------------------------------------
        # Target
        # ------------------------------------------------

        target_features = self.backbone(
            target_images
        )

        # ------------------------------------------------
        # Domain classification
        # ------------------------------------------------

        features = torch.cat(
            [
                source_features,
                target_features
            ],
            dim=0
        )

        # 0 = source
        # 1 = target
        domain_labels = torch.cat(
            [
                torch.zeros(
                    source_features.size(0),
                    dtype=torch.long,
                    device=features.device
                ),
                torch.ones(
                    target_features.size(0),
                    dtype=torch.long,
                    device=features.device
                )
            ],
            dim=0
        )

        # Standard DANN schedule
        alpha = self.get_alpha(progress)

        reversed_features = gradient_reverse(
            features,
            alpha
        )

        domain_logits = self.domain_discriminator(
            reversed_features
        )

        domain_loss = self.domain_loss(
            domain_logits,
            domain_labels
        )

        # Unit weight for domain loss
        total_loss = (
            cls_loss +
            self.lambda_domain * domain_loss
        )

        return (
            total_loss,
            cls_loss,
            domain_loss,
            alpha
        )