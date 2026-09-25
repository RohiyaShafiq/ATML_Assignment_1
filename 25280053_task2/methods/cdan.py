import math

import torch
import torch.nn as nn

from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead
from task2.models.domain_discriminator import (
    DomainDiscriminator,
    gradient_reverse,
)


class CDANMethod(nn.Module):
    def __init__(self, num_classes=7, lambda_domain=1.0):
        super().__init__()

        self.backbone = ResNet18Backbone()

        self.classifier = ClassifierHead(
            self.backbone.feature_dim,
            num_classes
        )

        # f: 512 dimensions
        # p: 7 dimensions
        # f ⊗ p: 512 * 7 = 3584 dimensions
        self.cdan_feature_dim = (
            self.backbone.feature_dim * num_classes
        )

        self.domain_discriminator = DomainDiscriminator(
            input_dim=self.cdan_feature_dim
        )

        self.lambda_domain = lambda_domain

        self.classification_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()

    @staticmethod
    def get_alpha(progress):
        return (
            2.0 / (1.0 + math.exp(-10.0 * progress))
        ) - 1.0

    def forward_source(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)

        return logits, features

    def forward_target(self, x):
        features = self.backbone(x)

        return features

    def compute_cdan_features(self, features, probabilities):
        """
        Compute g(x) = vec(f ⊗ p)

        features:
            [batch, 512]

        probabilities:
            [batch, 7]

        output:
            [batch, 512 * 7] = [batch, 3584]
        """

        outer_product = torch.bmm(
            probabilities.unsqueeze(2),
            features.unsqueeze(1)
        )

        return outer_product.view(
            features.size(0),
            -1
        )

    def compute_loss(
        self,
        source_images,
        source_labels,
        target_images,
        progress
    ):
        # --------------------------------------------------
        # Source
        # --------------------------------------------------
        source_features = self.backbone(source_images)

        source_logits = self.classifier(
            source_features
        )

        # Only source examples contribute to class loss
        cls_loss = self.classification_loss(
            source_logits,
            source_labels
        )

        # --------------------------------------------------
        # Target
        # --------------------------------------------------
        target_features = self.backbone(target_images)

        target_logits = self.classifier(
            target_features
        )

        # --------------------------------------------------
        # Probability vectors
        # --------------------------------------------------
        source_probabilities = torch.softmax(
            source_logits,
            dim=1
        )

        target_probabilities = torch.softmax(
            target_logits,
            dim=1
        )

        # --------------------------------------------------
        # CDAN representation
        # g(x) = vec(f ⊗ p)
        # --------------------------------------------------
        source_cdan = self.compute_cdan_features(
            source_features,
            source_probabilities
        )

        target_cdan = self.compute_cdan_features(
            target_features,
            target_probabilities
        )

        # Pool source and target examples
        cdan_features = torch.cat(
            [source_cdan, target_cdan],
            dim=0
        )

        # Source = 0
        # Target = 1
        domain_labels = torch.cat(
            [
                torch.zeros(
                    source_cdan.size(0),
                    dtype=torch.long,
                    device=cdan_features.device
                ),
                torch.ones(
                    target_cdan.size(0),
                    dtype=torch.long,
                    device=cdan_features.device
                )
            ],
            dim=0
        )

        # --------------------------------------------------
        # Gradient reversal
        # --------------------------------------------------
        alpha = self.get_alpha(progress)

        reversed_features = gradient_reverse(
            cdan_features,
            alpha
        )

        # --------------------------------------------------
        # Domain classification
        # --------------------------------------------------
        domain_logits = self.domain_discriminator(
            reversed_features
        )

        domain_loss = self.domain_loss(
            domain_logits,
            domain_labels
        )

        # Unit domain-loss weight
        total_loss = (
            cls_loss
            + self.lambda_domain * domain_loss
        )

        return (
            total_loss,
            cls_loss,
            domain_loss,
            alpha
        )