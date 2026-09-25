import torch
import torch.nn as nn

from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead


class DANMethod(nn.Module):

    def __init__(self, num_classes=7, lambda_dg=1.0):
        super().__init__()

        self.backbone = ResNet18Backbone()

        self.classifier = ClassifierHead(
            self.backbone.feature_dim,
            num_classes
        )

        self.lambda_dg = lambda_dg
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):

        features = self.backbone(x)
        logits = self.classifier(features)

        return logits, features

    def compute_mmd(
        self,
        features_a,
        features_b
    ):
        """
        Compute MMD^2 between two source-domain
        feature distributions.

        Three RBF kernels are used with bandwidths:
            0.5 * median
            1.0 * median
            2.0 * median

        The median is computed separately for
        this domain pair and current batch.
        """

        # ----------------------------------------------------
        # Pairwise squared Euclidean distances
        # ----------------------------------------------------

        features = torch.cat(
            [features_a, features_b],
            dim=0
        )

        dist_sq = torch.cdist(
            features,
            features,
            p=2
        ).pow(2)

        n = dist_sq.size(0)

        # Remove self-distances when computing median
        mask = ~torch.eye(
            n,
            dtype=torch.bool,
            device=dist_sq.device
        )

        median_dist = dist_sq[mask].median()

        # ----------------------------------------------------
        # Three RBF kernels
        # ----------------------------------------------------

        bandwidths = [
            0.5 * median_dist,
            1.0 * median_dist,
            2.0 * median_dist
        ]

        kernel = torch.zeros_like(dist_sq)

        for bandwidth in bandwidths:

            kernel += torch.exp(
                -dist_sq /
                (2.0 * bandwidth + 1e-8)
            )

        # ----------------------------------------------------
        # Split kernel into domain blocks
        # ----------------------------------------------------

        na = features_a.size(0)
        nb = features_b.size(0)

        K_aa = kernel[:na, :na]
        K_bb = kernel[na:, na:]
        K_ab = kernel[:na, na:]

        # Remove diagonal terms
        mask_a = ~torch.eye(
            na,
            dtype=torch.bool,
            device=kernel.device
        )

        mask_b = ~torch.eye(
            nb,
            dtype=torch.bool,
            device=kernel.device
        )

        # ----------------------------------------------------
        # Unbiased MMD^2
        # ----------------------------------------------------

        mmd_aa = K_aa[mask_a].mean()
        mmd_bb = K_bb[mask_b].mean()
        mmd_ab = K_ab.mean()

        mmd = (
            mmd_aa
            + mmd_bb
            - 2.0 * mmd_ab
        )

        return mmd

    def loss(self, source_batches):
        """
        source_batches:

        {
            "photo": (images, labels),
            "art_painting": (images, labels),
            "cartoon": (images, labels)
        }

        No Sketch/target data is used.
        """

        features = {}
        classification_losses = []

        # ----------------------------------------------------
        # Forward pass for all three source domains
        # ----------------------------------------------------

        for domain in [
            "photo",
            "art_painting",
            "cartoon"
        ]:

            images, labels = source_batches[domain]

            logits, domain_features = self.forward(images)

            features[domain] = domain_features

            classification_losses.append(
                self.ce_loss(
                    logits,
                    labels
                )
            )

        # ----------------------------------------------------
        # ERM classification loss
        # ----------------------------------------------------

        L_ERM = torch.stack(
            classification_losses
        ).mean()

        # ----------------------------------------------------
        # Three unordered source-domain pairs
        # ----------------------------------------------------

        mmd_photo_art = self.compute_mmd(
            features["photo"],
            features["art_painting"]
        )

        mmd_photo_cartoon = self.compute_mmd(
            features["photo"],
            features["cartoon"]
        )

        mmd_art_cartoon = self.compute_mmd(
            features["art_painting"],
            features["cartoon"]
        )

        # ----------------------------------------------------
        # Average MMD over the three pairs
        # ----------------------------------------------------

        average_mmd = (
            mmd_photo_art
            + mmd_photo_cartoon
            + mmd_art_cartoon
        ) / 3.0

        # ----------------------------------------------------
        # DAN-DG objective
        # ----------------------------------------------------

        total_loss = (
            L_ERM
            + self.lambda_dg * average_mmd
        )

        return (
            total_loss,
            L_ERM,
            average_mmd
        )