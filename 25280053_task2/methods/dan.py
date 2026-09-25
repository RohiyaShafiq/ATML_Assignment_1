import torch
import torch.nn as nn

from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead


class DANMethod(nn.Module):

    def __init__(self, num_classes=7, lambda_mmd=1.0):
        super().__init__()

        self.backbone = ResNet18Backbone()
        self.classifier = ClassifierHead(
            self.backbone.feature_dim,
            num_classes)

        self.lambda_mmd = lambda_mmd
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)

        return logits, features

    def compute_mmd(self, source_features, target_features):

        # Combine source and target features
        features = torch.cat(
            [source_features, target_features],
            dim=0)

        # Pairwise squared Euclidean distances
        dist_sq = torch.cdist(
            features,
            features,
            p=2).pow(2)

        # Median pairwise squared distance
        n = dist_sq.size(0)

        mask = ~torch.eye(
            n,
            dtype=torch.bool,
            device=dist_sq.device)

        median_dist = dist_sq[mask].median()

        # Three bandwidths:
        # 0.5, 1, and 2 times the median distance
        bandwidths = [
            0.5 * median_dist,
            1.0 * median_dist,
            2.0 * median_dist]

        # Sum of three RBF kernels
        kernel = 0.0

        for bandwidth in bandwidths:
            kernel = kernel + torch.exp(
                -dist_sq / (2.0 * bandwidth + 1e-8))

        ns = source_features.size(0)
        nt = target_features.size(0)

        # Source-source kernel
        K_ss = kernel[:ns, :ns]

        # Target-target kernel
        K_tt = kernel[ns:, ns:]

        # Source-target kernel
        K_st = kernel[:ns, ns:]

        # Remove diagonal terms from source-source
        # and target-target matrices
        source_mask = ~torch.eye(
            ns,
            dtype=torch.bool,
            device=kernel.device)

        target_mask = ~torch.eye(
            nt,
            dtype=torch.bool,
            device=kernel.device)

        # Unbiased MMD^2 estimator
        mmd_ss = K_ss[source_mask].mean()
        mmd_tt = K_tt[target_mask].mean()
        mmd_st = K_st.mean()

        mmd = mmd_ss + mmd_tt - 2.0 * mmd_st

        return mmd

    def loss(
        self,
        source_images,
        source_labels,
        target_images):

        # Source forward pass
        source_logits, source_features = self.forward(
            source_images)

        # Target forward pass
        _, target_features = self.forward(
            target_images)

        # Classification loss
        cls_loss = self.ce_loss(
            source_logits,
            source_labels)

        # DAN MMD loss
        mmd_loss = self.compute_mmd(
            source_features,
            target_features)

        # Total DAN loss
        total_loss = (cls_loss + self.lambda_mmd * mmd_loss)

        return total_loss, cls_loss, mmd_loss