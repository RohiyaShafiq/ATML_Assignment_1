import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)

import open_clip

# ResNet50 model
class ResNet50Backbone(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2

        model = resnet50(weights=weights)

        # Remove classification layer.
        model.fc = nn.Identity()

        self.model = model

        self.feature_dim = 2048

        self.freeze()

    def freeze(self):

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(self, x):

        features = self.model(x)

        return features

# ViT model 
class ViTB16Backbone(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1

        model = vit_b_16(weights=weights)

        # Replace classification head.
        model.heads = nn.Identity()

        self.model = model

        self.feature_dim = 768

        self.freeze()

    def freeze(self):

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    @torch.no_grad()
    def forward(self, x):

        features = self.model(x)

        return features

# CLIP
class CLIPViTB32Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai"
        )

        # Required for zero-shot CLIP evaluation
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.feature_dim = self.model.visual.output_dim

        # Freeze CLIP backbone
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):

        image_features = self.model.encode_image(x)

        # Normalize CLIP image embedding
        image_features = F.normalize(
            image_features,
            dim=1
        )

        return image_features

# linear classifier 
class LinearClassifier(nn.Module):

    def __init__(self, input_dim, num_classes=10):

        super().__init__()
        self.fc = nn.Linear(
            input_dim,
            num_classes)

    def forward(self, x):

        return self.fc(x)


# Classifier training
def train_linear_classifier(
    backbone,
    classifier,
    train_loader,
    val_loader,
    device,
    max_epochs=50,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=5,
):

    # Backbone is already frozen and its features
    # have already been extracted.
    backbone.eval()

    for param in backbone.parameters():
        param.requires_grad = False

    classifier = classifier.to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = -float("inf")
    best_state = None

    epochs_without_improvement = 0

    history = []

    for epoch in range(max_epochs):

        # ====================================================
        # Training
        # ====================================================

        classifier.train()

        correct = 0
        total = 0
        train_loss = 0.0

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Features are already extracted from
            # the frozen backbone.
            logits = classifier(features)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item() * features.size(0)
            )

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss /= total
        train_accuracy = correct / total

        # ====================================================
        # Validation
        # ====================================================

        classifier.eval()

        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():

            for features, labels in val_loader:

                features = features.to(device)
                labels = labels.to(device)

                logits = classifier(features)

                loss = criterion(
                    logits,
                    labels
                )

                val_loss += (
                    loss.item() * features.size(0)
                )

                predictions = logits.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch + 1:02d}/{max_epochs} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # ====================================================
        # Early stopping
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:

            print(
                f"Early stopping after "
                f"{epoch + 1} epochs."
            )

            break

    # ========================================================
    # Restore best classifier
    # ========================================================

    if best_state is not None:
        classifier.load_state_dict(best_state)

    classifier = classifier.to(device)

    return classifier, history