import torch.nn as nn

class SourceOnlyMethod:

    def __init__(self, backbone, classifier):

        self.backbone = backbone
        self.classifier = classifier

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, source_images):

        features = self.backbone(source_images)
        logits = self.classifier(features)

        return logits, features

    def loss(self, source_images, source_labels):

        logits, _ = self.forward(source_images)
        return self.loss_fn(logits, source_labels)