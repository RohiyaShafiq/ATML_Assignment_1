import torch.nn as nn


class GCSCLoss(nn.Module):
    """
    GCSC uses the standard cross-entropy classification loss.
    The difference from Vanilla is the training augmentation:
    RandAugment(num_ops=2, magnitude=9).
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.loss(logits, targets)