import torch
import torch.nn as nn


class VanillaLoss:

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, logits, targets):
        return self.criterion(logits, targets)