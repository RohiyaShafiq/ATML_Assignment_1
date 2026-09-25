import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.network(x)


class GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reverse(x, alpha):
    return GradientReversalFunction.apply(x, alpha)