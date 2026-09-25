import torch
import torch.nn as nn
import torch.nn.functional as F


class PROSERLoss(nn.Module):

    def __init__(
        self,
        num_known_classes=10,
        num_dummy_classes=5,
        beta=1.0,
        gamma=0.1
    ):

        super().__init__()

        self.num_known_classes = (
            num_known_classes
        )

        self.num_dummy_classes = (
            num_dummy_classes
        )

        self.beta = beta
        self.gamma = gamma

        self.ce = nn.CrossEntropyLoss()


    def classifier_placeholder_loss(
        self,
        known_logits,
        dummy_logits,
        labels
    ):

        # Strongest dummy response
        strongest_dummy = dummy_logits.max(
            dim=1
        ).values

        # Strongest known response
        known_without_true = known_logits.clone()

        known_without_true[
            torch.arange(
                labels.size(0),
                device=labels.device
            ),
            labels
        ] = -float("inf")

        strongest_other_known = (
            known_without_true.max(dim=1).values
        )

        # Dummy response should compete with
        # incorrect known-class responses.
        placeholder_logits = torch.stack(
            [
                strongest_other_known,
                strongest_dummy
            ],
            dim=1
        )

        target = torch.ones(
            labels.size(0),
            dtype=torch.long,
            device=labels.device
        )

        return self.ce(
            placeholder_logits,
            target
        )


    def data_placeholder_loss(
        self,
        mixed_known_logits,
        mixed_dummy_logits
    ):

        strongest_dummy = (
            mixed_dummy_logits.max(dim=1).values
        )

        strongest_known = (
            mixed_known_logits.max(dim=1).values
        )

        placeholder_logits = torch.stack(
            [
                strongest_known,
                strongest_dummy
            ],
            dim=1
        )

        target = torch.ones(
            mixed_known_logits.size(0),
            dtype=torch.long,
            device=mixed_known_logits.device
        )

        return self.ce(
            placeholder_logits,
            target
        )


    def forward(
        self,
        known_logits,
        dummy_logits,
        labels,
        mixed_known_logits=None,
        mixed_dummy_logits=None
    ):

        classification_loss = self.ce(
            known_logits,
            labels
        )

        classifier_placeholder = (
            self.classifier_placeholder_loss(
                known_logits,
                dummy_logits,
                labels
            )
        )

        total_loss = (
            classification_loss
            + self.beta * classifier_placeholder
        )

        if (
            mixed_known_logits is not None
            and mixed_dummy_logits is not None
        ):

            data_placeholder = (
                self.data_placeholder_loss(
                    mixed_known_logits,
                    mixed_dummy_logits
                )
            )

            total_loss = (
                total_loss
                + self.gamma * data_placeholder
            )

        else:

            data_placeholder = (
                torch.tensor(
                    0.0,
                    device=known_logits.device
                )
            )

        return (
            total_loss,
            classification_loss,
            classifier_placeholder,
            data_placeholder
        )