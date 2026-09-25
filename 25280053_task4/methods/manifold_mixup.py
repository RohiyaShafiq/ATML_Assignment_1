import torch


def make_different_class_permutation(labels):
    """
    Construct a one-to-one permutation where each sample
    is paired with a sample from a different class.
    """

    labels = labels.detach()

    batch_size = labels.size(0)

    if batch_size < 2:
        raise RuntimeError(
            "Batch must contain at least two samples."
        )

    # Start with a random permutation
    permutation = torch.randperm(
        batch_size,
        device=labels.device
    )

    # Iteratively repair same-class pairs
    #
    # CIFAR-10 batches contain many samples from different
    # classes, so this normally converges immediately.
    max_attempts = 100

    for _ in range(max_attempts):

        same_class = (
            labels
            ==
            labels[permutation]
        )

        if not torch.any(same_class):
            return permutation

        bad_positions = torch.nonzero(
            same_class,
            as_tuple=False
        ).flatten()

        # Try random permutations to repair conflicts
        candidate = permutation.clone()

        random_order = torch.randperm(
            batch_size,
            device=labels.device
        )

        candidate = candidate[random_order]

        if not torch.any(
            labels
            ==
            labels[candidate]
        ):
            return candidate

        # Swap conflicting entries
        for position in bad_positions:

            valid = torch.nonzero(
                labels[permutation]
                !=
                labels[position],
                as_tuple=False
            ).flatten()

            if valid.numel() == 0:
                continue

            random_index = valid[
                torch.randint(
                    0,
                    valid.numel(),
                    (1,),
                    device=labels.device
                )
            ]

            random_index = random_index.item()

            other_position = random_index

            temp = permutation[position].clone()

            permutation[position] = (
                permutation[other_position]
            )

            permutation[other_position] = temp

        if not torch.any(
            labels
            ==
            labels[permutation]
        ):
            return permutation

    # --------------------------------------------------------
    # Guaranteed fallback:
    #
    # Sort samples by class and cyclically shift the sorted
    # indices. For a normal CIFAR-10 batch this produces
    # different-class pairings.
    # --------------------------------------------------------

    sorted_indices = torch.argsort(
        labels
    )

    sorted_labels = labels[
        sorted_indices
    ]

    for shift in range(
        1,
        batch_size
    ):

        candidate = torch.roll(
            sorted_indices,
            shifts=shift,
            dims=0
        )

        if not torch.any(
            sorted_labels
            ==
            labels[candidate]
        ):
            return candidate

    raise RuntimeError(
        "Could not construct different-class pairing "
        "for manifold mixup."
    )


def manifold_mixup(
    representations,
    labels,
    alpha=2.0
):
    """
    Manifold Mixup.

    representations:
        Hidden representations after layer2.

    labels:
        CIFAR-10 labels.

    alpha:
        Beta distribution parameter.
    """

    batch_size = representations.size(0)

    permutation = make_different_class_permutation(
        labels
    )

    # --------------------------------------------------------
    # lambda ~ Beta(alpha, alpha)
    # --------------------------------------------------------

    beta_distribution = torch.distributions.Beta(
        alpha,
        alpha
    )

    lam = beta_distribution.sample(
        (batch_size,)
    ).to(
        device=representations.device,
        dtype=representations.dtype
    )

    # --------------------------------------------------------
    # Broadcast lambda over representation dimensions
    # --------------------------------------------------------

    view_shape = (
        [batch_size]
        +
        [1] * (
            representations.dim() - 1
        )
    )

    lam_view = lam.view(
        *view_shape
    )

    # --------------------------------------------------------
    # Mix representations
    # --------------------------------------------------------

    mixed_representations = (
        lam_view * representations
        +
        (1.0 - lam_view)
        * representations[permutation]
    )

    labels_a = labels

    labels_b = labels[
        permutation
    ]

    return (
        mixed_representations,
        labels_a,
        labels_b,
        lam
    )