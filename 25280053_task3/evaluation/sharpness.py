import torch


def collect_fixed_validation_batch(
    source_val_loaders,
    device,
    seed=6304,
    samples_per_domain=32
):
    """
    Select exactly 32 examples from each source validation
    domain using seed 6304.

    The same batch construction is used for every model.
    """

    generator = torch.Generator()

    generator.manual_seed(seed)

    all_images = []
    all_labels = []

    for domain in [
        "photo",
        "art_painting",
        "cartoon"
    ]:

        dataset = (
            source_val_loaders[domain].dataset
        )

        num_available = len(dataset)

        indices = torch.randperm(
            num_available,
            generator=generator
        )[:samples_per_domain]

        domain_images = []
        domain_labels = []

        for index in indices.tolist():

            sample = dataset[index]

            image = sample[0]
            label = sample[1]

            domain_images.append(
                image
            )

            domain_labels.append(
                label
            )

        all_images.append(
            torch.stack(domain_images)
        )

        all_labels.append(
            torch.tensor(
                domain_labels,
                dtype=torch.long
            )
        )

    images = torch.cat(
        all_images,
        dim=0
    ).to(device)

    labels = torch.cat(
        all_labels,
        dim=0
    ).to(device)

    return images, labels


def compute_loss(
    backbone,
    classifier,
    images,
    labels
):

    features = backbone(images)

    logits = classifier(features)

    loss = torch.nn.functional.cross_entropy(
        logits,
        labels
    )

    return loss


def local_sharpness(
    backbone,
    classifier,
    images,
    labels,
    rho=0.05
):
    """
    Compute:

    Delta_sharp =
        L(theta + epsilon) - L(theta)

    where

        epsilon =
        rho * grad / ||grad||_2
    """

    backbone.eval()
    classifier.eval()

    # --------------------------------------------------------
    # Clear gradients
    # --------------------------------------------------------

    backbone.zero_grad()
    classifier.zero_grad()

    # --------------------------------------------------------
    # Original loss
    # --------------------------------------------------------

    loss = compute_loss(
        backbone,
        classifier,
        images,
        labels
    )

    original_loss = loss.item()

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    loss.backward()

    parameters = []

    for parameter in backbone.parameters():

        if parameter.requires_grad:
            parameters.append(parameter)

    for parameter in classifier.parameters():

        if parameter.requires_grad:
            parameters.append(parameter)

    grad_norm = torch.norm(
        torch.stack([
            parameter.grad.norm(p=2)
            for parameter in parameters
            if parameter.grad is not None
        ]),
        p=2
    )

    # --------------------------------------------------------
    # Normalized perturbation
    # --------------------------------------------------------

    scale = rho / (
        grad_norm + 1e-12
    )

    perturbations = []

    with torch.no_grad():

        for parameter in parameters:

            if parameter.grad is None:
                perturbations.append(None)
                continue

            epsilon = (
                scale *
                parameter.grad
            )

            parameter.add_(epsilon)

            perturbations.append(
                epsilon
            )

    # --------------------------------------------------------
    # Loss at perturbed parameters
    # --------------------------------------------------------

    perturbed_loss = compute_loss(
        backbone,
        classifier,
        images,
        labels
    )

    perturbed_loss_value = (
        perturbed_loss.item()
    )

    # --------------------------------------------------------
    # Restore original parameters
    # --------------------------------------------------------

    with torch.no_grad():

        for parameter, epsilon in zip(
            parameters,
            perturbations
        ):

            if epsilon is not None:
                parameter.sub_(
                    epsilon
                )

    backbone.zero_grad()
    classifier.zero_grad()

    delta_sharp = (
        perturbed_loss_value
        - original_loss
    )

    return {
        "original_loss": float(
            original_loss
        ),

        "perturbed_loss": float(
            perturbed_loss_value
        ),

        "delta_sharp": float(
            delta_sharp
        ),

        "gradient_norm": float(
            grad_norm.item()
        )
    }