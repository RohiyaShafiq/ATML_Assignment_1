import sys
from pathlib import Path

sys.path.insert(
    0,
    "/content/drive/MyDrive"
)

# Reuse everything from Part 5
from task2.evaluate_final import (
    load_split_file,
    build_datasets,
    build_loaders,
    load_model,
    evaluate_classifier,
    domain_separability,
    set_seed,
    DEVICE,
    RESULTS_DIR,
    SOURCE_DOMAINS,
)


def main():

    set_seed(6304)

    split_data = load_split_file()

    datasets = build_datasets(
        split_data
    )

    loaders = build_loaders(
        datasets
    )

    source_loaders = {
        domain: loaders[f"{domain}_val"]
        for domain in SOURCE_DOMAINS
    }

    checkpoints = {
        "DAN λ=0.1":
            RESULTS_DIR / "dan_lambda_0.1_best.pth",

        "DAN λ=1":
            RESULTS_DIR / "dan_lambda_1.0_best.pth",

        "DAN λ=10":
            RESULTS_DIR / "dan_lambda_10.0_best.pth",
    }

    results = []

    print("\n" + "=" * 80)
    print("DAN LAMBDA SENSITIVITY STUDY")

    for method, checkpoint in checkpoints.items():

        print(f"\n{method}")

        backbone, classifier = load_model(
            checkpoint
        )

        # ----------------------------------------------------
        # Source validation
        # ----------------------------------------------------

        source_acc = []
        source_f1 = []

        for domain in SOURCE_DOMAINS:

            acc, f1, _, _ = evaluate_classifier(
                backbone,
                classifier,
                loaders[f"{domain}_val"]
            )

            source_acc.append(acc)
            source_f1.append(f1)

        mean_source_acc = sum(
            source_acc
        ) / len(source_acc)

        mean_source_f1 = sum(
            source_f1
        ) / len(source_f1)

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        (
            target_acc,
            target_f1,
            _,
            _
        ) = evaluate_classifier(
            backbone,
            classifier,
            loaders["target"]
        )

        # ----------------------------------------------------
        # Domain separability
        # ----------------------------------------------------

        domain_score = domain_separability(
            backbone,
            source_loaders,
            loaders["target"]
        )

        results.append(
            {
                "method": method,
                "source_accuracy":
                    mean_source_acc,
                "source_macro_f1":
                    mean_source_f1,
                "target_accuracy":
                    target_acc,
                "target_macro_f1":
                    target_f1,
                "domain_separability":
                    domain_score,
            }
        )

        print(
            f"Source Acc: {mean_source_acc:.4f}"
        )

        print(
            f"Source Macro-F1: {mean_source_f1:.4f}"
        )

        print(
            f"Target Acc: {target_acc:.4f}"
        )

        print(
            f"Target Macro-F1: {target_f1:.4f}"
        )

        print(
            f"Domain Separability: {domain_score:.4f}"
        )

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DAN LAMBDA SENSITIVITY RESULTS")

    print(
        f"{'Setting':15s}"
        f"{'Src Acc':>12s}"
        f"{'Src F1':>12s}"
        f"{'Target Acc':>14s}"
        f"{'Target F1':>14s}"
        f"{'Domain Sep.':>14s}"
    )

    for result in results:

        print(
            f"{result['method']:15s}"
            f"{result['source_accuracy']:12.4f}"
            f"{result['source_macro_f1']:12.4f}"
            f"{result['target_accuracy']:14.4f}"
            f"{result['target_macro_f1']:14.4f}"
            f"{result['domain_separability']:14.4f}"
        )


if __name__ == "__main__":
    main()