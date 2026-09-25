import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

NOTEBOOK_PATH = Path("/content/drive/MyDrive/Untitled2.ipynb")

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/task2/results/curves"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def get_output_text(cell):
    """
    Extract all textual output from a notebook cell.
    """
    outputs = cell.get("outputs", [])

    text_parts = []

    for output in outputs:

        # Normal printed output
        if "text" in output:
            text = output["text"]

            if isinstance(text, list):
                text = "".join(text)

            text_parts.append(text)

        # Stream output
        if output.get("output_type") == "stream":
            text = output.get("text", "")

            if isinstance(text, list):
                text = "".join(text)

            text_parts.append(text)

    return "\n".join(text_parts)


def get_cell_source(cell):
    """
    Return notebook cell source as a string.
    """
    source = cell.get("source", "")

    if isinstance(source, list):
        return "".join(source)

    return source


def find_method_cells(notebook, method_keywords):
    """
    Find cells whose source/output contains method keywords.
    """
    matches = []

    for cell in notebook["cells"]:

        source = get_cell_source(cell)
        output = get_output_text(cell)

        combined = source + "\n" + output

        if all(
            keyword.lower() in combined.lower()
            for keyword in method_keywords
        ):
            matches.append(combined)

    return matches


# ============================================================
# Regex parsers
# ============================================================

def parse_erm(text):
    """
    Parse ERM output.

    Expected format:

    Epoch 01 | Loss: 1.2345 | Mean Source Macro-F1: 0.8000
    """

    pattern = re.compile(
        r"Epoch\s+(\d+)\s*\|\s*"
        r"Loss:\s*([0-9.eE+-]+)\s*\|\s*"
        r"Mean Source Macro-F1:\s*([0-9.eE+-]+)"
    )

    records = []

    for match in pattern.finditer(text):

        epoch = int(match.group(1))
        loss = float(match.group(2))
        macro_f1 = float(match.group(3))

        records.append({
            "epoch": epoch,
            "loss": loss,
            "macro_f1": macro_f1
        })

    return records


def parse_dan(text):
    """
    Parse DAN output.

    Expected format:

    Epoch 01 | Total Loss: ... | Cls: ... | MMD: ... |
    Mean Source Macro-F1: ...
    """

    pattern = re.compile(
        r"Epoch\s+(\d+)\s*\|\s*"
        r"Total Loss:\s*([0-9.eE+-]+)\s*\|\s*"
        r"Cls:\s*([0-9.eE+-]+)\s*\|\s*"
        r"MMD:\s*([0-9.eE+-]+)\s*\|\s*"
        r"Mean Source Macro-F1:\s*([0-9.eE+-]+)"
    )

    records = []

    for match in pattern.finditer(text):

        records.append({
            "epoch": int(match.group(1)),
            "total_loss": float(match.group(2)),
            "cls_loss": float(match.group(3)),
            "alignment_loss": float(match.group(4)),
            "macro_f1": float(match.group(5))
        })

    return records


def parse_dann(text):
    """
    Parse DANN output.

    Expected format:

    Epoch 01 | Total Loss: ... | Cls: ... |
    Domain: ... | Alpha: ... |
    Mean Source Macro-F1: ...
    """

    pattern = re.compile(
        r"Epoch\s+(\d+)\s*\|\s*"
        r"Total Loss:\s*([0-9.eE+-]+)\s*\|\s*"
        r"Cls:\s*([0-9.eE+-]+)\s*\|\s*"
        r"Domain:\s*([0-9.eE+-]+)\s*\|\s*"
        r"Alpha:\s*([0-9.eE+-]+)\s*\|\s*"
        r"Mean Source Macro-F1:\s*([0-9.eE+-]+)"
    )

    records = []

    for match in pattern.finditer(text):

        records.append({
            "epoch": int(match.group(1)),
            "total_loss": float(match.group(2)),
            "cls_loss": float(match.group(3)),
            "domain_loss": float(match.group(4)),
            "alpha": float(match.group(5)),
            "macro_f1": float(match.group(6))
        })

    return records


def parse_cdan(text):
    """
    Parse CDAN output.

    CDAN uses the same printed format as DANN.
    """

    return parse_dann(text)


# ============================================================
# Remove duplicate epoch records
# ============================================================

def clean_records(records):
    """
    Keep one record per epoch.
    """
    unique = {}

    for record in records:
        unique[record["epoch"]] = record

    return [
        unique[epoch]
        for epoch in sorted(unique)
    ]


# ============================================================
# Find method-specific training output
# ============================================================

def collect_method_output(notebook, method):
    """
    Collect output belonging to one training method.

    We identify the training section using the printed
    method name and epoch-format output.
    """

    texts = []

    for cell in notebook["cells"]:

        source = get_cell_source(cell)
        output = get_output_text(cell)

        combined = source + "\n" + output

        # Only consider cells containing epoch output.
        if "Epoch 01" not in combined and "Epoch 1" not in combined:
            continue

        lower = combined.lower()

        if method == "ERM":

            if (
                "source_only" in lower
                or "train_source_only" in lower
                or "mean source macro-f1" in lower
            ):
                texts.append(combined)

        elif method == "DAN":

            if (
                "mmd" in lower
                and "train_dan" in lower
            ):
                texts.append(combined)

        elif method == "DANN":

            if (
                "domain:" in lower
                and "alpha:" in lower
                and "train_dann" in lower
            ):
                texts.append(combined)

        elif method == "CDAN":

            if (
                "domain:" in lower
                and "alpha:" in lower
                and "train_cdan" in lower
            ):
                texts.append(combined)

    return "\n".join(texts)


# ============================================================
# Plot ERM
# ============================================================

def plot_erm(records):

    if not records:
        print("No ERM records found.")
        return

    epochs = [r["epoch"] for r in records]
    loss = [r["loss"] for r in records]
    f1 = [r["macro_f1"] for r in records]

    fig, ax1 = plt.subplots()

    ax1.plot(
        epochs,
        loss,
        marker="o",
        label="Training CE Loss"
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training CE Loss")
    ax1.grid(True)

    ax2 = ax1.twinx()

    ax2.plot(
        epochs,
        f1,
        marker="s",
        label="Mean Source Macro-F1"
    )

    ax2.set_ylabel("Mean Source Macro-F1")

    fig.suptitle("ERM Training Curves")

    fig.tight_layout()

    path = OUTPUT_DIR / "erm_training_curves.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# Plot DAN
# ============================================================

def plot_dan(records):

    if not records:
        print("No DAN records found.")
        return

    epochs = [r["epoch"] for r in records]

    total = [r["total_loss"] for r in records]
    cls = [r["cls_loss"] for r in records]
    mmd = [r["alignment_loss"] for r in records]
    f1 = [r["macro_f1"] for r in records]

    # --------------------------------------------------------
    # Loss curves
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        total,
        marker="o",
        label="Total Loss"
    )

    ax.plot(
        epochs,
        cls,
        marker="s",
        label="Classification Loss"
    )

    ax.plot(
        epochs,
        mmd,
        marker="^",
        label="MMD Loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DAN Training Losses")
    ax.legend()
    ax.grid(True)

    path = OUTPUT_DIR / "dan_losses.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

    # --------------------------------------------------------
    # Validation curve
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        f1,
        marker="o",
        label="Mean Source Macro-F1"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("DAN Source Validation Macro-F1")
    ax.legend()
    ax.grid(True)

    path = OUTPUT_DIR / "dan_source_macro_f1.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# Plot DANN
# ============================================================

def plot_dann(records):

    if not records:
        print("No DANN records found.")
        return

    epochs = [r["epoch"] for r in records]

    total = [r["total_loss"] for r in records]
    cls = [r["cls_loss"] for r in records]
    domain = [r["domain_loss"] for r in records]
    alpha = [r["alpha"] for r in records]
    f1 = [r["macro_f1"] for r in records]

    # --------------------------------------------------------
    # Loss curves
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        total,
        marker="o",
        label="Total Loss"
    )

    ax.plot(
        epochs,
        cls,
        marker="s",
        label="Classification Loss"
    )

    ax.plot(
        epochs,
        domain,
        marker="^",
        label="Domain Loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DANN Training Losses")
    ax.legend()
    ax.grid(True)

    path = OUTPUT_DIR / "dann_losses.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

    # --------------------------------------------------------
    # Alpha
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        alpha,
        marker="o"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("GRL Alpha")
    ax.set_title("DANN GRL Schedule")
    ax.grid(True)

    path = OUTPUT_DIR / "dann_alpha.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

    # --------------------------------------------------------
    # Source Macro-F1
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        f1,
        marker="o"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Source Macro-F1")
    ax.set_title("DANN Source Validation Macro-F1")
    ax.grid(True)

    path = OUTPUT_DIR / "dann_source_macro_f1.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# Plot CDAN
# ============================================================

def plot_cdan(records):

    if not records:
        print("No CDAN records found.")
        return

    epochs = [r["epoch"] for r in records]

    total = [r["total_loss"] for r in records]
    cls = [r["cls_loss"] for r in records]
    domain = [r["domain_loss"] for r in records]
    alpha = [r["alpha"] for r in records]
    f1 = [r["macro_f1"] for r in records]

    # --------------------------------------------------------
    # Loss curves
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        total,
        marker="o",
        label="Total Loss"
    )

    ax.plot(
        epochs,
        cls,
        marker="s",
        label="Classification Loss"
    )

    ax.plot(
        epochs,
        domain,
        marker="^",
        label="Domain Loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CDAN Training Losses")
    ax.legend()
    ax.grid(True)

    path = OUTPUT_DIR / "cdan_losses.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

    # --------------------------------------------------------
    # Alpha
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        alpha,
        marker="o"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("GRL Alpha")
    ax.set_title("CDAN GRL Schedule")
    ax.grid(True)

    path = OUTPUT_DIR / "cdan_alpha.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")

    # --------------------------------------------------------
    # Source Macro-F1
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        f1,
        marker="o"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Source Macro-F1")
    ax.set_title("CDAN Source Validation Macro-F1")
    ax.grid(True)

    path = OUTPUT_DIR / "cdan_source_macro_f1.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Reading notebook")
    print("=" * 70)

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    print(f"Notebook: {NOTEBOOK_PATH}")

    # --------------------------------------------------------
    # ERM
    # --------------------------------------------------------

    erm_text = collect_method_output(
        notebook,
        "ERM"
    )

    erm_records = clean_records(
        parse_erm(erm_text)
    )

    print(
        f"\nERM epochs extracted: "
        f"{len(erm_records)}"
    )

    if erm_records:
        print(
            f"Epoch range: "
            f"{erm_records[0]['epoch']} - "
            f"{erm_records[-1]['epoch']}"
        )

    # --------------------------------------------------------
    # DAN
    # --------------------------------------------------------

    dan_text = collect_method_output(
        notebook,
        "DAN"
    )

    dan_records = clean_records(
        parse_dan(dan_text)
    )

    print(
        f"\nDAN epochs extracted: "
        f"{len(dan_records)}"
    )

    if dan_records:
        print(
            f"Epoch range: "
            f"{dan_records[0]['epoch']} - "
            f"{dan_records[-1]['epoch']}"
        )

    # --------------------------------------------------------
    # DANN
    # --------------------------------------------------------

    dann_text = collect_method_output(
        notebook,
        "DANN"
    )

    dann_records = clean_records(
        parse_dann(dann_text)
    )

    print(
        f"\nDANN epochs extracted: "
        f"{len(dann_records)}"
    )

    if dann_records:
        print(
            f"Epoch range: "
            f"{dann_records[0]['epoch']} - "
            f"{dann_records[-1]['epoch']}"
        )

    # --------------------------------------------------------
    # CDAN
    # --------------------------------------------------------

    cdan_text = collect_method_output(
        notebook,
        "CDAN"
    )

    cdan_records = clean_records(
        parse_cdan(cdan_text)
    )

    print(
        f"\nCDAN epochs extracted: "
        f"{len(cdan_records)}"
    )

    if cdan_records:
        print(
            f"Epoch range: "
            f"{cdan_records[0]['epoch']} - "
            f"{cdan_records[-1]['epoch']}"
        )

    # --------------------------------------------------------
    # Save extracted data
    # --------------------------------------------------------

    extracted = {
        "ERM": erm_records,
        "DAN": dan_records,
        "DANN": dann_records,
        "CDAN": cdan_records
    }

    json_path = OUTPUT_DIR / "training_history_extracted.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            extracted,
            f,
            indent=2
        )

    print(
        f"\nSaved extracted data: "
        f"{json_path}"
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print("\nGenerating plots...")

    plot_erm(erm_records)
    plot_dan(dan_records)
    plot_dann(dann_records)
    plot_cdan(cdan_records)

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)

    print(
        f"\nAll plots saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()