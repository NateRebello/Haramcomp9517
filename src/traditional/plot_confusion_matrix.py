"""
Plot confusion-matrix visualizations for a trained traditional model, per the
spec's requirement to place "particular emphasis on macro-averaged F1 and a
confusion-matrix analysis (e.g., a visualization... for all classes and more
detailed analyses for selected subset of classes)".

Produces two figures:
    1. A full NxN heatmap (row-normalized) across every class -- shows the
       overall diagonal structure / how sparse or diffuse the errors are.
    2. A zoomed-in heatmap over just the classes involved in the top
       confused pairs, labeled with real species names -- readable at a
       glance, good for the report's error-analysis discussion.

Works for either pipeline's results, not just the easy model -- point
--results-dir / --model-type at whichever *_results.joblib you want to plot
(e.g. Brandon's BoVW results, once saved in the same dict shape via
Evaluator.evaluate()).

Requires:
    - <results-dir>/{model_type}_results.joblib (a dict with a
      "confusion_matrix" key, as saved by Evaluator.evaluate())
    - data/subset/ (only scans folder names, doesn't read images)
    - species_names.csv (from build_species_names.py)

Usage:
    python plot_confusion_matrix.py --model-type svm --top-n-classes 20
    python plot_confusion_matrix.py --model-type random_forest --top-n-classes 20
"""
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import load_dataset
from evaluation import Evaluator


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix so each row sums to 1 (recall view)."""
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def plot_full_matrix(cm: np.ndarray, model_type: str, out_path: Path) -> None:
    normalized = normalize_rows(cm)

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Row-normalized frequency (recall)")
    ax.set_xlabel("Predicted class (index)")
    ax.set_ylabel("True class (index)")
    ax.set_title(f"Confusion matrix -- all {cm.shape[0]} classes ({model_type})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved full confusion matrix to {out_path}")


def plot_top_classes(
    cm: np.ndarray,
    class_labels: list[int],
    index_to_category_id: dict[int, str],
    category_id_to_name: dict[str, str],
    model_type: str,
    top_n_classes: int,
    top_n_pairs: int,
    out_path: Path,
) -> None:
    # Reuse Evaluator.most_confused_pairs -- it only reads the
    # "confusion_matrix" key, so a minimal stub dict is enough here.
    top_pairs = Evaluator.most_confused_pairs(
        {"confusion_matrix": cm}, class_labels, top_n=top_n_pairs
    )

    involved_indices: list[int] = []
    for true_idx, pred_idx, _ in top_pairs:
        for idx in (true_idx, pred_idx):
            if idx not in involved_indices:
                involved_indices.append(idx)
        if len(involved_indices) >= top_n_classes:
            break
    involved_indices = involved_indices[:top_n_classes]

    sub_cm = cm[np.ix_(involved_indices, involved_indices)]
    normalized = normalize_rows(sub_cm)

    labels = []
    for idx in involved_indices:
        cat_id = index_to_category_id.get(idx, "?")
        name = category_id_to_name.get(cat_id, f"id {cat_id}")
        labels.append(name)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Row-normalized frequency (recall)")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title(f"Confusion matrix -- most-confused species subset ({model_type})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved zoomed confusion matrix to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/subset")
    parser.add_argument("--results-dir", default="results/easy_traditional")
    parser.add_argument(
        "--model-type",
        default="svm",
        choices=["svm", "random_forest"],
    )
    parser.add_argument("--species-names", default="species_names.csv")
    parser.add_argument("--top-n-classes", type=int, default=20)
    parser.add_argument("--top-n-pairs", type=int, default=30)
    args = parser.parse_args()

    results_path = Path(args.results_dir) / f"{args.model_type}_results.joblib"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found -- run main_easy.py first so it can "
            "save this model's results."
        )
    results = joblib.load(results_path)
    cm = results["confusion_matrix"]
    class_labels = list(range(cm.shape[0]))

    full_out = Path(args.results_dir) / f"confusion_matrix_{args.model_type}_full.png"
    plot_full_matrix(cm, args.model_type, full_out)

    print("Scanning dataset folder names (no images read)...")
    dataset = load_dataset(args.dataset_root)
    class_to_index = dataset["class_to_index"]
    index_to_category_id = {v: k for k, v in class_to_index.items()}

    names_df = pd.read_csv(args.species_names)
    names_df["category_id"] = names_df["category_id"].astype(str)
    category_id_to_name = dict(zip(names_df["category_id"], names_df["species_name"]))

    zoomed_out = Path(args.results_dir) / f"confusion_matrix_{args.model_type}_top_classes.png"
    plot_top_classes(
        cm,
        class_labels,
        index_to_category_id,
        category_id_to_name,
        args.model_type,
        args.top_n_classes,
        args.top_n_pairs,
        zoomed_out,
    )


if __name__ == "__main__":
    main()
