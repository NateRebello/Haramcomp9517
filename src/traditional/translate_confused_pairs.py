"""
Translate main_easy.py's most-confused-pairs output back into real species
names, for the report.

main_easy.py's confused-pairs printout is in "sequential index space" --
dataset.py's load_dataset() renumbers the raw iNat category-id folder names
(e.g. "9036", "4729") into 0..N-1 based on sort order. This script undoes
that: sequential index -> raw category id (via dataset.py's class_to_index)
-> species name (via species_names.csv, built by build_species_names.py).

Requires:
    - data/subset/ to already exist (only scans folder names, doesn't read
      any images, so this is fast)
    - species_names.csv, built by build_species_names.py
    - results/easy_traditional/{model_type}_results.joblib, saved by
      main_easy.py's Evaluator.save_results() call

Usage:
    python translate_confused_pairs.py --model-type svm --top-n 15
    python translate_confused_pairs.py --model-type random_forest --top-n 15
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from dataset import load_dataset
from evaluation import Evaluator


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
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--out", default=None, help="Optional CSV path to also save the table")
    args = parser.parse_args()

    print("Scanning dataset folder names (no images read)...")
    dataset = load_dataset(args.dataset_root)
    class_to_index = dataset["class_to_index"]  # {"9036": 0, "4729": 1, ...}
    index_to_category_id = {v: k for k, v in class_to_index.items()}

    names_df = pd.read_csv(args.species_names)
    names_df["category_id"] = names_df["category_id"].astype(str)
    category_id_to_name = dict(zip(names_df["category_id"], names_df["species_name"]))

    results_path = Path(args.results_dir) / f"{args.model_type}_results.joblib"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found -- run main_easy.py first so it can "
            "save this model's results."
        )
    results = joblib.load(results_path)

    n_classes = results["confusion_matrix"].shape[0]
    class_labels = list(range(n_classes))

    top_pairs = Evaluator.most_confused_pairs(results, class_labels, top_n=args.top_n)

    rows = []
    for true_idx, pred_idx, count in top_pairs:
        true_cat = index_to_category_id.get(true_idx, "?")
        pred_cat = index_to_category_id.get(pred_idx, "?")
        true_name = category_id_to_name.get(true_cat, f"category {true_cat}")
        pred_name = category_id_to_name.get(pred_cat, f"category {pred_cat}")
        rows.append(
            {
                "true_species": true_name,
                "true_category_id": true_cat,
                "predicted_species": pred_name,
                "predicted_category_id": pred_cat,
                "count": count,
            }
        )

    print(f"\nTop {args.top_n} most confused species pairs ({args.model_type}):")
    for row in rows:
        print(
            f"  {row['count']:3d}x  true={row['true_species']} (id {row['true_category_id']})"
            f"  ->  predicted={row['predicted_species']} (id {row['predicted_category_id']})"
        )

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nSaved table to {args.out}")


if __name__ == "__main__":
    main()
