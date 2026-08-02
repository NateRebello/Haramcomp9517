"""
Prepare a reproducible 500-species subset of iNaturalist-2021 (mini).

Expects the official annotation files from:
  https://github.com/visipedia/inat_comp/tree/master/2021
    train_mini.json  (annotations for the 500,000-image train_mini set)
    val.json         (annotations for the 100,000-image val set)

Outputs three manifest CSVs (image path, category id, species name):
    manifests/train.csv   -> 40 images/class  (used for training)
    manifests/val.csv     -> 10 images/class  (held out from train_mini, for tuning)
    manifests/test.csv    -> 10 images/class  (from the official val set, final eval only)

and manifests/class_list.txt documenting exactly which category ids were sampled,
for reproducibility (report this + the seed in the report).

Usage:
    python prepare_dataset.py \
        --train-json /path/to/train_mini.json \
        --val-json /path/to/val.json \
        --train-img-root /path/to/train_mini \
        --val-img-root /path/to/val \
        --n-classes 500 \
        --seed 42 \
        --out-dir manifests

Run this ONCE, commit manifests/ to the shared repo (or shared drive), and have
everyone (both teams) load features from these same three CSVs. That is what
keeps train/val/test strictly separate and results comparable across models.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_annotations(json_path):
    """Returns (by_category: {category_id: [file_name, ...]}, cat_id_to_name: {id: name})."""
    with open(json_path, "r") as f:
        data = json.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    id_to_cat = {ann["image_id"]: ann["category_id"] for ann in data["annotations"]}
    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in data["categories"]}

    by_category = defaultdict(list)
    for image_id, file_name in id_to_file.items():
        cat_id = id_to_cat.get(image_id)
        if cat_id is not None:
            by_category[cat_id].append(file_name)
    return by_category, cat_id_to_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", required=True, help="Path to train_mini.json")
    parser.add_argument("--val-json", required=True, help="Path to val.json")
    parser.add_argument("--train-img-root", required=True, help="Root dir of extracted train_mini images")
    parser.add_argument("--val-img-root", required=True, help="Root dir of extracted val images")
    parser.add_argument("--n-classes", type=int, default=500)
    parser.add_argument("--n-train-per-class", type=int, default=40)
    parser.add_argument("--n-val-per-class", type=int, default=10)
    parser.add_argument("--n-test-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="manifests")
    args = parser.parse_args()

    random.seed(args.seed)

    train_by_cat, cat_names = load_annotations(args.train_json)
    val_by_cat, _ = load_annotations(args.val_json)

    all_classes = sorted(set(train_by_cat) & set(val_by_cat))
    if len(all_classes) < args.n_classes:
        raise ValueError(f"Only {len(all_classes)} classes have both train and val images.")
    chosen = sorted(random.sample(all_classes, args.n_classes))

    train_rows, val_rows, test_rows = [], [], []
    for cat_id in chosen:
        files = train_by_cat[cat_id][:]
        random.shuffle(files)
        need = args.n_train_per_class + args.n_val_per_class
        if len(files) < need:
            raise ValueError(f"Category {cat_id} has only {len(files)} train_mini images, need {need}.")
        train_files = files[: args.n_train_per_class]
        val_files = files[args.n_train_per_class: need]

        test_files = val_by_cat[cat_id][:]
        random.shuffle(test_files)
        if len(test_files) < args.n_test_per_class:
            raise ValueError(f"Category {cat_id} has only {len(test_files)} val images, need {args.n_test_per_class}.")
        test_files = test_files[: args.n_test_per_class]

        name = cat_names.get(cat_id, str(cat_id))
        for fn in train_files:
            train_rows.append((str(Path(args.train_img_root) / fn), cat_id, name))
        for fn in val_files:
            val_rows.append((str(Path(args.train_img_root) / fn), cat_id, name))
        for fn in test_files:
            test_rows.append((str(Path(args.val_img_root) / fn), cat_id, name))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["file_path", "category_id", "species_name"]
    pd.DataFrame(train_rows, columns=cols).to_csv(out_dir / "train.csv", index=False)
    pd.DataFrame(val_rows, columns=cols).to_csv(out_dir / "val.csv", index=False)
    pd.DataFrame(test_rows, columns=cols).to_csv(out_dir / "test.csv", index=False)

    with open(out_dir / "class_list.txt", "w") as f:
        f.write(f"# seed={args.seed} n_classes={len(chosen)}\n")
        for cat_id in chosen:
            f.write(f"{cat_id}\t{cat_names.get(cat_id, '')}\n")

    print(f"Seed: {args.seed}")
    print(f"Classes: {len(chosen)}")
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")
    print(f"Manifests written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
