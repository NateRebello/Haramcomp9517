"""
Build a category_id -> species_name lookup CSV from the official
iNaturalist-2021 annotation file (train_mini.json or val.json -- either
works, they share the same "categories" list).

Usage:
    python build_species_names.py --annotations train_mini.json --out species_names.csv
"""
import argparse

import pandas as pd
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to train_mini.json or val.json",
    )
    parser.add_argument("--out", default="species_names.csv")
    args = parser.parse_args()

    with open(args.annotations, "r") as f:
        data = json.load(f)

    rows = []
    for category in data["categories"]:
        rows.append(
            {
                "category_id": category["id"],
                "species_name": category.get("name", str(category["id"])),
            }
        )

    df = pd.DataFrame(rows).sort_values("category_id")
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} category names to {args.out}")


if __name__ == "__main__":
    main()
