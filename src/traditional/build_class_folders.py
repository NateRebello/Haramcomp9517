"""
Bridge script: turn the CSV manifests from prepare_dataset.py into the
folder-per-class layout that dataset.py's load_dataset() expects, e.g.

    data/subset/train/<category_id>/<image>.jpg
    data/subset/val/<category_id>/<image>.jpg
    data/subset/test/<category_id>/<image>.jpg

Run this once after prepare_dataset.py has produced manifests/{train,val,test}.csv.
Uses symlinks by default so it doesn't duplicate the images on disk; pass
--copy if your filesystem doesn't support symlinks (e.g. some Windows setups
without developer mode enabled).

Usage:
    python build_class_folders.py --manifest-dir manifests --out-dir data/subset
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def build_split(csv_path: Path, out_dir: Path, copy: bool) -> None:
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Linking {csv_path.name}"):
        src = Path(row["file_path"])
        class_dir = out_dir / str(row["category_id"])
        class_dir.mkdir(parents=True, exist_ok=True)
        dst = class_dir / src.name

        if dst.exists():
            continue

        if copy:
            shutil.copy2(src, dst)
            continue

        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="manifests")
    parser.add_argument("--out-dir", default="data/subset")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)

    for split in ("train", "val", "test"):
        csv_path = manifest_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing manifest: {csv_path}. Run prepare_dataset.py first.")
        build_split(csv_path, out_dir / split, args.copy)

    print(f"Class-folder dataset ready at {out_dir.resolve()}")


if __name__ == "__main__":
    main()
