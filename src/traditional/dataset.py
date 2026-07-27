from pathlib import Path

import numpy as np


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_images(folder: str | Path) -> list[Path]:
    """Recursively find all supported images inside a folder."""
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_split(
    split_dir: str | Path,
    class_to_index: dict[str, int] | None = None,
) -> tuple[list[Path], np.ndarray, dict[str, int]]:
    """
    Load one dataset split arranged into class folders.

    Expected structure:

        split_dir/
            4/
                image_1.jpg
                image_2.jpg
            26/
                image_3.jpg

    Args:
        split_dir:
            Path to the train, validation, or test directory.

        class_to_index:
            Existing class mapping. Pass the training mapping when loading
            validation and test data so all splits use identical labels.

    Returns:
        image_paths:
            Paths to all images in the split.

        labels:
            Integer labels corresponding to each image.

        class_to_index:
            Mapping from folder name to integer label.
    """
    split_dir = Path(split_dir)

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split does not exist: {split_dir}"
        )

    class_folders = sorted(
        (folder for folder in split_dir.iterdir() if folder.is_dir()),
        key=lambda folder: int(folder.name)
        if folder.name.isdigit()
        else folder.name,
    )

    if not class_folders:
        raise ValueError(f"No class folders found inside: {split_dir}")

    if class_to_index is None:
        class_to_index = {
            folder.name: index
            for index, folder in enumerate(class_folders)
        }

    image_paths: list[Path] = []
    labels: list[int] = []

    for class_folder in class_folders:
        class_name = class_folder.name

        if class_name not in class_to_index:
            print(
                f"Warning: skipping unknown class '{class_name}' "
                f"in {split_dir}"
            )
            continue

        class_images = find_images(class_folder)

        if not class_images:
            print(f"Warning: no images found in {class_folder}")
            continue

        label = class_to_index[class_name]

        image_paths.extend(class_images)
        labels.extend([label] * len(class_images))

    if not image_paths:
        raise ValueError(f"No supported images found inside: {split_dir}")

    return (
        image_paths,
        np.asarray(labels, dtype=np.int32),
        class_to_index,
    )


def load_dataset(
    dataset_root: str | Path,
) -> dict[str, object]:
    """
    Load the train, validation, and test splits.

    Expected root structure:

        dataset_root/
            train/
            val/
            test/
    """
    dataset_root = Path(dataset_root)

    train_paths, train_labels, class_to_index = load_split(
        dataset_root / "train"
    )

    val_paths, val_labels, _ = load_split(
        dataset_root / "val",
        class_to_index=class_to_index,
    )

    test_paths, test_labels, _ = load_split(
        dataset_root / "test",
        class_to_index=class_to_index,
    )

    return {
        "train_paths": train_paths,
        "train_labels": train_labels,
        "val_paths": val_paths,
        "val_labels": val_labels,
        "test_paths": test_paths,
        "test_labels": test_labels,
        "class_to_index": class_to_index,
    }


def print_dataset_summary(dataset: dict[str, object]) -> None:
    """Print basic information about the loaded dataset."""
    class_to_index = dataset["class_to_index"]

    print(f"Number of classes: {len(class_to_index)}")
    print(f"Training images: {len(dataset['train_paths'])}")
    print(f"Validation images: {len(dataset['val_paths'])}")
    print(f"Testing images: {len(dataset['test_paths'])}")

    print("\nFirst 10 class mappings:")

    for class_name, class_index in list(class_to_index.items())[:10]:
        print(f"{class_index:3d} -> folder {class_name}")

    if len(class_to_index) > 10:
        print(f"... and {len(class_to_index) - 10} more classes")


if __name__ == "__main__":
    dataset = load_dataset("data/subset")
    print_dataset_summary(dataset)