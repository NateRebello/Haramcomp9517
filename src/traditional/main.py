from pathlib import Path

import joblib
import numpy as np

from bovw import BagOfVisualWords
from dataset import load_dataset
from features import SIFTExtractor


def sample_vocabulary_descriptors(
    descriptors_per_image: list[np.ndarray | None],
    max_descriptors: int = 100_000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Combine and randomly sample SIFT descriptors for vocabulary training.

    Args:
        descriptors_per_image:
            One descriptor array per image.

        max_descriptors:
            Maximum number of descriptors used to fit MiniBatchKMeans.

        random_state:
            Random seed for reproducibility.

    Returns:
        Descriptor array with shape (n_sampled_descriptors, 128).
    """
    valid_descriptors = [
        descriptors
        for descriptors in descriptors_per_image
        if descriptors is not None and len(descriptors) > 0
    ]

    if not valid_descriptors:
        raise ValueError("No valid SIFT descriptors were extracted.")

    all_descriptors = np.vstack(valid_descriptors).astype(np.float32)

    if len(all_descriptors) <= max_descriptors:
        return all_descriptors

    rng = np.random.default_rng(random_state)

    selected_indices = rng.choice(
        len(all_descriptors),
        size=max_descriptors,
        replace=False,
    )

    return all_descriptors[selected_indices]


def main() -> None:
    dataset_root = Path("data/subset")

    print("Loading dataset...")
    dataset = load_dataset(dataset_root)

    train_paths = dataset["train_paths"]
    train_labels = np.asarray(
        dataset["train_labels"],
        dtype=np.int32,
    )

    val_paths = dataset["val_paths"]
    val_labels = np.asarray(
        dataset["val_labels"],
        dtype=np.int32,
    )

    test_paths = dataset["test_paths"]
    test_labels = np.asarray(
        dataset["test_labels"],
        dtype=np.int32,
    )

    random_state = 42

    print(f"Training images used: {len(train_paths)}")
    print(f"Validation images used: {len(val_paths)}")
    print(f"Testing images used: {len(test_paths)}")

    print("\nExtracting SIFT descriptors...")

    extractor = SIFTExtractor(max_features=500)

    train_descriptors = extractor.extract_from_dataset(
        train_paths
    )

    val_descriptors = extractor.extract_from_dataset(
        val_paths
    )

    test_descriptors = extractor.extract_from_dataset(
        test_paths
    )
    
    descriptor_path = Path(
        "results/sift_descriptors_500_classes.joblib"
    )

    descriptor_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "train_descriptors": train_descriptors,
            "train_labels": train_labels,
            "val_descriptors": val_descriptors,
            "val_labels": val_labels,
            "test_descriptors": test_descriptors,
            "test_labels": test_labels,
        },
        descriptor_path,
    )

    print(
        f"SIFT descriptors saved to: {descriptor_path}"
    )
    print(
        "\nSampling descriptors for the visual vocabulary..."
    )

    vocabulary_descriptors = sample_vocabulary_descriptors(
        train_descriptors,
        max_descriptors=100_000,
        random_state=random_state,
    )

    print(
        "Vocabulary descriptor shape: "
        f"{vocabulary_descriptors.shape}"
    )

    print("\nFitting Bag of Visual Words...")

    bovw = BagOfVisualWords(
        n_clusters=200,
        batch_size=2048,
        random_state=random_state,
    )

    bovw.fit(vocabulary_descriptors)

    print("\nGenerating BoVW histograms...")

    train_features = bovw.transform(
        train_descriptors
    )

    val_features = bovw.transform(
        val_descriptors
    )

    test_features = bovw.transform(
        test_descriptors
    )

    print(
        f"Train feature shape: {train_features.shape}"
    )

    print(
        f"Validation feature shape: {val_features.shape}"
    )

    print(
        f"Test feature shape: {test_features.shape}"
    )

    model_path = Path("models/bovw_200_500_classes.joblib")

    bovw.save(model_path)

    print(
        f"\nBoVW vocabulary saved to: {model_path}"
    )

    feature_path = Path(
        "results/bovw_features_500_classes.npz"
    )

    feature_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        feature_path,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
    )

    print(
        f"Feature matrices saved to: {feature_path}"
    )


if __name__ == "__main__":
    main()