from pathlib import Path
from collections.abc import Iterable

import cv2
import numpy as np
from tqdm import tqdm


class SIFTExtractor:
    """Extract SIFT keypoints and descriptors from images."""

    def __init__(self, max_features: int = 500) -> None:
        self.detector = cv2.SIFT_create(nfeatures=max_features)

    def extract(
        self,
        image_path: str | Path,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """
        Read an image and extract its SIFT features.

        Args:
            image_path:
                Path to the image.

        Returns:
            keypoints:
                Detected SIFT keypoints.

            descriptors:
                Array of shape (n_keypoints, 128), or None if no
                keypoints were detected.

        Raises:
            FileNotFoundError:
                If the image cannot be read.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.detector.detectAndCompute(
            grayscale,
            None,
        )

        return keypoints, descriptors

    def extract_descriptors(
        self,
        image_path: str | Path,
    ) -> np.ndarray | None:
        """
        Extract only the SIFT descriptors from one image.
        """
        _, descriptors = self.extract(image_path)
        return descriptors

    def extract_from_dataset(
        self,
        image_paths: Iterable[str | Path],
        show_progress: bool = True,
    ) -> list[np.ndarray | None]:
        """
        Extract SIFT descriptors from multiple images.

        Args:
            image_paths:
                Paths to the images.

            show_progress:
                Whether to display a progress bar.

        Returns:
            A list containing one descriptor array per image.
            Images with no detected features are represented by None.
        """
        image_paths = list(image_paths)

        iterator = tqdm(
            image_paths,
            desc="Extracting SIFT descriptors",
            disable=not show_progress,
        )

        descriptors_per_image: list[np.ndarray | None] = []

        for image_path in iterator:
            try:
                descriptors = self.extract_descriptors(image_path)
            except FileNotFoundError as error:
                print(f"Warning: {error}")
                descriptors = None

            descriptors_per_image.append(descriptors)

        return descriptors_per_image


if __name__ == "__main__":
    from dataset import load_dataset

    dataset = load_dataset("data/subset")

    sample_path = dataset["train_paths"][0]

    extractor = SIFTExtractor(max_features=500)
    keypoints, descriptors = extractor.extract(sample_path)

    print(f"Image: {sample_path}")
    print(f"Number of keypoints: {len(keypoints)}")

    if descriptors is None:
        print("No descriptors detected.")
    else:
        print(f"Descriptor shape: {descriptors.shape}")
        print(f"Descriptor dtype: {descriptors.dtype}")