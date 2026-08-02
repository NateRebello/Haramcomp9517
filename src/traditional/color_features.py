"""Extract color and texture features for the EASY traditional baseline.

Mirrors the interface of features.py's SIFTExtractor so both feature
extractors can be swapped into the same kind of pipeline.
"""
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm


class ColorLBPExtractor:
    """Extract a concatenated color-histogram + LBP texture feature vector."""

    def __init__(
        self,
        image_size: int = 128,
        color_bins: int = 32,
        lbp_radius: int = 2,
        lbp_points_per_radius: int = 8,
    ) -> None:
        self.image_size = image_size
        self.color_bins = color_bins
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_points_per_radius * lbp_radius

    @property
    def feature_dim(self) -> int:
        """Total length of the feature vector: 3 color channels + LBP histogram."""
        return 3 * self.color_bins + (self.lbp_n_points + 2)

    def _color_histogram(self, image_bgr: np.ndarray) -> np.ndarray:
        """Concatenated per-channel color histogram, L1-normalized."""
        channels = []
        for channel in range(3):
            hist = cv2.calcHist([image_bgr], [channel], None, [self.color_bins], [0, 256])
            channels.append(hist.flatten())
        histogram = np.concatenate(channels)
        total = histogram.sum()
        return histogram / total if total > 0 else histogram

    def _lbp_histogram(self, image_gray: np.ndarray) -> np.ndarray:
        """Uniform LBP texture histogram, L1-normalized."""
        lbp = local_binary_pattern(image_gray, self.lbp_n_points, self.lbp_radius, method="uniform")
        n_bins = self.lbp_n_points + 2
        histogram, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        total = histogram.sum()
        return histogram / total if total > 0 else histogram

    def extract(self, image_path: str | Path) -> np.ndarray:
        """
        Read an image and extract its color + LBP feature vector.

        Args:
            image_path: Path to the image.

        Returns:
            Feature vector with shape (feature_dim,).

        Raises:
            FileNotFoundError: If the image cannot be read.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.resize(image, (self.image_size, self.image_size))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return np.concatenate(
            [self._color_histogram(image), self._lbp_histogram(gray)]
        ).astype(np.float32)

    def extract_from_dataset(
        self,
        image_paths: Iterable[str | Path],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract features for multiple images.

        Args:
            image_paths: Paths to the images.
            show_progress: Whether to display a progress bar.

        Returns:
            Feature matrix with shape (n_images, feature_dim). Images that
            fail to read fall back to an all-zero feature vector (with a
            printed warning) so a single bad file doesn't crash a long run.
        """
        image_paths = list(image_paths)

        iterator = tqdm(
            image_paths,
            desc="Extracting color/LBP features",
            disable=not show_progress,
        )

        features = []
        for image_path in iterator:
            try:
                features.append(self.extract(image_path))
            except FileNotFoundError as error:
                print(f"Warning: {error}")
                features.append(np.zeros(self.feature_dim, dtype=np.float32))

        return np.vstack(features)


if __name__ == "__main__":
    from dataset import load_dataset

    dataset = load_dataset("data/subset")

    sample_path = dataset["train_paths"][0]

    extractor = ColorLBPExtractor()
    feature = extractor.extract(sample_path)

    print(f"Image: {sample_path}")
    print(f"Feature shape: {feature.shape}")
    print(f"Feature dtype: {feature.dtype}")
