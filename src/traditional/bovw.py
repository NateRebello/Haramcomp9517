from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.validation import check_is_fitted


class BagOfVisualWords:
    """Bag of Visual Words representation using SIFT descriptors."""

    def __init__(
        self,
        n_clusters: int = 200,
        batch_size: int = 2048,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = n_clusters

        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
            n_init="auto",
        )

    def fit(self, descriptors: np.ndarray) -> None:
        """Learn the visual vocabulary from SIFT descriptors."""
        if descriptors.ndim != 2 or descriptors.shape[1] != 128:
            raise ValueError(
                "Descriptors must have shape (n_descriptors, 128)."
            )

        if len(descriptors) < self.n_clusters:
            raise ValueError(
                "The number of descriptors must be at least equal to "
                "the number of clusters."
            )

        self.kmeans.fit(descriptors)

    def transform_one(
        self,
        descriptors: np.ndarray | None,
    ) -> np.ndarray:
        """
        Convert one image's SIFT descriptors into a BoVW histogram.

        Args:
            descriptors:
                SIFT descriptor array with shape (n_keypoints, 128).
                If None, an all-zero histogram is returned.

        Returns:
            Normalized histogram with shape (n_clusters,).
        """
        check_is_fitted(self.kmeans)

        histogram = np.zeros(self.n_clusters, dtype=np.float32)

        if descriptors is None or len(descriptors) == 0:
            return histogram

        if descriptors.ndim != 2 or descriptors.shape[1] != 128:
            raise ValueError(
                "Descriptors must have shape (n_descriptors, 128)."
            )

        visual_words = self.kmeans.predict(descriptors)

        histogram = np.bincount(
            visual_words,
            minlength=self.n_clusters,
        ).astype(np.float32)

        histogram_sum = histogram.sum()

        if histogram_sum > 0:
            histogram /= histogram_sum

        return histogram

    def transform(
        self,
        descriptors_per_image: list[np.ndarray | None],
    ) -> np.ndarray:
        """
        Convert descriptors from multiple images into BoVW histograms.

        Args:
            descriptors_per_image:
                One SIFT descriptor array per image.

        Returns:
            Feature matrix with shape (n_images, n_clusters).
        """
        if not descriptors_per_image:
            return np.empty(
                (0, self.n_clusters),
                dtype=np.float32,
            )

        histograms = [
            self.transform_one(descriptors)
            for descriptors in descriptors_per_image
        ]

        return np.vstack(histograms)

    def fit_transform(
        self,
        vocabulary_descriptors: np.ndarray,
        descriptors_per_image: list[np.ndarray | None],
    ) -> np.ndarray:
        """
        Learn the vocabulary and transform image descriptors.

        Args:
            vocabulary_descriptors:
                Sampled SIFT descriptors used to fit MiniBatchKMeans.

            descriptors_per_image:
                SIFT descriptors for each image to encode.

        Returns:
            BoVW feature matrix with shape (n_images, n_clusters).
        """
        self.fit(vocabulary_descriptors)

        return self.transform(descriptors_per_image)

    def save(self, file_path: str | Path) -> None:
        """Save the fitted BoVW model to disk."""
        check_is_fitted(self.kmeans)

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str | Path) -> "BagOfVisualWords":
        """Load a saved BoVW model from disk."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"BoVW model file does not exist: {file_path}"
            )

        model = joblib.load(file_path)

        if not isinstance(model, cls):
            raise TypeError(
                f"The file does not contain a {cls.__name__} model."
            )

        check_is_fitted(model.kmeans)

        return model