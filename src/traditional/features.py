from pathlib import Path

import cv2
import numpy as np


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

        Returns:
            keypoints: Detected SIFT keypoints.
            descriptors: Array of shape (n_keypoints, 128), or None if no
                keypoints were detected.

        Raises:
            FileNotFoundError: If the image cannot be read.
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