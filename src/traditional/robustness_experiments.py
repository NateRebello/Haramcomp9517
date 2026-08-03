from pathlib import Path
import csv
import time

import cv2
import numpy as np

from bovw import BagOfVisualWords
from classifier import TraditionalClassifier
from dataset import load_dataset
from evaluation import Evaluator


RANDOM_STATE = 42
MAX_SIFT_FEATURES = 500


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape,
    )

    noisy_image = image.astype(np.float32) + noise

    return np.clip(
        noisy_image,
        0,
        255,
    ).astype(np.uint8)


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """Apply Gaussian blur."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "Gaussian blur kernel size must be a positive odd integer."
        )

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=0,
    )


def adjust_brightness(
    image: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Adjust image brightness using a multiplication factor."""
    if factor < 0:
        raise ValueError(
            "Brightness factor must be non-negative."
        )

    adjusted = image.astype(np.float32) * factor

    return np.clip(
        adjusted,
        0,
        255,
    ).astype(np.uint8)


def apply_jpeg_compression(
    image: np.ndarray,
    quality: int,
) -> np.ndarray:
    """Apply JPEG compression and decode the image again."""
    if quality < 1 or quality > 100:
        raise ValueError(
            "JPEG quality must be between 1 and 100."
        )

    success, encoded_image = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise RuntimeError(
            "JPEG compression failed."
        )

    decoded_image = cv2.imdecode(
        encoded_image,
        cv2.IMREAD_COLOR,
    )

    if decoded_image is None:
        raise RuntimeError(
            "JPEG decompression failed."
        )

    return decoded_image


def apply_corruption(
    image: np.ndarray,
    corruption: str,
    severity_value: float | int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one corruption to an image."""
    if corruption == "clean":
        return image

    if corruption == "gaussian_noise":
        if severity_value is None:
            raise ValueError(
                "Noise severity must be provided."
            )

        return add_gaussian_noise(
            image,
            sigma=float(severity_value),
            rng=rng,
        )

    if corruption == "gaussian_blur":
        if severity_value is None:
            raise ValueError(
                "Blur severity must be provided."
            )

        return apply_gaussian_blur(
            image,
            kernel_size=int(severity_value),
        )

    if corruption == "brightness":
        if severity_value is None:
            raise ValueError(
                "Brightness severity must be provided."
            )

        return adjust_brightness(
            image,
            factor=float(severity_value),
        )

    if corruption == "jpeg":
        if severity_value is None:
            raise ValueError(
                "JPEG quality must be provided."
            )

        return apply_jpeg_compression(
            image,
            quality=int(severity_value),
        )

    raise ValueError(
        f"Unknown corruption: {corruption}"
    )


def extract_sift_descriptors(
    image: np.ndarray,
    sift: cv2.SIFT,
) -> np.ndarray | None:
    """Extract SIFT descriptors from an image."""
    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    _, descriptors = sift.detectAndCompute(
        gray_image,
        None,
    )

    if descriptors is None or len(descriptors) == 0:
        return None

    return descriptors.astype(np.float32)


def generate_corrupted_descriptors(
    image_paths: list[Path],
    corruption: str,
    severity_value: float | int | None,
    random_state: int,
) -> list[np.ndarray | None]:
    """Load images, corrupt them, and extract SIFT descriptors."""
    sift = cv2.SIFT_create(
        nfeatures=MAX_SIFT_FEATURES,
    )

    descriptors_per_image: list[np.ndarray | None] = []

    rng = np.random.default_rng(
        random_state
    )

    total_images = len(image_paths)

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        # Separate deterministic random stream per image.
        image_rng = np.random.default_rng(
            rng.integers(
                0,
                2**32 - 1,
            )
        )

        corrupted_image = apply_corruption(
            image=image,
            corruption=corruption,
            severity_value=severity_value,
            rng=image_rng,
        )

        descriptors = extract_sift_descriptors(
            corrupted_image,
            sift,
        )

        descriptors_per_image.append(
            descriptors
        )

        if index % 100 == 0 or index == total_images:
            print(
                f"Processed {index}/{total_images} images"
            )

    return descriptors_per_image


def evaluate_condition(
    test_paths: list[Path],
    test_labels: np.ndarray,
    bovw: BagOfVisualWords,
    svm: TraditionalClassifier,
    random_forest: TraditionalClassifier,
    corruption: str,
    severity_name: str,
    severity_value: float | int | None,
) -> list[dict[str, object]]:
    """Evaluate both trained models under one degradation condition."""
    print("\n" + "=" * 60)
    print(
        f"Corruption: {corruption} | "
        f"Severity: {severity_name}"
    )
    print("=" * 60)

    start_time = time.perf_counter()

    test_descriptors = generate_corrupted_descriptors(
        image_paths=test_paths,
        corruption=corruption,
        severity_value=severity_value,
        random_state=RANDOM_STATE,
    )

    test_features = bovw.transform(
        test_descriptors
    )

    processing_time = (
        time.perf_counter() - start_time
    )

    condition_results: list[dict[str, object]] = []

    models = {
        "svm": svm,
        "random_forest": random_forest,
    }

    for model_name, model in models.items():
        results = Evaluator.evaluate(
            model,
            test_features,
            test_labels,
        )

        Evaluator.print_summary(
            title=(
                f"{model_name.upper()} — "
                f"{corruption} ({severity_name})"
            ),
            results=results,
        )

        condition_results.append(
            {
                "corruption": corruption,
                "severity": severity_name,
                "severity_value": severity_value,
                "model": model_name,
                "accuracy": results["accuracy"],
                "top5_accuracy": results["top5_accuracy"],
                "precision_macro": results["precision_macro"],
                "recall_macro": results["recall_macro"],
                "macro_f1": results["f1_macro"],
                "processing_time_seconds": processing_time,
            }
        )

    return condition_results


def save_results(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save robustness results as CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "corruption",
        "severity",
        "severity_value",
        "model",
        "accuracy",
        "top5_accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "processing_time_seconds",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    dataset_root = Path(
        "data/subset"
    )

    bovw_path = Path(
        "models/bovw_200_500_classes.joblib"
    )

    svm_path = Path(
        "models/svm_bovw_200_500_classes.joblib"
    )

    random_forest_path = Path(
        "models/random_forest_bovw_200_500_classes.joblib"
    )

    required_files = [
        bovw_path,
        svm_path,
        random_forest_path,
    ]

    for required_file in required_files:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Required model not found: {required_file}"
            )

    print("Loading dataset...")

    dataset = load_dataset(
        dataset_root
    )

    test_paths = dataset["test_paths"]

    test_labels = np.asarray(
        dataset["test_labels"],
        dtype=np.int32,
    )

    print(
        f"Classes used: "
        f"{len(dataset['class_to_index'])}"
    )

    print(
        f"Testing images used: "
        f"{len(test_paths)}"
    )

    print("\nLoading trained models...")

    bovw = BagOfVisualWords.load(
        bovw_path
    )

    svm = TraditionalClassifier.load(
        svm_path
    )

    random_forest = TraditionalClassifier.load(
        random_forest_path
    )

    conditions = [
        {
            "corruption": "clean",
            "severity": "none",
            "value": None,
        },
        {
            "corruption": "gaussian_noise",
            "severity": "low",
            "value": 5,
        },
        {
            "corruption": "gaussian_noise",
            "severity": "medium",
            "value": 15,
        },
        {
            "corruption": "gaussian_noise",
            "severity": "high",
            "value": 30,
        },
        {
            "corruption": "gaussian_blur",
            "severity": "low",
            "value": 3,
        },
        {
            "corruption": "gaussian_blur",
            "severity": "medium",
            "value": 7,
        },
        {
            "corruption": "gaussian_blur",
            "severity": "high",
            "value": 11,
        },
        {
            "corruption": "brightness",
            "severity": "low",
            "value": 0.75,
        },
        {
            "corruption": "brightness",
            "severity": "medium",
            "value": 0.50,
        },
        {
            "corruption": "brightness",
            "severity": "high",
            "value": 0.25,
        },
        {
            "corruption": "jpeg",
            "severity": "low",
            "value": 75,
        },
        {
            "corruption": "jpeg",
            "severity": "medium",
            "value": 50,
        },
        {
            "corruption": "jpeg",
            "severity": "high",
            "value": 25,
        },
    ]

    all_results: list[dict[str, object]] = []

    for condition in conditions:
        condition_results = evaluate_condition(
            test_paths=test_paths,
            test_labels=test_labels,
            bovw=bovw,
            svm=svm,
            random_forest=random_forest,
            corruption=condition["corruption"],
            severity_name=condition["severity"],
            severity_value=condition["value"],
        )

        all_results.extend(
            condition_results
        )

    output_path = Path(
        "results/robustness_results_500_classes.csv"
    )

    save_results(
        all_results,
        output_path,
    )

    print("\n" + "=" * 60)
    print("Robustness Experiment Summary")
    print("=" * 60)

    for result in all_results:
        top5 = result["top5_accuracy"]

        top5_text = (
            f"{top5:.4f}"
            if top5 is not None
            else "N/A"
        )

        print(
            f"{result['model']:14}"
            f" | {result['corruption']:16}"
            f" | {result['severity']:6}"
            f" | top1={result['accuracy']:.4f}"
            f" | top5={top5_text}"
            f" | f1={result['macro_f1']:.4f}"
        )

    print(
        f"\nResults saved to: {output_path}"
    )


if __name__ == "__main__":
    main()