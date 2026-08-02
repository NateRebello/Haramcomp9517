from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from bovw import BagOfVisualWords
from classifier import TraditionalClassifier
from evaluation import Evaluator
from main import sample_vocabulary_descriptors


def run_experiment(
    train_descriptors,
    train_labels,
    val_descriptors,
    val_labels,
    vocabulary_size: int,
) -> list[dict]:
    """Run one BoVW experiment."""

    print(f"\nVocabulary size: {vocabulary_size}")

    vocabulary_descriptors = sample_vocabulary_descriptors(
        train_descriptors,
        max_descriptors=100_000,
    )

    bovw = BagOfVisualWords(
        n_clusters=vocabulary_size,
        random_state=42,
    )

    start = time.perf_counter()

    bovw.fit(vocabulary_descriptors)

    train_features = bovw.transform(train_descriptors)
    val_features = bovw.transform(val_descriptors)

    feature_time = time.perf_counter() - start

    experiment_results = []

    for model_name in ["svm", "random_forest"]:

        classifier = TraditionalClassifier(
            model_type=model_name,
            random_state=42,
        )

        start = time.perf_counter()

        classifier.fit(
            train_features,
            train_labels,
        )

        training_time = time.perf_counter() - start

        results = Evaluator.evaluate(
            classifier,
            val_features,
            val_labels,
        )

        experiment_results.append(
            {
                "vocabulary_size": vocabulary_size,
                "classifier": model_name,
                "accuracy": results["accuracy"],
                "top5_accuracy": results["top5_accuracy"],
                "macro_f1": results["f1_macro"],
                "feature_time": feature_time,
                "training_time": training_time,
            }
        )

        Evaluator.print_summary(
            f"{model_name.upper()} ({vocabulary_size})",
            results,
        )

    return experiment_results


def main():

    descriptor_file = Path(
        "results/sift_descriptors_500_classes.joblib"
    )

    descriptor_data = joblib.load(
        descriptor_file
    )

    train_descriptors = descriptor_data["train_descriptors"]
    train_labels = descriptor_data["train_labels"]

    val_descriptors = descriptor_data["val_descriptors"]
    val_labels = descriptor_data["val_labels"]

    all_results = []

    for vocabulary_size in [100, 200, 500]:

        all_results.extend(
            run_experiment(
                train_descriptors,
                train_labels,
                val_descriptors,
                val_labels,
                vocabulary_size,
            )
        )

    print("\n==============================")
    print("Experiment Summary")
    print("==============================")

    for result in all_results:

        print(
            f"{result['classifier']:15}"
            f" | words={result['vocabulary_size']:3}"
            f" | acc={result['accuracy']:.4f}"
            f" | top5={result['top5_accuracy']:.4f}"
            f" | f1={result['macro_f1']:.4f}"
            f" | feature={result['feature_time']:.2f}s"
            f" | train={result['training_time']:.2f}s"
        )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = results_dir / "experiment_results_500_classes.joblib"

    joblib.dump(
        all_results,
        joblib_path,
    )

    csv_path = results_dir / "experiment_results_500_classes.csv"

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        csv_path,
        index=False,
    )

    print(f"\nJoblib results saved to: {joblib_path}")
    print(f"CSV results saved to: {csv_path}")


if __name__ == "__main__":
    main()