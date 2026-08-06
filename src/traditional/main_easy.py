"""
Entry point for Team 1's EASY traditional baseline:
color histogram + LBP texture features -> SVM / Random Forest.

Trains both classifiers (same pattern as Brandon's train_classifier.py for
the BoVW pipeline) so the easy model also gets a cheap classifier comparison
for the "comprehensive" marking tier. Uses classifier.py's TraditionalClassifier
and evaluation.py's Evaluator so results are directly comparable to the tough
(BoVW-SIFT) pipeline.

Usage:
    python main_easy.py
"""
import time
from pathlib import Path

from classifier import TraditionalClassifier
from color_features import ColorLBPExtractor
from dataset import load_dataset
from evaluation import Evaluator


def main() -> None:
    dataset_root = Path("data/subset")
    results_dir = Path("results/easy_traditional")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(dataset_root)

    train_paths = dataset["train_paths"]
    train_labels = dataset["train_labels"]
    test_paths = dataset["test_paths"]
    test_labels = dataset["test_labels"]

    print(f"Training images: {len(train_paths)}")
    print(f"Testing images: {len(test_paths)}")
    print(f"Classes: {len(dataset['class_to_index'])}")

    print("\nExtracting color + LBP features...")
    extractor = ColorLBPExtractor()
    train_features = extractor.extract_from_dataset(train_paths)
    test_features = extractor.extract_from_dataset(test_paths)
    print(f"Feature dimensionality: {train_features.shape[1]}")

    for model_type in ("svm", "random_forest"):
        print(f"\nTraining {model_type}...")
        classifier = TraditionalClassifier(model_type=model_type, random_state=42)

        fit_start = time.perf_counter()
        classifier.fit(train_features, train_labels)
        train_time_s = time.perf_counter() - fit_start
        print(f"Trained in {train_time_s:.2f}s")

        eval_start = time.perf_counter()
        results = Evaluator.evaluate(classifier, test_features, test_labels)
        test_time_s = time.perf_counter() - eval_start
        print(f"Evaluated (predict + metrics) in {test_time_s:.2f}s")

        results["train_time_s"] = train_time_s
        results["test_time_s"] = test_time_s

        Evaluator.print_summary(
            f"Easy traditional -- {model_type} (color histogram + LBP)", results
        )

        model_path = results_dir / f"{model_type}_model.joblib"
        classifier.save(model_path)
        Evaluator.save_results(results, results_dir / f"{model_type}_results.joblib")
        print(f"Saved model to {model_path}")

        top_pairs = Evaluator.most_confused_pairs(
            results, class_labels=list(classifier.classes_), top_n=10
        )
        print(f"Most confused class pairs (index space, cross-reference class_to_index): {top_pairs}")


if __name__ == "__main__":
    main()
