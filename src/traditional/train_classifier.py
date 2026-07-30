from pathlib import Path

import numpy as np

from classifier import TraditionalClassifier
from evaluation import Evaluator


def main() -> None:
    feature_path = Path(
        "results/bovw_features_50_classes.npz"
    )

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}\n"
            "Run main.py first to generate BoVW features."
        )

    print("Loading BoVW features...")

    with np.load(feature_path) as data:
        train_features = data["train_features"]
        train_labels = data["train_labels"]

        val_features = data["val_features"]
        val_labels = data["val_labels"]

        test_features = data["test_features"]
        test_labels = data["test_labels"]

    print(f"Training features: {train_features.shape}")
    print(f"Validation features: {val_features.shape}")
    print(f"Testing features: {test_features.shape}")

    # ---------------------------------------------------------
    # Train and evaluate SVM
    # ---------------------------------------------------------

    print("\nTraining SVM...")

    svm = TraditionalClassifier(
        model_type="svm",
        random_state=42,
    )

    svm.fit(
        train_features,
        train_labels,
    )

    svm_val_results = Evaluator.evaluate(
        svm,
        val_features,
        val_labels,
    )

    Evaluator.print_summary(
        title="SVM Validation Results",
        results=svm_val_results,
    )

    svm_path = Path(
        "models/svm_bovw_200.joblib"
    )

    svm.save(svm_path)

    print(f"SVM saved to: {svm_path}")

    Evaluator.save_results(
        svm_val_results,
        "results/svm_validation_results.joblib",
    )

    # ---------------------------------------------------------
    # Train and evaluate Random Forest
    # ---------------------------------------------------------

    print("\nTraining Random Forest...")

    random_forest = TraditionalClassifier(
        model_type="random_forest",
        random_state=42,
    )

    random_forest.fit(
        train_features,
        train_labels,
    )

    rf_val_results = Evaluator.evaluate(
        random_forest,
        val_features,
        val_labels,
    )

    Evaluator.print_summary(
        title="Random Forest Validation Results",
        results=rf_val_results,
    )

    rf_path = Path(
        "models/random_forest_bovw_200.joblib"
    )

    random_forest.save(rf_path)

    print(f"Random Forest saved to: {rf_path}")

    Evaluator.save_results(
        rf_val_results,
        "results/random_forest_validation_results.joblib",
    )

    # ---------------------------------------------------------
    # Compare validation results
    # ---------------------------------------------------------

    print("\nValidation comparison:")
    print(
        f"SVM accuracy: "
        f"{svm_val_results['accuracy']:.4f}"
    )
    print(
        f"SVM macro F1: "
        f"{svm_val_results['f1_macro']:.4f}"
    )

    print(
        f"Random Forest accuracy: "
        f"{rf_val_results['accuracy']:.4f}"
    )
    print(
        f"Random Forest macro F1: "
        f"{rf_val_results['f1_macro']:.4f}"
    )

    # Select the model using validation macro F1.
    if svm_val_results["f1_macro"] >= rf_val_results["f1_macro"]:
        best_model = svm
        best_model_name = "SVM"
    else:
        best_model = random_forest
        best_model_name = "Random Forest"

    print(
        f"\nBest validation model: {best_model_name}"
    )

    # ---------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------

    test_results = Evaluator.evaluate(
        best_model,
        test_features,
        test_labels,
    )

    Evaluator.print_summary(
        title=f"{best_model_name} Final Test Results",
        results=test_results,
    )

    Evaluator.save_results(
        test_results,
        "results/best_model_test_results.joblib",
    )

    print(
        "\nEvaluation results saved in the results folder."
    )


if __name__ == "__main__":
    main()