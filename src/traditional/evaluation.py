from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluator:
    """Evaluate classification models."""

    @staticmethod
    def evaluate(
        model,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, object]:
        """
        Evaluate a classifier on a labelled dataset.

        Returns:
            Dictionary containing evaluation metrics.
        """
        predictions = model.predict(features)

        results = {
            "accuracy": accuracy_score(labels, predictions),
            "precision_macro": precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                labels,
                predictions,
            ),
            "classification_report": classification_report(
                labels,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }

        return results

    @staticmethod
    def print_summary(
        title: str,
        results: dict[str, object],
    ) -> None:
        """Print a concise evaluation summary."""
        print(f"\n{'=' * 50}")
        print(title)
        print("=" * 50)

        print(
            f"Accuracy : {results['accuracy']:.4f}"
        )
        print(
            f"Precision: {results['precision_macro']:.4f}"
        )
        print(
            f"Recall   : {results['recall_macro']:.4f}"
        )
        print(
            f"Macro F1 : {results['f1_macro']:.4f}"
        )

    @staticmethod
    def save_results(
        results: dict[str, object],
        file_path: str | Path,
    ) -> None:
        """Save evaluation results."""
        file_path = Path(file_path)
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(results, file_path)