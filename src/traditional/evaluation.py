"""Evaluate classification models.

This is Brandon's evaluation.py (from his feature branch) with two additions:

  - top5_accuracy in Evaluator.evaluate(), using model.decision_scores() if
    the model exposes it (classifier.py's TraditionalClassifier does after
    today's patch). The spec requires top-1 AND top-5 accuracy; this was
    missing before.
  - Evaluator.most_confused_pairs(), for the "hardest, most-confused species
    pairs" analysis the spec explicitly asks for.

Existing keys (accuracy, precision_macro, recall_macro, f1_macro,
confusion_matrix, classification_report) and existing methods are unchanged,
so run_experiments.py and train_classifier.py keep working without edits --
they'll just start getting top5_accuracy in their results dict for free.
"""
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
    top_k_accuracy_score,
)


class Evaluator:
    """Evaluate classification models."""

    @staticmethod
    def evaluate(
        model,
        features: np.ndarray,
        labels: np.ndarray,
        top_k: int = 5,
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

        results[f"top{top_k}_accuracy"] = Evaluator._safe_top_k_accuracy(
            model, features, labels, top_k
        )

        return results

    @staticmethod
    def _safe_top_k_accuracy(model, features, labels, top_k: int):
        """Compute top-k accuracy if the model can produce per-class scores,
        otherwise return None instead of crashing the whole evaluation."""
        if not hasattr(model, "decision_scores"):
            return None

        try:
            scores = model.decision_scores(features)
            return top_k_accuracy_score(
                labels, scores, k=top_k, labels=model.classes_
            )
        except Exception as error:  # noqa: BLE001 - degrade gracefully, don't crash a run
            print(f"Warning: could not compute top-{top_k} accuracy: {error}")
            return None

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

        top_k_key = next((k for k in results if k.startswith("top") and k.endswith("_accuracy")), None)
        if top_k_key and results[top_k_key] is not None:
            print(f"{top_k_key.replace('_', ' ').title()}: {results[top_k_key]:.4f}")

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
    def most_confused_pairs(
        results: dict[str, object],
        class_labels: list[int],
        top_n: int = 15,
    ) -> list[tuple[int, int, int]]:
        """
        Return the top_n (true_class, predicted_class, count) off-diagonal
        confusions from an evaluate() results dict -- use this for the
        "hardest, most-confused species pairs" analysis in the report.

        class_labels must be in the same order as the confusion matrix rows/
        columns, i.e. model.classes_ from whichever model produced `results`.
        """
        cm = results["confusion_matrix"]
        pairs = []
        n = cm.shape[0]

        for i in range(n):
            for j in range(n):
                if i != j and cm[i, j] > 0:
                    pairs.append((class_labels[i], class_labels[j], int(cm[i, j])))

        pairs.sort(key=lambda item: item[2], reverse=True)
        return pairs[:top_n]

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
