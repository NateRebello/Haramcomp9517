"""Train and evaluate traditional machine-learning classifiers.

This is Brandon's classifier.py (from his feature branch) with one addition:
a `decision_scores()` method + `classes_` property so top-5 accuracy (required
by the spec) can be computed. SVC is deliberately left with probability=False
-- decision_function() gives a valid per-class score for ranking without the
expensive Platt-scaling calibration that probability=True triggers, which
matters at 500 classes. Everything else is unchanged, so run_experiments.py,
train_classifier.py, and main.py all keep working without modification.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class TraditionalClassifier:
    """Train and evaluate traditional machine-learning classifiers."""

    def __init__(
        self,
        model_type: str = "svm",
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = self._create_model()

    def _create_model(self):
        """Create the selected classifier."""
        if self.model_type == "svm":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        SVC(
                            kernel="rbf",
                            C=10.0,
                            gamma="scale",
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )

        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )

        raise ValueError(
            "model_type must be either 'svm' or 'random_forest'."
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Train the classifier."""
        self._validate_data(features, labels)
        self.model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels for a feature matrix."""
        if features.ndim != 2:
            raise ValueError(
                "Features must have shape "
                "(n_samples, n_features)."
            )

        return self.model.predict(features)

    def decision_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Per-class scores used for top-k accuracy, shape (n_samples, n_classes)
        with columns matching `self.classes_`.

        Prefers predict_proba (Random Forest has this by default). Falls back
        to decision_function for SVM, since probability=False here -- the
        ranking from decision_function is exactly what top_k_accuracy_score
        needs, without paying for probability calibration.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)

        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(features)

        raise AttributeError(
            f"{self.model_type} classifier supports neither "
            "predict_proba nor decision_function."
        )

    @property
    def classes_(self) -> np.ndarray:
        """Class ids in the same order as decision_scores()'s columns."""
        return self.model.classes_

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        print_report: bool = True,
    ) -> dict[str, object]:
        """Evaluate the classifier."""
        self._validate_data(features, labels)

        predictions = self.predict(features)
        accuracy = accuracy_score(labels, predictions)

        report = classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        if print_report:
            print(f"Accuracy: {accuracy:.4f}")
            print(
                classification_report(
                    labels,
                    predictions,
                    zero_division=0,
                )
            )

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "classification_report": report,
        }

    def save(self, file_path: str | Path) -> None:
        """Save the trained classifier."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, file_path)

    @classmethod
    def load(
        cls,
        file_path: str | Path,
    ) -> "TraditionalClassifier":
        """Load a saved classifier."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Classifier file does not exist: {file_path}"
            )

        classifier = joblib.load(file_path)

        if not isinstance(classifier, cls):
            raise TypeError(
                f"The file does not contain a {cls.__name__} object."
            )

        return classifier

    @staticmethod
    def _validate_data(
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Validate feature and label arrays."""
        if features.ndim != 2:
            raise ValueError(
                "Features must have shape "
                "(n_samples, n_features)."
            )

        if labels.ndim != 1:
            raise ValueError(
                "Labels must have shape (n_samples,)."
            )

        if len(features) != len(labels):
            raise ValueError(
                "The number of feature rows must match "
                "the number of labels."
            )

        if len(features) == 0:
            raise ValueError("The dataset cannot be empty.")
