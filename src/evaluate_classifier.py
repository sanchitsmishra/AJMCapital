"""Evaluate the best loan approval classifier and save reports/plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)

try:
    from train_classifier import BEST_MODEL_PATH, PROCESSED_DATA_DIR, TARGET_COLUMN
except ImportError:
    from .train_classifier import BEST_MODEL_PATH, PROCESSED_DATA_DIR, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_best_model(model_path: Path = BEST_MODEL_PATH) -> Any:
    """Load the persisted best classifier."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}. Run training first.",
        )
    return joblib.load(model_path)


def load_test_data(
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load preprocessed test data."""
    X_test_path = processed_dir / "X_test.csv"
    y_test_path = processed_dir / "y_test.csv"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError("Processed test data not found.")

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)[TARGET_COLUMN]
    return X_test, y_test


def save_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Save a confusion matrix plot."""
    display = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Rejected", "Approved"],
        cmap="Blues",
    )
    display.ax_.set_title("Loan Approval Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def save_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Save classification report as text and CSV."""
    predictions = model.predict(X_test)
    report_dict = classification_report(
        y_test,
        predictions,
        target_names=["Rejected", "Approved"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predictions,
        target_names=["Rejected", "Approved"],
        zero_division=0,
    )

    (output_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        output_dir / "classification_report.csv",
    )


def save_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Save the ROC curve plot."""
    display = RocCurveDisplay.from_estimator(model, X_test, y_test)
    display.ax_.set_title("Loan Approval ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close()


def save_feature_importance(
    model: Any,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 20,
) -> None:
    """Save feature importance for tree-based models when available."""
    if not hasattr(model, "feature_importances_"):
        logger.info("Model does not expose tree feature importances; skipping.")
        return

    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        },
    ).sort_values(by="importance", ascending=False)
    importances.to_csv(output_dir / "feature_importance.csv", index=False)

    top_importances = importances.head(top_n).sort_values(
        by="importance",
        ascending=True,
    )
    plt.figure(figsize=(10, 7))
    plt.barh(top_importances["feature"], top_importances["importance"])
    plt.title("Top Loan Approval Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close()


def evaluate_classifier(output_dir: Path = OUTPUT_DIR) -> None:
    """Run full evaluation and save all requested artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_best_model()
    X_test, y_test = load_test_data()

    save_confusion_matrix(model, X_test, y_test, output_dir)
    save_classification_report(model, X_test, y_test, output_dir)
    save_roc_curve(model, X_test, y_test, output_dir)
    save_feature_importance(model, X_test.columns.tolist(), output_dir)

    logger.info("Evaluation artifacts saved to %s", output_dir)


def main() -> None:
    """Execute evaluation from the command line."""
    evaluate_classifier()


if __name__ == "__main__":
    main()
