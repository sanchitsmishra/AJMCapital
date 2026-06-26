"""Train loan approval classifiers and persist the best model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_classifier.pkl"
METRICS_PATH = MODELS_DIR / "classifier_metrics.csv"
TARGET_COLUMN = "Loan_Status"
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_processed_data(
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed train and test datasets from disk."""
    required_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
    ]
    missing_files = [
        file_name
        for file_name in required_files
        if not (processed_dir / file_name).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing processed files: "
            f"{missing_files}. Run src/loan_preprocess.py first.",
        )

    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")[TARGET_COLUMN]
    y_test = pd.read_csv(processed_dir / "y_test.csv")[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def get_candidate_models() -> dict[str, Any]:
    """Create the classifier candidates available in the environment."""
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        logger.info("XGBoost is installed and will be trained.")
    except ImportError:
        logger.info("XGBoost is not installed; skipping XGBoost classifier.")

    return models


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calculate classification metrics for a trained model."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, str, Any]:
    """Train all candidate classifiers and select the best by ROC AUC."""
    rows: list[dict[str, float | str]] = []
    trained_models: dict[str, Any] = {}

    for model_name, model in get_candidate_models().items():
        logger.info("Training %s", model_name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": model_name, **metrics})
        trained_models[model_name] = model
        logger.info("%s metrics: %s", model_name, metrics)

    metrics_df = pd.DataFrame(rows).sort_values(
        by="roc_auc",
        ascending=False,
    )
    best_model_name = str(metrics_df.iloc[0]["model"])
    return metrics_df, best_model_name, trained_models[best_model_name]


def save_training_artifacts(
    best_model: Any,
    metrics_df: pd.DataFrame,
    best_model_name: str,
    output_dir: Path = MODELS_DIR,
) -> None:
    """Save the best model and the model comparison metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)

    logger.info("Best model: %s", best_model_name)
    logger.info("Best classifier saved to %s", BEST_MODEL_PATH)
    logger.info("Classifier metrics saved to %s", METRICS_PATH)


def run_training() -> tuple[pd.DataFrame, str]:
    """Run the full training workflow."""
    X_train, X_test, y_train, y_test = load_processed_data()
    metrics_df, best_model_name, best_model = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    save_training_artifacts(best_model, metrics_df, best_model_name)
    return metrics_df, best_model_name


def main() -> None:
    """Execute model training from the command line."""
    metrics_df, best_model_name = run_training()
    logger.info("Training complete. Best model: %s", best_model_name)
    logger.info("Metrics:\n%s", metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
