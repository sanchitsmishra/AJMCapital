"""Prediction utilities for the loan approval classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_classifier.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "loan_preprocessor.pkl"
APPROVAL_THRESHOLD = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path = BEST_MODEL_PATH) -> Any:
    """Load the trained best classifier."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first.",
        )
    return joblib.load(model_path)


def load_preprocessor(preprocessor_path: Path = PREPROCESSOR_PATH) -> Any:
    """Load the fitted preprocessing pipeline."""
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {preprocessor_path}. "
            "Run preprocessing first.",
        )
    return joblib.load(preprocessor_path)


def _to_dataframe(customer_data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Convert one or more customer records to a DataFrame."""
    if isinstance(customer_data, pd.DataFrame):
        return customer_data.copy()
    return pd.DataFrame([customer_data])


def _risk_level(approval_probability: float) -> str:
    """Map approval probability to a human-readable risk level."""
    if approval_probability >= 80.0:
        return "Low"
    if approval_probability >= 50.0:
        return "Medium"
    return "High"


def _transform_customer_data(
    customer_data: dict[str, Any] | pd.DataFrame,
    preprocessor: Any,
) -> pd.DataFrame:
    """Transform raw customer data and preserve preprocessed feature names."""
    customer_frame = _to_dataframe(customer_data)
    processed_data = preprocessor.transform(customer_frame)
    return pd.DataFrame(
        processed_data,
        columns=preprocessor.get_feature_names_out(),
    )


def predict_probability(
    customer_data: dict[str, Any] | pd.DataFrame,
    model: Any | None = None,
    preprocessor: Any | None = None,
) -> float | list[float]:
    """Predict loan approval probability as a percentage."""
    classifier = model or load_model()
    transformer = preprocessor or load_preprocessor()

    processed_data = _transform_customer_data(customer_data, transformer)
    probabilities = classifier.predict_proba(processed_data)[:, 1] * 100

    if len(probabilities) == 1:
        return round(float(probabilities[0]), 2)
    return [round(float(probability), 2) for probability in probabilities]


def predict(
    customer_data: dict[str, Any] | pd.DataFrame,
    model: Any | None = None,
    preprocessor: Any | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict approval status, probability, risk, and confidence."""
    classifier = model or load_model()
    transformer = preprocessor or load_preprocessor()

    processed_data = _transform_customer_data(customer_data, transformer)
    approval_probabilities = classifier.predict_proba(processed_data)[:, 1]

    results = []
    for approval_probability in approval_probabilities:
        approved = bool(approval_probability >= APPROVAL_THRESHOLD)
        approval_percentage = round(float(approval_probability * 100), 2)
        confidence_score = round(
            float(max(approval_probability, 1 - approval_probability) * 100),
            2,
        )
        results.append(
            {
                "approved": approved,
                "approval_probability": approval_percentage,
                "risk_level": _risk_level(approval_percentage),
                "confidence_score": confidence_score,
            },
        )

    if isinstance(customer_data, pd.DataFrame):
        return results
    return results[0]


if __name__ == "__main__":
    logger.info("Import this module and call predict(customer_data).")
