"""Explainability utilities for loan approval classifier predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "explainability"
BEST_MODEL_PATH = MODELS_DIR / "best_classifier.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "loan_preprocessor.pkl"
TOP_N_FACTORS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path = BEST_MODEL_PATH) -> Any:
    """Load the trained classifier."""
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
    """Convert a single customer record to a one-row DataFrame."""
    if isinstance(customer_data, pd.DataFrame):
        if len(customer_data) != 1:
            raise ValueError("explain_prediction expects exactly one record.")
        return customer_data.copy()
    return pd.DataFrame([customer_data])


def _transform_customer_data(
    customer_data: dict[str, Any] | pd.DataFrame,
    preprocessor: Any,
) -> pd.DataFrame:
    """Apply preprocessing and preserve transformed feature names."""
    customer_frame = _to_dataframe(customer_data)
    processed_data = preprocessor.transform(customer_frame)
    return pd.DataFrame(
        processed_data,
        columns=preprocessor.get_feature_names_out(),
    )


def _is_tree_based_model(model: Any) -> bool:
    """Return True when the model is a tree-based estimator."""
    module_name = model.__class__.__module__.lower()
    class_name = model.__class__.__name__.lower()
    tree_markers = ("tree", "forest", "gradientboosting", "xgb", "lgbm", "catboost")
    return any(marker in module_name or marker in class_name for marker in tree_markers)


def _get_shap_contributions(
    model: Any,
    processed_customer: pd.DataFrame,
) -> pd.Series | None:
    """Return SHAP contribution values for tree models when SHAP is available."""
    if not _is_tree_based_model(model):
        return None

    try:
        import shap
    except ImportError:
        logger.info("SHAP is not installed; using fallback explanations.")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_customer)
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return None

    if isinstance(shap_values, list):
        values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        values = shap_values

    if hasattr(values, "values"):
        values = values.values

    if getattr(values, "ndim", 1) == 3:
        values = values[:, :, 1]

    return pd.Series(values[0], index=processed_customer.columns)


def _get_fallback_contributions(
    model: Any,
    processed_customer: pd.DataFrame,
) -> pd.Series:
    """Estimate feature contributions using importances or coefficients."""
    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        contributions = coefficients * processed_customer.iloc[0].to_numpy()
        return pd.Series(contributions, index=processed_customer.columns)

    if hasattr(model, "feature_importances_"):
        probabilities = model.predict_proba(processed_customer)[0]
        direction = 1 if probabilities[1] >= probabilities[0] else -1
        contributions = model.feature_importances_ * direction
        return pd.Series(contributions, index=processed_customer.columns)

    raise ValueError(
        "Model does not support SHAP, feature_importances_, or coef_ explanations.",
    )


def _format_factor(feature: str, contribution: float) -> dict[str, float | str]:
    """Format a feature contribution for API output."""
    return {
        "feature": feature,
        "impact": round(float(contribution), 6),
    }


def _split_factors(
    contributions: pd.Series,
    top_n: int = TOP_N_FACTORS,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    """Split contributions into top positive and negative factors."""
    positive = contributions[contributions > 0].sort_values(ascending=False)
    negative = contributions[contributions < 0].sort_values(ascending=True)

    top_positive = [
        _format_factor(feature, value)
        for feature, value in positive.head(top_n).items()
    ]
    top_negative = [
        _format_factor(feature, value)
        for feature, value in negative.head(top_n).items()
    ]
    return top_positive, top_negative


def _save_feature_importance_plot(
    contributions: pd.Series,
    output_path: Path = OUTPUT_DIR / "feature_importance.png",
    top_n: int = TOP_N_FACTORS,
) -> None:
    """Save a bar chart of the strongest positive and negative factors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strongest = contributions.reindex(
        contributions.abs().sort_values(ascending=False).head(top_n).index,
    ).sort_values()

    colors = ["#C44E52" if value < 0 else "#4C72B0" for value in strongest]
    plt.figure(figsize=(10, 7))
    plt.barh(strongest.index, strongest.values, color=colors)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title("Loan Approval Prediction Factors")
    plt.xlabel("Contribution to approval prediction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Explainability plot saved to %s", output_path)


def explain_prediction(
    customer_data: dict[str, Any] | pd.DataFrame,
    model: Any | None = None,
    preprocessor: Any | None = None,
) -> dict[str, list[dict[str, float | str]]]:
    """Explain one loan approval prediction with positive and negative factors."""
    classifier = model or load_model()
    transformer = preprocessor or load_preprocessor()
    processed_customer = _transform_customer_data(customer_data, transformer)

    contributions = _get_shap_contributions(classifier, processed_customer)
    if contributions is None:
        contributions = _get_fallback_contributions(
            classifier,
            processed_customer,
        )

    top_positive, top_negative = _split_factors(contributions)
    _save_feature_importance_plot(contributions)

    return {
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
    }


if __name__ == "__main__":
    logger.info("Import this module and call explain_prediction(customer_data).")
