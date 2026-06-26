"""Streamlit dashboard for the AJM Capital loan approval classifier."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "best_classifier.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "loan_preprocessor.pkl"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "explainability" / "feature_importance.png"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "classification" / "confusion_matrix.png"
ROC_CURVE_PATH = OUTPUTS_DIR / "classification" / "roc_curve.png"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from explain import explain_prediction
from predict_classifier import predict


st.set_page_config(
    page_title="AJM Capital AI",
    page_icon="AJM",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[Any, Any]:
    """Load persisted model and preprocessor artifacts."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {BEST_MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing preprocessor artifact: {PREPROCESSOR_PATH}",
        )

    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def get_feature_schema(preprocessor: Any) -> tuple[list[str], dict[str, list[Any]]]:
    """Extract raw numeric and categorical feature schema from the preprocessor."""
    numerical_columns: list[str] = []
    categorical_columns: list[str] = []
    categorical_values: dict[str, list[Any]] = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            numerical_columns = list(columns)
        if name == "cat":
            categorical_columns = list(columns)
            encoder = transformer.named_steps["encoder"]
            categorical_values = {
                column: list(categories)
                for column, categories in zip(categorical_columns, encoder.categories_)
            }

    return numerical_columns, categorical_values


def get_numeric_defaults(preprocessor: Any, numerical_columns: list[str]) -> dict[str, float]:
    """Use fitted median imputer statistics as sensible numeric defaults."""
    for name, transformer, _ in preprocessor.transformers_:
        if name == "num":
            imputer = transformer.named_steps["imputer"]
            return {
                column: float(value)
                for column, value in zip(numerical_columns, imputer.statistics_)
            }
    return {column: 0.0 for column in numerical_columns}


def prettify_feature_name(feature_name: str) -> str:
    """Make transformed feature names easier to read in the dashboard."""
    cleaned = feature_name.replace("num__", "").replace("cat__", "")
    return cleaned.replace("_", " ")


def inject_styles() -> None:
    """Add dashboard-specific styling."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            border-bottom: 1px solid #d9e2ec;
            padding-bottom: 1.2rem;
            margin-bottom: 1.5rem;
        }
        .eyebrow {
            color: #52616b;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
        }
        .hero h1 {
            color: #102a43;
            font-size: 2.5rem;
            line-height: 1.1;
            margin: 0.25rem 0 0;
        }
        .section-title {
            color: #102a43;
            font-size: 1.2rem;
            font-weight: 700;
            margin: 0.6rem 0 0.8rem;
        }
        .prediction-card {
            border-radius: 8px;
            border: 1px solid #d9e2ec;
            padding: 1.25rem;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(16, 42, 67, 0.06);
        }
        .approved {
            border-left: 8px solid #168a4a;
        }
        .rejected {
            border-left: 8px solid #c92a2a;
        }
        .status-approved {
            color: #168a4a;
            font-size: 1.8rem;
            font-weight: 800;
        }
        .status-rejected {
            color: #c92a2a;
            font-size: 1.8rem;
            font-weight: 800;
        }
        .metric-label {
            color: #52616b;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .metric-value {
            color: #102a43;
            font-size: 1.45rem;
            font-weight: 800;
        }
        .factor-row {
            border-bottom: 1px solid #edf2f7;
            padding: 0.55rem 0;
        }
        .factor-impact-positive {
            color: #168a4a;
            font-weight: 700;
        }
        .factor-impact-negative {
            color: #c92a2a;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render dashboard title area."""
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">AJM Capital AI</div>
            <h1>Loan Approval Predictor</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_customer_form(preprocessor: Any) -> dict[str, Any] | None:
    """Render dynamic customer input widgets from the trained schema."""
    numerical_columns, categorical_values = get_feature_schema(preprocessor)
    numeric_defaults = get_numeric_defaults(preprocessor, numerical_columns)

    st.markdown('<div class="section-title">Customer Information</div>', unsafe_allow_html=True)

    with st.form("customer_information"):
        left, right = st.columns(2, gap="large")
        customer_data: dict[str, Any] = {}

        for index, column in enumerate(numerical_columns):
            container = left if index % 2 == 0 else right
            default_value = numeric_defaults.get(column, 0.0)
            minimum = 0.0
            step = 1.0
            if column == "Credit_History":
                customer_data[column] = container.selectbox(
                    column.replace("_", " "),
                    options=[1.0, 0.0],
                    format_func=lambda value: "Yes" if value == 1.0 else "No",
                    index=0 if default_value >= 0.5 else 1,
                )
                continue

            customer_data[column] = container.number_input(
                column.replace("_", " "),
                min_value=minimum,
                value=default_value,
                step=step,
            )

        for index, (column, values) in enumerate(categorical_values.items()):
            container = left if index % 2 == 0 else right
            customer_data[column] = container.selectbox(
                column.replace("_", " "),
                options=values,
            )

        submitted = st.form_submit_button(
            "Predict Loan Approval",
            type="primary",
            use_container_width=True,
        )

    return customer_data if submitted else None


def render_prediction_card(result: dict[str, Any]) -> None:
    """Render prediction card with approval details."""
    approved = bool(result["approved"])
    status_text = "Approved" if approved else "Rejected"
    card_class = "approved" if approved else "rejected"
    status_class = "status-approved" if approved else "status-rejected"

    st.markdown('<div class="section-title">Prediction Card</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="prediction-card {card_class}">
            <div class="{status_class}">{status_text}</div>
            <div style="height: 1rem;"></div>
            <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem;">
                <div>
                    <div class="metric-label">Approval Probability</div>
                    <div class="metric-value">{result["approval_probability"]}%</div>
                </div>
                <div>
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value">{result["risk_level"]}</div>
                </div>
                <div>
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value">{result["confidence_score"]}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_factor_list(
    title: str,
    factors: list[dict[str, Any]],
    positive: bool,
) -> None:
    """Render explainability factor rows."""
    st.markdown(f"**{title}**")
    impact_class = "factor-impact-positive" if positive else "factor-impact-negative"

    if not factors:
        st.caption("No factors available.")
        return

    for factor in factors:
        feature = prettify_feature_name(str(factor["feature"]))
        impact = float(factor["impact"])
        st.markdown(
            f"""
            <div class="factor-row">
                <span>{feature}</span>
                <span class="{impact_class}" style="float: right;">{impact:.4f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_explainability(explanation: dict[str, list[dict[str, Any]]]) -> None:
    """Render positive and negative prediction factors."""
    st.markdown('<div class="section-title">Explainability</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        render_factor_list(
            "Top Positive Factors",
            explanation["top_positive_factors"],
            positive=True,
        )
    with right:
        render_factor_list(
            "Top Negative Factors",
            explanation["top_negative_factors"],
            positive=False,
        )


def render_visualizations() -> None:
    """Render saved model visualization artifacts."""
    st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)

    first, second, third = st.columns(3, gap="large")
    visualizations = [
        (first, "Feature Importance", FEATURE_IMPORTANCE_PATH),
        (second, "Confusion Matrix", CONFUSION_MATRIX_PATH),
        (third, "ROC Curve", ROC_CURVE_PATH),
    ]

    for container, title, image_path in visualizations:
        with container:
            st.markdown(f"**{title}**")
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.info(f"{title} image is not available yet.")


def main() -> None:
    """Run the AJM Capital AI Streamlit dashboard."""
    inject_styles()
    render_header()

    try:
        model, preprocessor = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    customer_data = render_customer_form(preprocessor)

    if customer_data is None:
        st.info("Enter customer information and run a prediction.")
        render_visualizations()
        return

    prediction_result = predict(
        customer_data,
        model=model,
        preprocessor=preprocessor,
    )
    explanation = explain_prediction(
        customer_data,
        model=model,
        preprocessor=preprocessor,
    )

    render_prediction_card(prediction_result)
    render_explainability(explanation)
    render_visualizations()


if __name__ == "__main__":
    main()
