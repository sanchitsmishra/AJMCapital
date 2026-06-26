"""Preprocess loan approval data for binary classification."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "loan_prediction.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "loan_preprocessor.pkl"
TARGET_COLUMN = "Loan_Status"
TEST_SIZE = 0.2
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw loan approval dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    logger.info("Loading dataset from %s", file_path)
    return pd.read_csv(file_path)


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and normalize the target column."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")

    cleaned = data.drop_duplicates().copy()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"Y": 1, "N": 0})
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    logger.info("Cleaned dataset shape: %s", cleaned.shape)
    return cleaned


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the cleaned dataset into feature and target objects."""
    features = data.drop(columns=[TARGET_COLUMN])
    identifier_columns = [
        column
        for column in features.columns
        if column.lower() == "id" or column.lower().endswith("_id")
    ]
    if identifier_columns:
        logger.info("Dropping identifier columns: %s", identifier_columns)
        features = features.drop(columns=identifier_columns)

    target = data[TARGET_COLUMN]
    return features, target


def detect_feature_types(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Detect numeric and categorical feature columns automatically."""
    numerical_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(
        include=["object", "str", "string", "category", "bool"],
    ).columns.tolist()

    logger.info("Detected numerical columns: %s", numerical_columns)
    logger.info("Detected categorical columns: %s", categorical_columns)
    return numerical_columns, categorical_columns


def build_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build a preprocessing transformer for numeric and categorical columns."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ],
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def save_processed_dataset(
    X_train_processed: pd.DataFrame,
    X_test_processed: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Save processed train and test datasets to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train_processed.to_csv(output_dir / "X_train.csv", index=False)
    X_test_processed.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_frame(name=TARGET_COLUMN).to_csv(
        output_dir / "y_train.csv",
        index=False,
    )
    y_test.to_frame(name=TARGET_COLUMN).to_csv(
        output_dir / "y_test.csv",
        index=False,
    )

    logger.info("Processed datasets saved to %s", output_dir)


def preprocess_data(
    input_path: Path = RAW_DATA_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    preprocessor_path: Path = PREPROCESSOR_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Run the complete preprocessing workflow and save artifacts."""
    data = clean_dataset(load_dataset(input_path))
    features, target = split_features_target(data)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    numerical_columns, categorical_columns = detect_feature_types(X_train)
    preprocessor = build_preprocessor(numerical_columns, categorical_columns)

    logger.info("Fitting preprocessor on X_train only.")
    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_processed = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=X_test.index,
    )

    save_processed_dataset(
        X_train_processed,
        X_test_processed,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        processed_dir,
    )

    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Preprocessing pipeline saved to %s", preprocessor_path)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def main() -> None:
    """Execute preprocessing from the command line."""
    preprocess_data()


if __name__ == "__main__":
    main()
