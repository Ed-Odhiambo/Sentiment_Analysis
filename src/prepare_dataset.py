"""Prepare the Kaggle airline feedback dataset for sentiment modelling."""

from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Tweets.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customer_reviews.csv"
)

# Expected columns and labels
TEXT_COLUMN = "text"
LABEL_COLUMN = "airline_sentiment"
VALID_SENTIMENTS = {"positive", "neutral", "negative"}


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load the original Kaggle dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            "Download Tweets.csv and place it inside data/raw/."
        )

    return pd.read_csv(file_path)


def validate_columns(dataframe: pd.DataFrame) -> None:
    """Ensure that the required dataset columns are available."""
    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The dataset is missing the following required columns: "
            f"{sorted(missing_columns)}"
        )


def prepare_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Select, rename, validate and clean the required columns."""
    prepared = dataframe[[TEXT_COLUMN, LABEL_COLUMN]].copy()

    prepared = prepared.rename(
        columns={
            TEXT_COLUMN: "review",
            LABEL_COLUMN: "sentiment",
        }
    )

    # Use pandas' string type so missing values remain identifiable.
    prepared["review"] = prepared["review"].astype("string").str.strip()
    prepared["sentiment"] = (
        prepared["sentiment"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Remove records without usable text or sentiment.
    prepared = prepared.dropna(subset=["review", "sentiment"])
    prepared = prepared[prepared["review"] != ""]

    invalid_sentiments = (
        set(prepared["sentiment"].unique()) - VALID_SENTIMENTS
    )

    if invalid_sentiments:
        raise ValueError(
            "Unexpected sentiment labels found: "
            f"{sorted(invalid_sentiments)}"
        )

    # Remove duplicate review text to reduce data leakage.
    prepared = prepared.drop_duplicates(subset=["review"])

    return prepared.reset_index(drop=True)


def display_summary(
    original_data: pd.DataFrame,
    prepared_data: pd.DataFrame,
) -> None:
    """Display basic information about the prepared dataset."""
    removed_rows = len(original_data) - len(prepared_data)

    print("\nDATASET PREPARATION SUMMARY")
    print("-" * 40)
    print(f"Original rows: {len(original_data):,}")
    print(f"Prepared rows: {len(prepared_data):,}")
    print(f"Removed rows: {removed_rows:,}")

    print("\nSentiment distribution:")
    print(prepared_data["sentiment"].value_counts())

    print("\nSentiment percentages:")
    percentages = (
        prepared_data["sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )
    print(percentages.astype(str) + "%")

    print("\nFirst five prepared records:")
    print(prepared_data.head().to_string(index=False))


def main() -> None:
    """Run the complete data-preparation process."""
    raw_data = load_raw_data(RAW_DATA_PATH)
    validate_columns(raw_data)

    prepared_data = prepare_data(raw_data)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared_data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
        encoding="utf-8",
    )

    display_summary(raw_data, prepared_data)

    print(f"\nPrepared dataset saved to:\n{PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()