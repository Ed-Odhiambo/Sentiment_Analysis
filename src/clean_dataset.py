"""Apply text preprocessing to the prepared customer review dataset."""

from pathlib import Path

import pandas as pd

from src.preprocessing import clean_review, verify_nltk_resources


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_reviews.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_reviews_cleaned.csv"
)


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the standardized customer review dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    required_columns = {"review", "sentiment"}
    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add a cleaned review column and remove unusable records."""
    cleaned = dataframe.copy()

    cleaned["clean_review"] = (
        cleaned["review"]
        .fillna("")
        .apply(clean_review)
    )

    original_count = len(cleaned)

    cleaned = cleaned[
        cleaned["clean_review"].str.strip() != ""
    ].copy()

    cleaned = cleaned.drop_duplicates(
        subset=["clean_review", "sentiment"]
    )

    cleaned = cleaned.reset_index(drop=True)

    removed_count = original_count - len(cleaned)

    print(f"Rows before cleaning: {original_count:,}")
    print(f"Rows after cleaning:  {len(cleaned):,}")
    print(f"Rows removed:         {removed_count:,}")

    return cleaned


def display_examples(dataframe: pd.DataFrame) -> None:
    """Display examples of original and cleaned reviews."""
    print("\nCLEANING EXAMPLES")
    print("-" * 80)

    example_columns = [
        "review",
        "clean_review",
        "sentiment",
    ]

    print(
        dataframe[example_columns]
        .head(10)
        .to_string(index=False)
    )


def main() -> None:
    """Run the complete dataset-cleaning process."""
    verify_nltk_resources()

    dataframe = load_dataset(INPUT_PATH)
    cleaned_dataframe = clean_dataset(dataframe)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    display_examples(cleaned_dataframe)

    print(f"\nCleaned dataset saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()