"""Exploratory analysis for the cleaned sentiment dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_reviews_cleaned.csv"
)

FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures"

SENTIMENT_ORDER = [
    "negative",
    "neutral",
    "positive",
]


def load_dataset() -> pd.DataFrame:
    """Load the cleaned dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def add_review_length_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate character and word counts for each review."""
    analysed = dataframe.copy()

    analysed["character_count"] = (
        analysed["review"]
        .astype(str)
        .str.len()
    )

    analysed["word_count"] = (
        analysed["review"]
        .astype(str)
        .str.split()
        .str.len()
    )

    analysed["clean_word_count"] = (
        analysed["clean_review"]
        .astype(str)
        .str.split()
        .str.len()
    )

    return analysed


def print_summary(dataframe: pd.DataFrame) -> None:
    """Print dataset and sentiment statistics."""
    print("\nEXPLORATORY DATA ANALYSIS")
    print("-" * 50)

    print(f"Number of reviews: {len(dataframe):,}")
    print(f"Missing reviews: {dataframe['review'].isna().sum():,}")
    print(
        "Missing cleaned reviews: "
        f"{dataframe['clean_review'].isna().sum():,}"
    )

    print("\nSentiment counts:")
    print(
        dataframe["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER)
    )

    print("\nSentiment percentages:")
    percentages = (
        dataframe["sentiment"]
        .value_counts(normalize=True)
        .reindex(SENTIMENT_ORDER)
        .mul(100)
        .round(2)
    )
    print(percentages.astype(str) + "%")

    print("\nReview-length statistics:")
    print(
        dataframe[
            [
                "character_count",
                "word_count",
                "clean_word_count",
            ]
        ]
        .describe()
        .round(2)
    )

    print("\nAverage word count by sentiment:")
    print(
        dataframe.groupby("sentiment")["word_count"]
        .mean()
        .reindex(SENTIMENT_ORDER)
        .round(2)
    )


def save_sentiment_chart(dataframe: pd.DataFrame) -> None:
    """Save a bar chart showing the sentiment distribution."""
    counts = (
        dataframe["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER)
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    counts.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title("Distribution of Customer Review Sentiments")
    axis.set_xlabel("Sentiment")
    axis.set_ylabel("Number of Reviews")
    axis.tick_params(
        axis="x",
        rotation=0,
    )

    for position, count in enumerate(counts):
        axis.text(
            position,
            count,
            f"{count:,}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIRECTORY
        / "sentiment_distribution.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"\nSentiment chart saved to:\n{output_path}")


def save_review_length_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Save a histogram showing review word lengths."""
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.hist(
        dataframe["word_count"],
        bins=30,
    )

    axis.set_title("Distribution of Review Lengths")
    axis.set_xlabel("Number of Words")
    axis.set_ylabel("Number of Reviews")

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / "review_length_distribution.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Review-length chart saved to:\n{output_path}")


def main() -> None:
    """Run the exploratory data analysis."""
    dataframe = load_dataset()
    dataframe = add_review_length_features(dataframe)

    print_summary(dataframe)
    save_sentiment_chart(dataframe)
    save_review_length_chart(dataframe)


if __name__ == "__main__":
    main()