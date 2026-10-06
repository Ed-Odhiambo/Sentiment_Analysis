"""Tests for the trained sentiment prediction module."""

import pandas as pd
import pytest

from src.predict import SentimentPredictor


@pytest.fixture(scope="module")
def predictor() -> SentimentPredictor:
    """Load the trained predictor once for this test module."""
    return SentimentPredictor()


def test_predict_one_returns_valid_sentiment(
    predictor: SentimentPredictor,
) -> None:
    """A usable review should receive a valid sentiment."""
    result = predictor.predict_one(
        "The service was excellent."
    )

    assert result["sentiment"] in {
        "positive",
        "neutral",
        "negative",
    }

    assert result["processing_status"] == "processed"
    assert result["clean_review"]


def test_predict_many_keeps_all_rows(
    predictor: SentimentPredictor,
) -> None:
    """Batch prediction should preserve the number of reviews."""
    reviews = [
        "The staff were helpful.",
        "The flight was delayed.",
        "It was an ordinary journey.",
    ]

    results = predictor.predict_many(reviews)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == len(reviews)


def test_empty_review_is_skipped(
    predictor: SentimentPredictor,
) -> None:
    """Blank reviews should not be passed to the classifier."""
    results = predictor.predict_many(
        [
            "",
            "The service was good.",
        ]
    )

    assert (
        results.iloc[0]["processing_status"]
        == "skipped_empty"
    )

    assert pd.isna(
        results.iloc[0]["sentiment"]
    )

    assert (
        results.iloc[1]["processing_status"]
        == "processed"
    )


def test_predict_one_rejects_empty_review(
    predictor: SentimentPredictor,
) -> None:
    """Single-review prediction should reject blank input."""
    with pytest.raises(ValueError):
        predictor.predict_one("")