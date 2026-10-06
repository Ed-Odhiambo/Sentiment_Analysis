"""Tests for incoming-review automation utilities."""

import pandas as pd
import pytest

from src.automation import find_text_column
from src.notification import assess_negative_reviews


def test_find_review_column() -> None:
    """The standard review column should be detected."""
    dataframe = pd.DataFrame(
        {
            "review": [
                "Excellent service.",
            ]
        }
    )

    assert find_text_column(dataframe) == "review"


def test_find_feedback_column() -> None:
    """Alternative feedback column names should be detected."""
    dataframe = pd.DataFrame(
        {
            "feedback": [
                "The service was delayed.",
            ]
        }
    )

    assert find_text_column(dataframe) == "feedback"


def test_missing_text_column_raises_error() -> None:
    """A file without review text should be rejected."""
    dataframe = pd.DataFrame(
        {
            "customer_id": [
                1,
                2,
            ]
        }
    )

    with pytest.raises(ValueError):
        find_text_column(dataframe)


def test_negative_alert_is_triggered() -> None:
    """An alert should trigger when negative feedback is high."""
    dataframe = pd.DataFrame(
        {
            "predicted_sentiment": [
                "negative",
                "negative",
                "negative",
                "positive",
                "neutral",
            ],
            "processing_status": [
                "processed",
                "processed",
                "processed",
                "processed",
                "processed",
            ],
        }
    )

    alert = assess_negative_reviews(
        dataframe=dataframe,
        threshold_percentage=50.0,
        minimum_reviews=5,
    )

    assert alert.triggered is True
    assert alert.total_reviews == 5
    assert alert.negative_reviews == 3
    assert alert.negative_percentage == 60.0


def test_alert_requires_minimum_reviews() -> None:
    """Small batches should not trigger the alert."""
    dataframe = pd.DataFrame(
        {
            "predicted_sentiment": [
                "negative",
                "negative",
            ],
            "processing_status": [
                "processed",
                "processed",
            ],
        }
    )

    alert = assess_negative_reviews(
        dataframe=dataframe,
        threshold_percentage=50.0,
        minimum_reviews=5,
    )

    assert alert.triggered is False
    assert alert.total_reviews == 2